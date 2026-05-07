import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import argparse
from tqdm import tqdm
from utils import read_yaml
from img_datasets import LatentDataset, LatentClassesDataset
from img_transforms import MultiCropTensorTransform, TensorTransform
from model import DINOV2Analog, DinoClassifier
from outdated.ssl_train import DINOLoss, warmup_cosine_schedule
from outdated.class_train import class_train


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str)
    args = parser.parse_args()
    config = read_yaml(args.config)
    # Handle device configuration with fallback
    device_config = config['device']
    if device_config == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    elif device_config == 'cuda' and not torch.cuda.is_available():
        print("Warning: CUDA requested but not available. Using CPU.")
        device = torch.device('cpu')
    else:
        device = torch.device(device_config)
    print(f"Using device: {device}")

    lr = config['learning_rate']
    num_epochs = config['n_epochs']
    if config['train_method'] == 'supervised':
        transform = TensorTransform(target_size=config['img_size'])
        data = LatentClassesDataset(config['data_path'], config['target_path'], transform=transform)
        dataloader = DataLoader(data, batch_size=config['batch_size'], shuffle=True)
        model = DinoClassifier()
        model = model.to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        for epoch in range(num_epochs):
            loss, acc = class_train(model, dataloader, criterion, optimizer, device)
            print(f"Epoch {epoch + 1}: Loss={loss:.4f}, Accuracy={acc:.4f}")
            if (epoch + 1) % 5 == 0:
                torch.save(model.state_dict(), f"checkpoint_epoch_{epoch + 1}.pth")

    elif config['train_method'] == 'self-supervised':
        transform = MultiCropTensorTransform(
            global_size=config['img_size'], local_size=config['crop_size'], num_channels=config['num_channels']
        )
        data = LatentDataset(config['data_path'], transform=transform)
        dataloader = DataLoader(data, batch_size=config['batch_size'], shuffle=True)
        student = DINOV2Analog(base_img_size=140, patch_size=14).to(device)
        teacher = DINOV2Analog(base_img_size=140, patch_size=14).to(device)
        dino_loss = DINOLoss(device).to(device)
        optimizer = torch.optim.AdamW(student.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
        scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None
        MOMENTUM_TEACHER = config['momentum']
        for epoch in range(num_epochs):
            student.train()
            total_loss = 0
            progress = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}")

            lr = lr * warmup_cosine_schedule(epoch)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            for images in progress:
                images = [img.to(device) for img in images]  # Multiple crops

                # Use autocast only if CUDA is available
                if device.type == 'cuda':
                    with torch.cuda.amp.autocast():
                        student_projs = [student(img)['projection'] for img in images]

                        with torch.no_grad():
                            teacher_projs = [teacher(img)['projection'] for img in images[:1]]

                        loss = 0
                        for t_proj in teacher_projs:
                            for s_proj in student_projs:
                                loss += dino_loss(s_proj, t_proj)
                        loss /= len(teacher_projs) * len(student_projs)
                else:
                    # CPU training without autocast
                    student_projs = [student(img)['projection'] for img in images]

                    with torch.no_grad():
                        teacher_projs = [teacher(img)['projection'] for img in images[:1]]

                    loss = 0
                    for t_proj in teacher_projs:
                        for s_proj in student_projs:
                            loss += dino_loss(s_proj, t_proj)
                    loss /= len(teacher_projs) * len(student_projs)

                optimizer.zero_grad()
                if scaler:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

                with torch.no_grad():
                    for param_s, param_t in zip(student.parameters(), teacher.parameters()):
                        param_t.data = param_t.data * MOMENTUM_TEACHER + param_s.data * (1 - MOMENTUM_TEACHER)

                dino_loss.update_center(torch.cat(teacher_projs))

                total_loss += loss.item()
                progress.set_postfix({"loss": f"{loss.item():.4f}", "lr": f"{lr:.2e}"})

            if (epoch + 1) % 10 == 0:
                torch.save({
                    'student': student.state_dict(),
                    'teacher': teacher.state_dict(),
                    'optimizer': optimizer.state_dict(),
                }, f"checkpoint_epoch_{epoch + 1}.pth")

    else:
        raise ValueError('train_method should be supervised or self-supervised')

