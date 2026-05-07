import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
from typing import Dict
from tqdm import tqdm
from model import DINOV2Analog, DINOV2Analog3D, DinoClassifier, DinoRegressor, DinoClassifier3D, DinoRegressor3D, DINOV2FineTune, DinoClassifierFromScratch, DINOV2Analog3DPretrained, DinoClassifier3DPretrained, DinoRegressor3DPretrained
from torcheval.metrics import R2Score


class DINOLoss(nn.Module):
    def __init__(self, device, output_dim=256, temp=0.1):
        super().__init__()
        self.temp = temp
        self.center = torch.zeros(output_dim).to(device)
        self.last_center = torch.zeros(output_dim).to(device)
        self.center_momentum = 0.9  # Make this adaptive
        self.iteration = 0

    def forward(self, student_out, teacher_out, return_components=False):
        # Student projections (log-softmax)
        student_centered = (student_out - self.center) / self.temp
        student_proj = F.log_softmax(student_centered, dim=-1)
        
        # Teacher projections (softmax) 
        teacher_centered = (teacher_out - self.center) / self.temp
        teacher_proj = F.softmax(teacher_centered, dim=-1)
        
        # Main consistency loss
        consistency_loss = -torch.sum(teacher_proj * student_proj, dim=-1).mean()
        
        if return_components:
            # Calculate additional loss components for detailed tracking
            # Temperature regularization (encourages moderate temperatures)
            temp_reg = torch.tensor(abs(self.temp - 0.1) * 0.01)
            
            # Center magnitude penalty (prevents center from growing too large)
            center_penalty = torch.norm(self.center) * 0.001
            
            # Sharpness penalty (encourages diverse predictions)
            entropy_teacher = -torch.sum(teacher_proj * torch.log(teacher_proj + 1e-8), dim=-1).mean()
            sharpness_penalty = torch.exp(-entropy_teacher) * 0.01
            
            return {
                'total_loss': consistency_loss + temp_reg + center_penalty + sharpness_penalty,
                'consistency_loss': consistency_loss,
                'temperature_reg': temp_reg,
                'center_penalty': center_penalty,
                'sharpness_penalty': sharpness_penalty,
                'teacher_entropy': entropy_teacher
            }
        
        return consistency_loss

    def update_center(self, teacher_out):
        self.last_center = self.center.clone()
        
        # Adaptive center momentum - higher momentum early, lower later
        self.iteration += 1
        adaptive_momentum = max(0.9 - (self.iteration / 10000) * 0.1, 0.8)
        
        self.center = (self.center * adaptive_momentum + 
                       teacher_out.mean(dim=0) * (1 - adaptive_momentum))
        
        # Add small noise to prevent getting stuck
        if self.iteration % 100 == 0:
            noise = torch.randn_like(self.center) * 0.01
            self.center = self.center + noise
        
        return torch.norm(self.center - self.last_center).item()


def compute_feature_similarity(features1: torch.Tensor, features2: torch.Tensor) -> float:
    """Compute cosine similarity between two sets of features."""
    features1 = F.normalize(features1, dim=1)
    features2 = F.normalize(features2, dim=1)
    similarity = torch.mm(features1, features2.t())
    return similarity.mean().item()


class UniversalTrainer:
    def __init__(self, config, run=None):
        self.config = config
        self.run = run
        # Initialize global step counter for iteration-wise logging
        self.global_step = 0
        self.id = run["sys/id"].fetch() if run is not None else "test_run"
        
        # Initialize best metrics tracking
        self.best_metrics = {'train': {}, 'test': {}}
        self.overfitting_detected = False
        self.overfitting_patience = 0
        self.max_overfitting_patience = 5
        self.run['patience'] = 1.0 - self.overfitting_patience / self.max_overfitting_patience
        # Handle device configuration with fallback
        device_config = config['training']['device']
        if device_config == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        elif device_config == 'cuda' and not torch.cuda.is_available():
            print("Warning: CUDA requested but not available. Using CPU.")
            self.device = torch.device('cpu')
        else:
            self.device = torch.device(device_config)
        print(f"Using device: {self.device}")
        
        # Initialize models
        if config['model']['type'] == 'dino':
            # Choose model based on mode and data type
            if config['model'].get('data_type') == '3d' and config['model']['mode'] == 'pretrained':
                self.student = DINOV2Analog3DPretrained(config).to(self.device)
                self.teacher = DINOV2Analog3DPretrained(config).to(self.device)
                
            elif config['model'].get('data_type') == '3d':
                self.student = DINOV2Analog3D(config).to(self.device)
                self.teacher = DINOV2Analog3D(config).to(self.device)
            elif config['model']['mode'] == 'fine_tune':
                self.student = DINOV2FineTune(config).to(self.device)
                self.teacher = DINOV2FineTune(config).to(self.device)
            else:
                self.student = DINOV2Analog(config).to(self.device)
                self.teacher = DINOV2Analog(config).to(self.device)
            run["model"] = str(self.student)
            # Initialize teacher with student weights
            for param_s, param_t in zip(self.student.parameters(), self.teacher.parameters()):
                param_t.data.copy_(param_s.data)
                param_t.requires_grad = False
                
            self.criterion = DINOLoss(
                self.device, 
                output_dim=config['model']['projection_head']['output_dim']
            ).to(self.device)
            self.optimizer = torch.optim.AdamW(
                self.student.parameters(),
                lr=config['training']['learning_rate'],
                weight_decay=config['training']['weight_decay']
            )
        elif config['model']['type'] == 'classifier':
            # Choose classifier based on mode and data type
            if config['model'].get('data_type') == '3d' and config['model']['mode'] == 'pretrained':
                self.model = DinoClassifier3DPretrained(config).to(self.device)
            elif config['model'].get('data_type') == '3d':
                self.model = DinoClassifier3D(config).to(self.device)
            elif config['model']['mode'] == 'from_scratch':
                self.model = DinoClassifierFromScratch(config).to(self.device)
            else:
                self.model = DinoClassifier(config).to(self.device)
            run["model"] = str(self.model)

            self.criterion = nn.CrossEntropyLoss()
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=config['training']['learning_rate'],
                weight_decay=config['training']['weight_decay']
            )
        else:
            # Choose 3D or 2D regressor based on data type and mode
            if config['model'].get('data_type') == '3d' and config['model']['mode'] == 'pretrained':
                self.model = DinoRegressor3DPretrained(config).to(self.device)
            elif config['model'].get('data_type') == '3d':
                self.model = DinoRegressor3D(config).to(self.device)
            else:
                self.model = DinoRegressor(config).to(self.device)
            run["model"] = str(self.model)
            self.criterion = nn.MSELoss()
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=config['training']['learning_rate'],
                weight_decay=config['training']['weight_decay']
            )
            
    def setup_model(self):
        if self.config['model']['type'] == 'dino':
            if self.config['model']['mode'] == 'from_scratch':
                if self.config['model'].get('data_type') == '3d':
                    self.student = DINOV2Analog3D(self.config).to(self.device)
                    self.teacher = DINOV2Analog3D(self.config).to(self.device)
                else:
                    self.student = DINOV2Analog(self.config).to(self.device)
                    self.teacher = DINOV2Analog(self.config).to(self.device)
                # Initialize teacher with student's parameters
                for param_t, param_s in zip(self.teacher.parameters(), self.student.parameters()):
                    param_t.data.copy_(param_s.data)
                    param_t.requires_grad = False
            elif self.config['model']['mode'] == 'pretrained' and self.config['model'].get('data_type') == '3d':
                self.student = DINOV2Analog3DPretrained(self.config).to(self.device)
                self.teacher = DINOV2Analog3DPretrained(self.config).to(self.device)
                # Initialize teacher with student's parameters
                for param_t, param_s in zip(self.teacher.parameters(), self.student.parameters()):
                    param_t.data.copy_(param_s.data)
                    param_t.requires_grad = False
            elif self.config['model']['mode'] == 'fine_tune':
                self.student = DINOV2FineTune(self.config).to(self.device)
                self.teacher = DINOV2FineTune(self.config).to(self.device)
                # Initialize teacher with student's parameters
                for param_t, param_s in zip(self.teacher.parameters(), self.student.parameters()):
                    param_t.data.copy_(param_s.data)
                    param_t.requires_grad = False
            else:
                raise NotImplementedError("Pretrained DINO not implemented yet")
                
        elif self.config['model']['type'] == 'classifier':
            if self.config['model'].get('data_type') == '3d':
                self.model = DinoClassifier3D(self.config).to(self.device)
            elif self.config['model']['mode'] == 'from_scratch':
                self.model = DinoClassifierFromScratch(self.config).to(self.device)
            else:
                self.model = DinoClassifier(self.config).to(self.device)
            
    def setup_optimizer(self):
        lr = self.config['training']['learning_rate']
        weight_decay = self.config['training']['weight_decay']
        
        if self.config['model']['type'] == 'dino':
            self.optimizer = optim.AdamW(self.student.parameters(), lr=lr, weight_decay=weight_decay)
            self.criterion = DINOLoss(self.device)
        else:
            self.optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
            self.criterion = nn.CrossEntropyLoss()
        
    def warmup_cosine_schedule(self, epoch: int) -> float:
        warmup_epochs = self.config['training']['warmup_epochs']
        num_epochs = self.config['training']['num_epochs']
        
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        
        # Cosine annealing with minimum learning rate
        cosine_factor = 0.5 * (1 + math.cos(math.pi * (epoch - warmup_epochs) / (num_epochs - warmup_epochs)))
        min_lr_ratio = 0.01  # Minimum LR is 1% of base LR
        
        return max(cosine_factor, min_lr_ratio)
    
    def cosine_restart_schedule(self, epoch: int) -> float:
        """Cosine annealing with warm restarts to prevent lr from getting too small"""
        warmup_epochs = self.config['training']['warmup_epochs']
        num_epochs = self.config['training']['num_epochs']
        
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        
        # Restart every 20 epochs after warmup
        restart_period = 20
        epoch_in_restart = (epoch - warmup_epochs) % restart_period
        
        cosine_factor = 0.5 * (1 + math.cos(math.pi * epoch_in_restart / restart_period))
        min_lr_ratio = 0.01  # Minimum LR is 1% of base LR
        
        return max(cosine_factor, min_lr_ratio)
        
    def train_step_dino(self, images: list) -> Dict[str, float]:
        images = [img.to(self.device) for img in images]
        
        # Use autocast only if CUDA is available
        if self.device.type == 'cuda':
            with torch.cuda.amp.autocast():
                # Get student outputs for all views
                student_outputs = [self.student(img) for img in images]
                student_projs = [out['projection'] for out in student_outputs]
                student_features = [out['cls_token'] for out in student_outputs]
                
                # Get teacher outputs for first view only
                with torch.no_grad():
                    teacher_outputs = [self.teacher(img) for img in images[:1]]
                    teacher_projs = [out['projection'] for out in teacher_outputs]
                    teacher_features = [out['cls_token'] for out in teacher_outputs]
                    
            # Calculate DINO loss with components (same as CUDA branch)
            loss_components = {'total_loss': 0, 'consistency_loss': 0, 'temperature_reg': 0, 
                             'center_penalty': 0, 'sharpness_penalty': 0, 'teacher_entropy': 0}
            
            for i, t_proj in enumerate(teacher_projs):
                for j, s_proj in enumerate(student_projs):
                    loss_parts = self.criterion(s_proj, t_proj, return_components=True)
                    for k, v in loss_parts.items():
                        loss_components[k] += v
            
            # Average across all projection pairs
            num_pairs = len(teacher_projs) * len(student_projs)
            for k in loss_components:
                loss_components[k] /= num_pairs
            
            loss = loss_components['total_loss']
        else:
            # CPU training without autocast
            # Get student outputs for all views
            student_outputs = [self.student(img) for img in images]
            student_projs = [out['projection'] for out in student_outputs]
            student_features = [out['cls_token'] for out in student_outputs]
            
            # Get teacher outputs for first view only
            with torch.no_grad():
                teacher_outputs = [self.teacher(img) for img in images[:1]]
                teacher_projs = [out['projection'] for out in teacher_outputs]
                teacher_features = [out['cls_token'] for out in teacher_outputs]
                
            # Calculate DINO loss with components
            loss_components = {'total_loss': 0, 'consistency_loss': 0, 'temperature_reg': 0, 
                             'center_penalty': 0, 'sharpness_penalty': 0, 'teacher_entropy': 0}
            
            for i, t_proj in enumerate(teacher_projs):
                for j, s_proj in enumerate(student_projs):
                    loss_parts = self.criterion(s_proj, t_proj, return_components=True)
                    for k, v in loss_parts.items():
                        loss_components[k] += v
            
            # Average across all projection pairs
            num_pairs = len(teacher_projs) * len(student_projs)
            for k in loss_components:
                loss_components[k] /= num_pairs
            
            loss = loss_components['total_loss']
            
        self.optimizer.zero_grad()
        loss.backward()
        
        # Add gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(self.student.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        
        # Adaptive teacher momentum - start high, gradually decrease
        iteration = getattr(self, 'iteration', 0)
        self.iteration = iteration + 1
        base_momentum = 0.996
        min_momentum = 0.99
        momentum_decay = 0.0001
        
        adaptive_momentum = max(
            base_momentum - (self.iteration * momentum_decay), 
            min_momentum
        )
        
        # Update teacher with adaptive EMA
        total_param_norm = 0
        with torch.no_grad():
            for param_s, param_t in zip(self.student.parameters(), self.teacher.parameters()):
                param_t.data = param_t.data * adaptive_momentum + param_s.data * (1 - adaptive_momentum)
                total_param_norm += torch.norm(param_t.data - param_s.data).item()
                
        # Update center and get center update magnitude
        center_update = self.criterion.update_center(torch.cat(teacher_projs))
        
        # Calculate feature similarities
        view_similarities = []
        for i in range(len(student_features)):
            for j in range(i + 1, len(student_features)):
                sim = compute_feature_similarity(student_features[i], student_features[j])
                view_similarities.append(sim)
        
        # Calculate projection norms
        student_proj_norm = torch.mean(torch.stack([torch.norm(p, dim=1).mean() for p in student_projs])).item()
        teacher_proj_norm = torch.mean(torch.stack([torch.norm(p, dim=1).mean() for p in teacher_projs])).item()
        
        metrics = {
            'loss': loss.item(),
            'consistency_loss': loss_components['consistency_loss'].item(),
            'temperature_reg': loss_components['temperature_reg'].item(),
            'center_penalty': loss_components['center_penalty'].item(),
            'sharpness_penalty': loss_components['sharpness_penalty'].item(),
            'teacher_entropy': loss_components['teacher_entropy'].item(),
            'teacher_momentum': adaptive_momentum,
            'center_update': center_update,
            'param_update_norm': total_param_norm,
            'student_proj_norm': student_proj_norm,
            'teacher_proj_norm': teacher_proj_norm,
            'view_similarity': sum(view_similarities) / len(view_similarities) if view_similarities else 0.0
        }
        
        return metrics
        
    def train_step_supervised(self, batch: tuple) -> Dict[str, float]:
        x, y = batch
        x, y = x.to(self.device), y.to(self.device).to(torch.long)
        
        # Ensure input is float32
        x = x.float()
        
        logits, _ = self.model(x)
        
        # Ensure logits are float32
        logits = logits.float()
        
        if self.config['model']['type'] == 'classifier':
            # Handle different label formats
                       # Simplified: assume labels are already class indices
            if len(y.shape) > 1:
                y = y.squeeze()
            y = y.long()
            
            loss = self.criterion(logits, y)
            acc = (logits.argmax(1) == y).float().mean()
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            return {'loss': loss.item(), 'accuracy': acc.item()}
        else:
            # Regression case - ensure float32
            y = y.float()
            mse_loss = self.criterion(logits, y)
            
            self.optimizer.zero_grad()
            mse_loss.backward()
            self.optimizer.step()
            
            # Calculate both MSE and R2 metrics
            r2_metric = R2Score()
            r2_metric.update(logits, y)
            r2_score = r2_metric.compute()
            
            # MSE is already calculated as the loss
            mse_score = mse_loss.item()
            
            # Return basic metrics (MAPE will be calculated epoch-wise)
            return {
                'loss': mse_score,  # Keep loss as MSE for compatibility
                'mse': mse_score,   # Explicit MSE metric
                'r2': r2_score.item() if hasattr(r2_score, 'item') else float(r2_score)
            }

    def val_step_supervised(self, batch: tuple) -> Dict[str, float]:
        """Validation step for supervised classification."""
        self.model.eval()
        with torch.no_grad():
            x, y = batch
            x, y = x.to(self.device), y.to(self.device).to(torch.long)
            
            logits, _ = self.model(x)
            loss = self.criterion(logits, y)
            
            if self.config['model']['type'] == 'classifier':
                acc = (logits.argmax(1) == y).float().mean()
                return {'val_loss': loss.item(), 'val_accuracy': acc.item()}
            else:
                # For regression, calculate MSE and R2 (MAPE will be calculated epoch-wise)
                r2_metric = R2Score()
                r2_metric.update(logits, y)
                r2_score = r2_metric.compute()
                
                return {
                    'val_loss': loss.item(),
                    'val_mse': loss.item(),  # MSE loss
                    'val_r2': r2_score.item() if hasattr(r2_score, 'item') else float(r2_score)
                }
        
    def train_epoch(self, dataloader, epoch: int) -> Dict[str, float]:
        if self.config['model']['type'] == 'dino':
            self.student.train()
        else:
            self.model.train()
            
        # Update learning rate
        lr = self.config['training']['learning_rate'] * self.warmup_cosine_schedule(epoch)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
            
        # Initialize metrics based on training type
        if self.config['model']['type'] == 'dino':
            metrics = {
                'loss': 0.0,
                'teacher_momentum': 0.0,
                'center_update': 0.0,
                'param_update_norm': 0.0,
                'student_proj_norm': 0.0,
                'teacher_proj_norm': 0.0,
                'view_similarity': 0.0
            }
        elif self.config['model']['type'] == 'classifier':
            metrics = {'loss': 0.0, 'accuracy': 0.0}
        elif self.config['model']['type'] == 'regressor':
            # Initialize basic metrics (MAPE will be calculated epoch-wise)
            metrics = {'loss': 0.0, 'mse': 0.0, 'r2': 0.0}
        
        progress = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{self.config['training']['num_epochs']}")
        for batch in progress:
            step_metrics = (
                self.train_step_dino(batch) if self.config['model']['type'] == 'dino'
                else self.train_step_supervised(batch)
            )
            
            # Iteration-wise logging to Neptune
            if self.run is not None:
                for k, v in step_metrics.items():
                    self.run[f"train/batch/{k}_train"].append(v)
                # Log learning rate for this step
                current_lr = self.optimizer.param_groups[0]['lr']
                self.run["train/batch/learning_rate_train"].append(current_lr)
            
            # Increment global step counter
            self.global_step += 1
            
            for k, v in step_metrics.items():
                metrics[k] += v
                    
            #progress.set_postfix({k: f"{v:.4f}" for k, v in step_metrics.items()})
            
        # Average metrics
        for k in metrics:
            metrics[k] /= len(dataloader)
                
        if self.run is not None:
            for k, v in metrics.items():
                self.run[f"train/{k}_train"].append(v)
            
        return metrics
        
    def save_checkpoint(self, epoch: int, metrics: Dict[str, float], path: str):
        checkpoint = {
            'epoch': epoch,
            'metrics': metrics,
            'config': self.config,
        }
        
        if self.config['model']['type'] == 'dino':
            checkpoint.update({
                'student_state_dict': self.student.state_dict(),
                'teacher_state_dict': self.teacher.state_dict(),
            })
        else:
            checkpoint.update({
                'model_state_dict': self.model.state_dict(),
            })
            
        checkpoint['optimizer_state_dict'] = self.optimizer.state_dict()
        torch.save(checkpoint, path)
        
    def validate_epoch(self, dataloader) -> Dict[str, float]:
        """Run validation epoch for supervised classification."""
        if self.config['model']['type'] not in ['classifier', 'regressor']:
            return {}
            
        self.model.eval()
        if self.config['model']['type'] == 'classifier':
            metrics = {'val_loss': 0.0, 'val_accuracy': 0.0}
            progress = tqdm(dataloader, desc="Validation")
            for batch in progress:
                step_metrics = self.val_step_supervised(batch)
                
                for k, v in step_metrics.items():
                    metrics[k] += v
                        
                #progress.set_postfix({k: f"{v:.4f}" for k, v in step_metrics.items()})
                
            # Average metrics
            for k in metrics:
                metrics[k] /= len(dataloader)
                    
            if self.run is not None:
                for k, v in metrics.items():
                    self.run[f"test/{k.replace('val_', '')}"].log(v)
                
            return metrics
        
        elif self.config['model']['type'] == 'regressor':
            all_preds = []
            all_targets = []
            total_loss = 0.0
            num_samples = 0
            progress = tqdm(dataloader, desc="Validation")
            
            for batch in progress:
                self.model.eval()
                with torch.no_grad():
                    x, y = batch
                    x, y = x.to(self.device), y.to(self.device)
                    
                    # Get step metrics
                    step_metrics = self.val_step_supervised(batch)
                    
                    # Collect predictions for epoch-wise calculations
                    preds, _ = self.model(x)
                    all_preds.append(preds.cpu())
                    all_targets.append(y.cpu())
                    
                    # Update totals
                    total_loss += step_metrics['val_loss'] * x.size(0)
                    num_samples += x.size(0)
                    
                    # Log batch-wise test metrics
                    if self.run is not None:
                        for k, v in step_metrics.items():
                            test_key = k.replace('val_', '') if k.startswith('val_') else k
                            self.run[f"train/batch/{test_key}_test"].append(v)
            
            # Calculate epoch-wise metrics
            all_preds = torch.cat(all_preds, dim=0)
            all_targets = torch.cat(all_targets, dim=0)
            
            # R2 score
            r2_metric = R2Score()
            r2_metric.update(all_preds, all_targets)
            r2_score = r2_metric.compute()
            
            # Calculate epoch-wise MAPE for each context variable
            mape_per_var = self._calculate_mape_per_variable(all_preds, all_targets)
            
            metrics = {
                'val_loss': total_loss / num_samples,
                'val_mse': total_loss / num_samples,
                'val_r2': r2_score.item() if hasattr(r2_score, 'item') else float(r2_score)
            }
            
            # Add MAPE for each context variable
            for i, mape_val in enumerate(mape_per_var):
                metrics[f'val_mape_var_{i}'] = mape_val.item() if hasattr(mape_val, 'item') else float(mape_val)
            
            # Log epoch-level test metrics with new structure
            if self.run is not None:
                for k, v in metrics.items():
                    test_key = k.replace('val_', '') if k.startswith('val_') else k
                    self.run[f"train/{test_key}_test"].append(v)
            
            return metrics

    def train(self, train_dataloader, val_dataloader=None) -> Dict[str, float]:
        """Train the model with overfitting detection and best metrics tracking."""
        # Get first metric from config for overfitting detection
        first_metric = self.config['metrics'][0]['name'] if 'metrics' in self.config else 'loss'
        
        patience = self.config.get('training', {}).get('early_stopping', {}).get('patience', 5)
        patience_counter = 0
        
        for epoch in range(self.config['training']['num_epochs']):
            print(f"\n=== Epoch {epoch + 1}/{self.config['training']['num_epochs']} ===")
            
            # Training epoch
            train_metrics = self.train_epoch(train_dataloader, epoch)
            
            # Validation epoch (test)
            test_metrics = {}
            if val_dataloader:
                test_metrics = self.validate_epoch(val_dataloader)
                
            # Update best metrics and log them
            self._update_and_log_best_metrics(train_metrics, test_metrics, first_metric)
            
            # Check for overfitting (first metric declines on test but grows on validation)
            if self._check_overfitting(train_metrics, test_metrics, first_metric):
                self.overfitting_patience += 1
                print(f"Overfitting detected! Patience: {self.overfitting_patience}/{self.max_overfitting_patience}")
                self.run['patience'] = 1.0 - self.overfitting_patience / self.max_overfitting_patience
                
                if self.overfitting_patience >= self.max_overfitting_patience:
                    print(f"Early stopping due to overfitting at epoch {epoch + 1}")
                    break
            else:
                self.overfitting_patience = 0
                self.run['patience'] = 1.0 - self.overfitting_patience / self.max_overfitting_patience
            
            # Save checkpoint
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(
                    epoch,
                    {**train_metrics, **test_metrics},
                    f"checkpoint_{self.config['model']['type']}_{self.config['model']['mode']}_{epoch + 1}_{self.id}.pth"
                )
            
            # Regular early stopping based on loss
            current_loss = test_metrics.get('val_loss', train_metrics['loss'])
            if current_loss < self.best_metrics['test'].get('loss', float('inf')):
                patience_counter = 0
                self.run['patience'] = 1.0 - self.overfitting_patience / self.max_overfitting_patience
                self.save_checkpoint(
                    epoch,
                    {**train_metrics, **test_metrics},
                    f"best_{self.config['model']['type']}_{self.config['model']['mode']}_{self.id}.pth"
                )
            else:
                patience_counter += 1
                self.run['patience'] = 1.0 - self.overfitting_patience / self.max_overfitting_patience
                
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch + 1}")
                break
                      
        return self.best_metrics
    
    def _update_and_log_best_metrics(self, train_metrics, test_metrics, first_metric):
        """Update and log best metrics."""
        # Update best training metrics
        for metric_name, value in train_metrics.items():
            if (metric_name not in self.best_metrics['train'] or 
                self._is_better_metric(value, self.best_metrics['train'][metric_name], metric_name)):
                self.best_metrics['train'][metric_name] = value
        
        # Update best test metrics
        for metric_name, value in test_metrics.items():
            test_key = metric_name.replace('val_', '') if metric_name.startswith('val_') else metric_name
            if (test_key not in self.best_metrics['test'] or 
                self._is_better_metric(value, self.best_metrics['test'][test_key], test_key)):
                self.best_metrics['test'][test_key] = value
        
        # Log best metrics to Neptune
        if self.run is not None:
            for metric_name, value in self.best_metrics['train'].items():
                self.run[f"train/best/{metric_name}_train"].append(value)
            
            for metric_name, value in self.best_metrics['test'].items():
                self.run[f"train/best/{metric_name}_test"].append(value)
                
            # Log feature-specific best metrics (MAPE variables)
            for metric_name, value in self.best_metrics['test'].items():
                if metric_name.startswith('mape_var_'):
                    feature_name = metric_name  # mape_var_0, mape_var_1, etc.
                    self.run[f"best/{feature_name}_test"].append(value)
    
    def _is_better_metric(self, new_value, best_value, metric_name):
        """Check if new metric value is better than current best."""
        # Metrics to minimize (lower is better)
        minimize_metrics = ['loss', 'mse', 'val_loss', 'val_mse'] + [f'mape_var_{i}' for i in range(10)] + [f'val_mape_var_{i}' for i in range(10)]
        
        if any(metric_name.endswith(m) or metric_name == m for m in minimize_metrics):
            return new_value < best_value
        else:
            # Metrics to maximize (higher is better) - r2, accuracy
            return new_value > best_value
    
    def _check_overfitting(self, train_metrics, test_metrics, first_metric):
        """Check for overfitting: first metric declines on test but grows on train/validation."""
        if not test_metrics:
            return False
            
        # Get the test version of the first metric
        test_metric_key = f'val_{first_metric}' if f'val_{first_metric}' in test_metrics else first_metric
        train_metric_key = first_metric
        
        if test_metric_key not in test_metrics or train_metric_key not in train_metrics:
            return False
        
        # Check if we have previous values to compare
        if not hasattr(self, 'prev_train_metric') or not hasattr(self, 'prev_test_metric'):
            self.prev_train_metric = train_metrics[train_metric_key]
            self.prev_test_metric = test_metrics[test_metric_key]
            return False
        
        # Check for overfitting pattern
        current_train = train_metrics[train_metric_key]
        current_test = test_metrics[test_metric_key]
        
        # For loss metrics (lower is better), overfitting = train improves but test gets worse
        if first_metric in ['loss', 'mse'] or first_metric.startswith('mape_'):
            train_improved = current_train < self.prev_train_metric
            test_degraded = current_test > self.prev_test_metric
        else:
            # For accuracy/r2 metrics (higher is better), overfitting = train improves but test gets worse
            train_improved = current_train > self.prev_train_metric
            test_degraded = current_test < self.prev_test_metric
        
        # Update previous values
        self.prev_train_metric = current_train
        self.prev_test_metric = current_test
        
        return train_improved and test_degraded
    
    def _calculate_mape_per_variable(self, predictions, targets):
        """Calculate MAPE (Mean Absolute Percentage Error) for each context variable."""
        # Avoid division by zero by adding small epsilon
        epsilon = 1e-8
        # Calculate percentage error: |pred - true| / (|true| + epsilon) * 100
        percentage_errors = torch.abs(predictions - targets) / (torch.abs(targets) + epsilon) * 100
        # Average across batch dimension to get MAPE per variable
        mape_per_var = torch.mean(percentage_errors, dim=0)
        return mape_per_var
