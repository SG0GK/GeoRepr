import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List


def detect_loss_plateau(loss_history: List[float], window_size: int = 50, 
                       threshold: float = 0.001) -> bool:
    """
    Detect if loss has plateaued by checking if the standard deviation
    of recent losses is below a threshold.
    """
    if len(loss_history) < window_size:
        return False
    
    recent_losses = loss_history[-window_size:]
    loss_std = np.std(recent_losses)
    
    return loss_std < threshold


def compute_gradient_norm(model: nn.Module) -> float:
    """Compute the L2 norm of gradients for debugging purposes."""
    total_norm = 0.0
    for param in model.parameters():
        if param.grad is not None:
            param_norm = param.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    return total_norm ** 0.5


def adjust_temperature_dynamically(current_temp: float, loss_history: List[float], 
                                  min_temp: float = 0.07, max_temp: float = 0.3) -> float:
    """
    Dynamically adjust temperature based on loss progression.
    If loss is plateauing, increase temperature to soften the distribution.
    """
    if len(loss_history) < 10:
        return current_temp
    
    # Check if loss is plateauing
    recent_losses = loss_history[-10:]
    loss_std = np.std(recent_losses)
    
    if loss_std < 0.001:  # Loss is plateauing
        # Increase temperature to soften distribution
        new_temp = min(current_temp * 1.1, max_temp)
    else:
        # Decrease temperature for better convergence
        new_temp = max(current_temp * 0.99, min_temp)
    
    return new_temp


def add_weight_noise(model: nn.Module, noise_std: float = 0.01):
    """
    Add small amount of noise to model weights to escape local minima.
    """
    with torch.no_grad():
        for param in model.parameters():
            noise = torch.randn_like(param) * noise_std
            param.add_(noise)


def reset_optimizer_state(optimizer: torch.optim.Optimizer):
    """
    Reset optimizer state to help escape plateaus.
    """
    optimizer.state = {}


class LossPlateauDetector:
    """
    Class to detect and handle loss plateaus during training.
    """
    
    def __init__(self, window_size: int = 100, threshold: float = 0.001,
                 patience: int = 5):
        self.window_size = window_size
        self.threshold = threshold
        self.patience = patience
        self.loss_history = []
        self.plateau_count = 0
        
    def update(self, loss: float) -> Dict[str, bool]:
        """
        Update with new loss value and return actions to take.
        """
        self.loss_history.append(loss)
        
        # Keep only recent history
        if len(self.loss_history) > self.window_size * 2:
            self.loss_history = self.loss_history[-self.window_size:]
        
        is_plateau = detect_loss_plateau(self.loss_history, 
                                        self.window_size, 
                                        self.threshold)
        
        if is_plateau:
            self.plateau_count += 1
        else:
            self.plateau_count = 0
            
        actions = {
            'add_noise': self.plateau_count >= self.patience,
            'reset_optimizer': self.plateau_count >= self.patience * 2,
            'adjust_temperature': is_plateau,
            'is_plateau': is_plateau
        }
        
        return actions


def create_advanced_dino_loss(device: torch.device, output_dim: int = 256, 
                             temp: float = 0.1):
    """
    Create an advanced DINO loss with dynamic temperature adjustment.
    """
    class AdvancedDINOLoss(nn.Module):
        def __init__(self, device, output_dim=256, temp=0.1):
            super().__init__()
            self.temp = temp
            self.initial_temp = temp
            self.center = torch.zeros(output_dim).to(device)
            self.last_center = torch.zeros(output_dim).to(device)
            self.loss_history = []
            
        def forward(self, student_out, teacher_out):
            # Dynamic temperature adjustment
            if len(self.loss_history) > 0:
                self.temp = adjust_temperature_dynamically(
                    self.temp, self.loss_history
                )
            
            student_proj = torch.nn.functional.log_softmax(
                (student_out - self.center) / self.temp, dim=-1
            )
            teacher_proj = torch.nn.functional.softmax(
                (teacher_out - self.center) / self.temp, dim=-1
            )
            
            loss = -torch.sum(teacher_proj * student_proj, dim=-1).mean()
            self.loss_history.append(loss.item())
            
            # Keep only recent history
            if len(self.loss_history) > 100:
                self.loss_history = self.loss_history[-50:]
            
            return loss
            
        def update_center(self, teacher_out):
            self.last_center = self.center.clone()
            
            # More conservative center update
            momentum = 0.9
            teacher_mean = teacher_out.mean(dim=0)
            self.center = (self.center * momentum +
                           teacher_mean * (1 - momentum))
            
            return torch.norm(self.center - self.last_center).item()
    
    return AdvancedDINOLoss(device, output_dim, temp)


def get_training_diagnostics(model: nn.Module, loss_history: List[float]) -> Dict:
    """
    Get comprehensive training diagnostics to identify issues.
    """
    diagnostics = {}
    
    # Gradient diagnostics
    grad_norm = compute_gradient_norm(model)
    diagnostics['gradient_norm'] = grad_norm
    diagnostics['gradient_exploding'] = grad_norm > 10.0
    diagnostics['gradient_vanishing'] = grad_norm < 0.01
    
    # Loss diagnostics
    if len(loss_history) >= 10:
        recent_losses = loss_history[-10:]
        diagnostics['loss_std'] = np.std(recent_losses)
        diagnostics['loss_plateau'] = diagnostics['loss_std'] < 0.001
        loss_trend = np.polyfit(range(len(recent_losses)),
                                recent_losses, 1)[0]
        diagnostics['loss_trend'] = loss_trend
    
    # Parameter diagnostics
    param_norms = []
    for param in model.parameters():
        if param.data is not None:
            param_norms.append(param.data.norm().item())
    
    diagnostics['avg_param_norm'] = np.mean(param_norms)
    diagnostics['param_norm_std'] = np.std(param_norms)
    
    return diagnostics 