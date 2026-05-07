"""
Plotting utilities for BYOL training visualization.

Provides functions to visualize:
- Training and validation loss curves
- Learning rate schedule
- Target network momentum schedule
- Comparison plots
"""

import matplotlib.pyplot as plt
import numpy as np
import json
from pathlib import Path
import seaborn as sns

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 11


def load_history(history_path):
    """
    Load training history from JSON file.
    
    Args:
        history_path (str or Path): Path to training_history.json
    
    Returns:
        dict: Training history dictionary
    """
    with open(history_path, 'r') as f:
        history = json.load(f)
    return history


def plot_loss_curves(history, save_path=None, show=True):
    """
    Plot training and validation loss curves.
    
    Args:
        history (dict): Training history containing 'train_loss' and 'val_loss'
        save_path (str or Path, optional): Path to save the figure
        show (bool): Whether to display the plot
    """
    epochs = np.arange(1, len(history['train_loss']) + 1)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot losses
    ax.plot(epochs, history['train_loss'], label='Training Loss', 
            linewidth=2, marker='o', markersize=4, alpha=0.8)
    ax.plot(epochs, history['val_loss'], label='Validation Loss', 
            linewidth=2, marker='s', markersize=4, alpha=0.8)
    
    # Mark best validation loss
    best_epoch = np.argmin(history['val_loss']) + 1
    best_loss = np.min(history['val_loss'])
    ax.axvline(best_epoch, color='red', linestyle='--', alpha=0.5, 
               label=f'Best Val Loss (Epoch {best_epoch})')
    ax.plot(best_epoch, best_loss, 'r*', markersize=15)
    
    ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax.set_ylabel('Loss', fontsize=12, fontweight='bold')
    ax.set_title('BYOL Training and Validation Loss', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved loss curves to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_learning_rate(history, save_path=None, show=True):
    """
    Plot learning rate schedule.
    
    Args:
        history (dict): Training history containing 'learning_rate'
        save_path (str or Path, optional): Path to save the figure
        show (bool): Whether to display the plot
    """
    epochs = np.arange(1, len(history['learning_rate']) + 1)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.plot(epochs, history['learning_rate'], linewidth=2, color='green')
    ax.fill_between(epochs, history['learning_rate'], alpha=0.3, color='green')
    
    ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax.set_ylabel('Learning Rate', fontsize=12, fontweight='bold')
    ax.set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Use log scale if learning rate varies significantly
    lr_range = max(history['learning_rate']) / (min(history['learning_rate']) + 1e-10)
    if lr_range > 100:
        ax.set_yscale('log')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved learning rate plot to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_target_momentum(history, save_path=None, show=True):
    """
    Plot target network momentum (EMA decay rate) schedule.
    
    Args:
        history (dict): Training history containing 'target_momentum'
        save_path (str or Path, optional): Path to save the figure
        show (bool): Whether to display the plot
    """
    epochs = np.arange(1, len(history['target_momentum']) + 1)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.plot(epochs, history['target_momentum'], linewidth=2, color='purple')
    ax.fill_between(epochs, history['target_momentum'], alpha=0.3, color='purple')
    
    ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax.set_ylabel('Target Momentum (τ)', fontsize=12, fontweight='bold')
    ax.set_title('Target Network EMA Momentum Schedule', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0.99, 1.001])  # Focus on the relevant range
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved momentum plot to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_training_summary(history, save_path=None, show=True):
    """
    Create a comprehensive summary plot with all training metrics.
    
    Args:
        history (dict): Training history
        save_path (str or Path, optional): Path to save the figure
        show (bool): Whether to display the plot
    """
    epochs = np.arange(1, len(history['train_loss']) + 1)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('BYOL Training Summary', fontsize=16, fontweight='bold')
    
    # Plot 1: Loss curves
    ax = axes[0, 0]
    ax.plot(epochs, history['train_loss'], label='Training', 
            linewidth=2, marker='o', markersize=3, alpha=0.8)
    ax.plot(epochs, history['val_loss'], label='Validation', 
            linewidth=2, marker='s', markersize=3, alpha=0.8)
    best_epoch = np.argmin(history['val_loss']) + 1
    best_loss = np.min(history['val_loss'])
    ax.axvline(best_epoch, color='red', linestyle='--', alpha=0.5)
    ax.plot(best_epoch, best_loss, 'r*', markersize=15)
    ax.set_xlabel('Epoch', fontweight='bold')
    ax.set_ylabel('Loss', fontweight='bold')
    ax.set_title('Loss Curves', fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Loss difference (train - val)
    ax = axes[0, 1]
    loss_diff = np.array(history['train_loss']) - np.array(history['val_loss'])
    ax.plot(epochs, loss_diff, linewidth=2, color='orange')
    ax.axhline(0, color='black', linestyle='--', alpha=0.5)
    ax.fill_between(epochs, loss_diff, alpha=0.3, color='orange')
    ax.set_xlabel('Epoch', fontweight='bold')
    ax.set_ylabel('Loss Difference', fontweight='bold')
    ax.set_title('Training - Validation Loss', fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Learning rate
    ax = axes[1, 0]
    ax.plot(epochs, history['learning_rate'], linewidth=2, color='green')
    ax.fill_between(epochs, history['learning_rate'], alpha=0.3, color='green')
    ax.set_xlabel('Epoch', fontweight='bold')
    ax.set_ylabel('Learning Rate', fontweight='bold')
    ax.set_title('Learning Rate Schedule', fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Target momentum
    ax = axes[1, 1]
    ax.plot(epochs, history['target_momentum'], linewidth=2, color='purple')
    ax.fill_between(epochs, history['target_momentum'], alpha=0.3, color='purple')
    ax.set_xlabel('Epoch', fontweight='bold')
    ax.set_ylabel('Momentum (τ)', fontweight='bold')
    ax.set_title('Target Network Momentum', fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved training summary to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_smoothed_loss(history, window_size=10, save_path=None, show=True):
    """
    Plot smoothed loss curves using moving average.
    
    Args:
        history (dict): Training history
        window_size (int): Window size for moving average
        save_path (str or Path, optional): Path to save the figure
        show (bool): Whether to display the plot
    """
    def moving_average(data, window):
        """Compute moving average."""
        return np.convolve(data, np.ones(window)/window, mode='valid')
    
    epochs = np.arange(1, len(history['train_loss']) + 1)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot original losses with low alpha
    ax.plot(epochs, history['train_loss'], label='Training (raw)', 
            linewidth=1, alpha=0.3, color='blue')
    ax.plot(epochs, history['val_loss'], label='Validation (raw)', 
            linewidth=1, alpha=0.3, color='orange')
    
    # Plot smoothed losses
    if len(history['train_loss']) > window_size:
        train_smooth = moving_average(history['train_loss'], window_size)
        val_smooth = moving_average(history['val_loss'], window_size)
        smooth_epochs = epochs[window_size-1:]
        
        ax.plot(smooth_epochs, train_smooth, label=f'Training (smoothed, w={window_size})', 
                linewidth=2, color='blue')
        ax.plot(smooth_epochs, val_smooth, label=f'Validation (smoothed, w={window_size})', 
                linewidth=2, color='orange')
    
    ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax.set_ylabel('Loss', fontsize=12, fontweight='bold')
    ax.set_title('Smoothed Loss Curves', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved smoothed loss plot to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_loss_statistics(history, save_path=None, show=True):
    """
    Plot statistics of loss values (min, max, mean, std).
    
    Args:
        history (dict): Training history
        save_path (str or Path, optional): Path to save the figure
        show (bool): Whether to display the plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Training loss statistics
    ax = axes[0]
    train_losses = history['train_loss']
    stats = {
        'Min': min(train_losses),
        'Max': max(train_losses),
        'Mean': np.mean(train_losses),
        'Median': np.median(train_losses),
        'Std': np.std(train_losses)
    }
    
    bars = ax.bar(stats.keys(), stats.values(), color='skyblue', alpha=0.7, edgecolor='black')
    ax.set_ylabel('Loss Value', fontweight='bold')
    ax.set_title('Training Loss Statistics', fontweight='bold', fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}',
                ha='center', va='bottom', fontsize=10)
    
    # Validation loss statistics
    ax = axes[1]
    val_losses = history['val_loss']
    stats = {
        'Min': min(val_losses),
        'Max': max(val_losses),
        'Mean': np.mean(val_losses),
        'Median': np.median(val_losses),
        'Std': np.std(val_losses)
    }
    
    bars = ax.bar(stats.keys(), stats.values(), color='lightcoral', alpha=0.7, edgecolor='black')
    ax.set_ylabel('Loss Value', fontweight='bold')
    ax.set_title('Validation Loss Statistics', fontweight='bold', fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}',
                ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved loss statistics to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def create_all_plots(history_or_path, output_dir='plots', show=False):
    """
    Create all available plots and save them to a directory.
    
    Args:
        history_or_path: Either a history dict or path to training_history.json
        output_dir (str or Path): Directory to save plots
        show (bool): Whether to display plots
    """
    # Load history if path is provided
    if isinstance(history_or_path, (str, Path)):
        history = load_history(history_or_path)
    else:
        history = history_or_path
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Creating all plots...")
    print("=" * 80)
    
    # Generate all plots
    plot_loss_curves(
        history, 
        save_path=output_dir / 'loss_curves.png', 
        show=show
    )
    
    plot_learning_rate(
        history, 
        save_path=output_dir / 'learning_rate.png', 
        show=show
    )
    
    plot_target_momentum(
        history, 
        save_path=output_dir / 'target_momentum.png', 
        show=show
    )
    
    plot_training_summary(
        history, 
        save_path=output_dir / 'training_summary.png', 
        show=show
    )
    
    plot_smoothed_loss(
        history, 
        window_size=10, 
        save_path=output_dir / 'smoothed_loss.png', 
        show=show
    )
    
    plot_loss_statistics(
        history, 
        save_path=output_dir / 'loss_statistics.png', 
        show=show
    )
    
    print("=" * 80)
    print(f"All plots saved to {output_dir}")


def compare_runs(history_paths, labels, save_path=None, show=True):
    """
    Compare multiple training runs.
    
    Args:
        history_paths (list): List of paths to training_history.json files
        labels (list): List of labels for each run
        save_path (str or Path, optional): Path to save the figure
        show (bool): Whether to display the plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Training Runs Comparison', fontsize=16, fontweight='bold')
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(history_paths)))
    
    for idx, (path, label) in enumerate(zip(history_paths, labels)):
        history = load_history(path)
        epochs = np.arange(1, len(history['train_loss']) + 1)
        color = colors[idx]
        
        # Training loss
        axes[0].plot(epochs, history['train_loss'], label=label, 
                    linewidth=2, alpha=0.8, color=color)
        
        # Validation loss
        axes[1].plot(epochs, history['val_loss'], label=label, 
                    linewidth=2, alpha=0.8, color=color)
    
    axes[0].set_xlabel('Epoch', fontweight='bold')
    axes[0].set_ylabel('Loss', fontweight='bold')
    axes[0].set_title('Training Loss', fontweight='bold')
    axes[0].legend(loc='best')
    axes[0].grid(True, alpha=0.3)
    
    axes[1].set_xlabel('Epoch', fontweight='bold')
    axes[1].set_ylabel('Loss', fontweight='bold')
    axes[1].set_title('Validation Loss', fontweight='bold')
    axes[1].legend(loc='best')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved comparison plot to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


if __name__ == "__main__":
    # Example usage
    print("BYOL Plotting Utilities")
    print("=" * 80)
    print("\nThis module provides various plotting functions for BYOL training.")
    print("\nExample usage:")
    print("""
    from byol.plotting import create_all_plots, load_history, plot_loss_curves
    
    # Create all plots from a checkpoint directory
    create_all_plots(
        'checkpoints/training_history.json',
        output_dir='plots',
        show=False
    )
    
    # Or plot individual metrics
    history = load_history('checkpoints/training_history.json')
    plot_loss_curves(history, save_path='loss.png', show=True)
    plot_learning_rate(history, save_path='lr.png', show=True)
    
    # Compare multiple runs
    compare_runs(
        history_paths=['run1/training_history.json', 'run2/training_history.json'],
        labels=['Run 1', 'Run 2'],
        save_path='comparison.png',
        show=True
    )
    """)
    print("=" * 80)

