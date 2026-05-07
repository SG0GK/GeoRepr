"""
Test script to verify BYOL training pipeline with DataCubes dataset.

This script tests each component step-by-step to ensure everything works.
"""

import os
import sys
import torch
import yaml
from pathlib import Path

# Add paths
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))


def test_imports():
    """Test that all imports work."""
    print("=" * 80)
    print("TEST 1: Imports")
    print("=" * 80)
    
    try:
        from byol.data_parce.data import split_dataset
        print("✓ Can import split_dataset")
        
        from byol.data_parce.byol_wrapper import (
            BYOLDataCubesWrapper,
            prepare_byol_dataloaders_from_datacubes
        )
        print("✓ Can import BYOL wrapper")
        
        from byol.augmentation import get_medium_augmentation
        print("✓ Can import augmentation")
        
        from byol.trainer import BYOLTrainer
        print("✓ Can import trainer")
        
        from byol import BYOL
        print("✓ Can import BYOL model")
        
        print("\n✓ All imports successful!\n")
        return True
    except Exception as e:
        print(f"\n✗ Import error: {e}\n")
        return False


def test_config_loading(config_path):
    """Test config loading."""
    print("=" * 80)
    print("TEST 2: Config Loading")
    print("=" * 80)
    
    try:
        if not os.path.exists(config_path):
            print(f"Config file not found: {config_path}")
            print("Skipping config test. Please create a config file.")
            return None
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        print(f"✓ Config loaded from {config_path}")
        print(f"  Data path: {config.get('data', {}).get('data_path', 'N/A')}")
        print(f"  Batch size: {config.get('batch_size', 'N/A')}")
        print(f"  Max epochs: {config.get('training', {}).get('max_epochs', 'N/A')}")
        
        print("\n✓ Config loading successful!\n")
        return config
    except Exception as e:
        print(f"\n✗ Config loading error: {e}\n")
        return None


def test_mock_dataset():
    """Test with mock dataset."""
    print("=" * 80)
    print("TEST 3: Mock Dataset (DataCubes simulation)")
    print("=" * 80)
    
    try:
        from torch.utils.data import Dataset
        from byol.data_parce.byol_wrapper import BYOLDataCubesWrapper
        from byol.augmentation import get_light_augmentation
        
        # Create mock dataset that mimics DataCubesWithStatsAndVerticalWells
        class MockDataCubes(Dataset):
            def __init__(self, n=10):
                self.n = n
            
            def __len__(self):
                return self.n
            
            def __getitem__(self, idx):
                # Return (X, X_C, y) like the real dataset
                X = torch.randn(3, 150, 144, 80)
                X_C = torch.randn(4, 150, 144, 80)
                y = torch.randn(10)
                return X, X_C, y
        
        # Create mock dataset
        mock_dataset = MockDataCubes(20)
        print(f"✓ Created mock dataset with {len(mock_dataset)} samples")
        
        # Test wrapper
        transform = get_light_augmentation(crop_size=(96, 96, 48))
        wrapped = BYOLDataCubesWrapper(mock_dataset, transform=transform)
        print("✓ Wrapped dataset with BYOL wrapper")
        
        # Test single item
        item = wrapped[0]
        print(f"✓ Got single item:")
        print(f"  Type: {type(item)}")
        print(f"  View1 shape: {item['view1'].shape}")
        print(f"  View2 shape: {item['view2'].shape}")
        
        # Test that views are different
        diff = (item['view1'] - item['view2']).abs().mean()
        assert diff > 0, "Views should be different!"
        print(f"  Mean diff: {diff:.4f}")
        
        # Test dataloader
        from torch.utils.data import DataLoader
        loader = DataLoader(wrapped, batch_size=4, shuffle=False)
        batch = next(iter(loader))
        print(f"✓ Got batch:")
        print(f"  View1 batch shape: {batch['view1'].shape}")
        print(f"  View2 batch shape: {batch['view2'].shape}")
        
        print("\n✓ Mock dataset test successful!\n")
        return True
    except Exception as e:
        print(f"\n✗ Mock dataset test error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_model_creation():
    """Test BYOL model creation."""
    print("=" * 80)
    print("TEST 4: BYOL Model Creation")
    print("=" * 80)
    
    try:
        from byol import BYOL
        
        model_config = {
            'encoder_3d': {
                'in_channels': 3,
                'block_out_channels': [16, 32, 64],
                'block_out_types': ['down', 'down', 'same'],
                'num_groups': 16,
                'dropout': 0.01,
                'scales_w': [2, 2],
                'scales_h': [2, 2],
                'scales_d': [2, 2]
            }
        }
        
        model = BYOL(model_config, projection_dim=256, hidden_dim=4096)
        print("✓ Created BYOL model")
        
        # Test forward pass with dummy data
        view1 = torch.randn(2, 3, 64, 64, 32)
        view2 = torch.randn(2, 3, 64, 64, 32)
        
        outputs = model(view1, view2)
        print(f"✓ Forward pass successful:")
        print(f"  pred1 shape: {outputs['pred1'].shape}")
        print(f"  pred2 shape: {outputs['pred2'].shape}")
        print(f"  proj1 shape: {outputs['proj1'].shape}")
        print(f"  proj2 shape: {outputs['proj2'].shape}")
        
        # Test loss computation
        from byol import compute_byol_loss
        loss = compute_byol_loss(outputs)
        print(f"✓ Loss computation successful: {loss.item():.4f}")
        
        print("\n✓ Model creation test successful!\n")
        return True
    except Exception as e:
        print(f"\n✗ Model creation test error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_training_pipeline_mock():
    """Test training pipeline with mock data."""
    print("=" * 80)
    print("TEST 5: Training Pipeline (Mock Data, 1 Epoch)")
    print("=" * 80)
    
    try:
        from torch.utils.data import Dataset, DataLoader
        from byol.data_parce.byol_wrapper import BYOLDataCubesWrapper
        from byol.augmentation import get_light_augmentation
        from byol.trainer import BYOLTrainer
        
        # Create mock dataset
        class MockDataCubes(Dataset):
            def __init__(self, n=20):
                self.n = n
            
            def __len__(self):
                return self.n
            
            def __getitem__(self, idx):
                X = torch.randn(3, 96, 96, 48)
                X_C = torch.randn(4, 96, 96, 48)
                y = torch.randn(10)
                return X, X_C, y
        
        # Create datasets
        train_dataset = MockDataCubes(20)
        val_dataset = MockDataCubes(5)
        print("✓ Created mock datasets")
        
        # Wrap for BYOL
        transform = get_light_augmentation(crop_size=(64, 64, 32))
        train_wrapped = BYOLDataCubesWrapper(train_dataset, transform=transform)
        val_wrapped = BYOLDataCubesWrapper(val_dataset, transform=transform)
        print("✓ Wrapped datasets")
        
        # Create dataloaders
        train_loader = DataLoader(train_wrapped, batch_size=4, shuffle=True)
        val_loader = DataLoader(val_wrapped, batch_size=4, shuffle=False)
        print("✓ Created dataloaders")
        
        # Model config
        model_config = {
            'encoder_3d': {
                'in_channels': 3,
                'block_out_channels': [16, 32],
                'block_out_types': ['down', 'same'],
                'num_groups': 8,
                'dropout': 0.01,
                'scales_w': [2],
                'scales_h': [2],
                'scales_d': [2]
            }
        }
        
        # Create trainer
        trainer = BYOLTrainer(
            model_config=model_config,
            train_loader=train_loader,
            val_loader=val_loader,
            device='cpu',  # Use CPU for testing
            learning_rate=0.01,
            projection_dim=128,
            hidden_dim=256,
            max_epochs=1,  # Just 1 epoch for testing
            warmup_epochs=0,
            checkpoint_dir='test_checkpoints',
            log_interval=1,
            save_interval=1
        )
        print("✓ Created trainer")
        
        # Train for 1 epoch
        print("\nTraining for 1 epoch (this may take a moment)...")
        trainer.train()
        
        print("\n✓ Training pipeline test successful!\n")
        
        # Clean up test checkpoints
        import shutil
        if os.path.exists('test_checkpoints'):
            shutil.rmtree('test_checkpoints')
            print("✓ Cleaned up test checkpoints\n")
        
        return True
    except Exception as e:
        print(f"\n✗ Training pipeline test error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "=" * 80)
    print("BYOL DataCubes Pipeline Test Suite")
    print("=" * 80 + "\n")
    
    results = {}
    
    # Test 1: Imports
    results['imports'] = test_imports()
    
    # Test 2: Config (optional)
    config_path = 'byol/config_datacubes_example.yaml'
    results['config'] = test_config_loading(config_path) is not None
    
    # Test 3: Mock dataset
    results['mock_dataset'] = test_mock_dataset()
    
    # Test 4: Model creation
    results['model'] = test_model_creation()
    
    # Test 5: Training pipeline (optional, takes time)
    print("Run training pipeline test? (1 epoch, may take 1-2 minutes)")
    run_training = input("Enter 'y' to run, or press Enter to skip: ").lower() == 'y'
    if run_training:
        results['training'] = test_training_pipeline_mock()
    else:
        print("Skipping training pipeline test.\n")
        results['training'] = None
    
    # Summary
    print("=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    
    for test_name, result in results.items():
        if result is None:
            status = "⊘ SKIPPED"
        elif result:
            status = "✓ PASSED"
        else:
            status = "✗ FAILED"
        print(f"{status:12} - {test_name}")
    
    print("=" * 80)
    
    # Overall result
    failed_tests = [name for name, result in results.items() 
                   if result is not None and not result]
    
    if not failed_tests:
        print("\n✓ All tests passed!")
        print("\nYou're ready to train BYOL on your DataCubes dataset!")
        print("Next steps:")
        print("1. Create your config file (copy config_datacubes_example.yaml)")
        print("2. Update data paths and parameters")
        print("3. Run: python -m byol.train_byol --config your_config.yaml")
    else:
        print(f"\n✗ {len(failed_tests)} test(s) failed: {', '.join(failed_tests)}")
        print("Please check the errors above and fix any issues.")
    
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()

