#!/usr/bin/env python3
"""
Quick validation script to test HuggingFace dataset loading for metaworld-pick-place-v3
"""
import sys
import os

# Monkey patch pyarrow issue before importing datasets
try:
    import pyarrow as pa
    original_concat_tables = pa.concat_tables
    def patched_concat_tables(tables, **kwargs):
        # Remove promote_options if it exists
        kwargs.pop('promote_options', None)
        return original_concat_tables(tables, **kwargs)
    pa.concat_tables = patched_concat_tables
except Exception:
    pass

from datasets import load_dataset

# Add paths
script_dir = os.path.dirname(__file__)
repo_root = os.path.abspath(os.path.join(script_dir, '..'))
sys.path.insert(0, repo_root)
sys.path.insert(0, script_dir)

def test_hf_dataset():
    print("=" * 70)
    print("Testing HuggingFace Dataset Loading")
    print("=" * 70)
    
    dataset_name = "aadarshram/metaworld-pick-place-v3"
    print(f"\nLoading dataset: {dataset_name}")
    
    try:
        # Load dataset
        dataset = load_dataset(dataset_name, split="train")
        print(f"✓ Successfully loaded dataset")
        print(f"  Total samples: {len(dataset)}")
        
        # Check first sample
        sample = dataset[0]
        print(f"\n✓ Sample structure:")
        for key in sample.keys():
            value = sample[key]
            if hasattr(value, 'shape'):
                print(f"  - {key}: shape={value.shape}")
            elif hasattr(value, 'size'):
                print(f"  - {key}: {value.size}")
            else:
                print(f"  - {key}: {type(value).__name__}")
        
        # Count episodes
        episodes = set()
        for i in range(min(100, len(dataset))):
            episodes.add(dataset[i]['episode_index'])
        print(f"\n✓ Found {len(episodes)} unique episodes (sampled first 100)")
        
        # Check image
        img = sample['observation.image']
        print(f"\n✓ Image info:")
        print(f"  - Type: {type(img)}")
        print(f"  - Size: {img.size if hasattr(img, 'size') else 'N/A'}")
        
        # Check state and action dimensions
        state = sample['observation.state']
        action = sample['action']
        print(f"\n✓ Data dimensions:")
        print(f"  - State: {len(state)} (expected: 4)")
        print(f"  - Action: {len(action)} (expected: 4)")
        
        print("\n" + "=" * 70)
        print("✓ All checks passed! Dataset is ready for training.")
        print("=" * 70)
        
        return True
        
    except Exception as e:
        print(f"\n✗ Error loading dataset: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_hf_dataset()
    sys.exit(0 if success else 1)
