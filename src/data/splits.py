"""
MC-UAV-Bench Dataset Splits Module
"""

import json
import random
from pathlib import Path
from typing import Dict, List
from collections import defaultdict


def create_stratified_splits(
    samples: List[Dict],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
    stratify_by: List[str] = None
) -> Dict[str, List[Dict]]:
    """
    Create stratified train/val/test splits.
    """
    if stratify_by is None:
        stratify_by = ['category', 'difficulty']
    
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
    
    random.seed(seed)
    
    # Group by stratification key
    groups = defaultdict(list)
    for s in samples:
        key = tuple(s.get(k, 'unknown') for k in stratify_by)
        groups[key].append(s)
    
    train_set, val_set, test_set = [], [], []
    
    for key, group in groups.items():
        random.shuffle(group)
        n = len(group)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        
        train_set.extend(group[:n_train])
        val_set.extend(group[n_train:n_train + n_val])
        test_set.extend(group[n_train + n_val:])
    
    random.shuffle(train_set)
    random.shuffle(val_set)
    random.shuffle(test_set)
    
    return {
        'train': train_set,
        'val': val_set,
        'test': test_set
    }


def save_splits(
    splits: Dict[str, List[Dict]],
    output_dir: str,
    prefix: str = "mc_uav_bench"
) -> Dict[str, str]:
    """Save splits to JSON files"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    saved_paths = {}
    
    for split_name, samples in splits.items():
        filepath = output_path / f"{prefix}_{split_name}.json"
        
        with open(filepath, 'w') as f:
            json.dump({
                'split': split_name,
                'count': len(samples),
                'samples': samples
            }, f, indent=2)
        
        saved_paths[split_name] = str(filepath)
        print(f"✓ Saved {split_name}: {len(samples)} samples → {filepath}")
    
    return saved_paths


def load_splits(split_dir: str, prefix: str = "mc_uav_bench") -> Dict[str, List[Dict]]:
    """Load splits from directory"""
    split_path = Path(split_dir)
    splits = {}
    
    for split_name in ['train', 'val', 'test']:
        filepath = split_path / f"{prefix}_{split_name}.json"
        if filepath.exists():
            with open(filepath, 'r') as f:
                data = json.load(f)
            splits[split_name] = data['samples']
            print(f"✓ Loaded {split_name}: {len(splits[split_name])} samples")
    
    return splits


def verify_splits(splits: Dict[str, List[Dict]]) -> Dict:
    """Verify split quality"""
    report = {
        'counts': {},
        'category_distribution': {},
        'difficulty_distribution': {},
        'overlap_check': True
    }
    
    for split_name, samples in splits.items():
        report['counts'][split_name] = len(samples)
    
    for split_name, samples in splits.items():
        cat_counts = defaultdict(int)
        for s in samples:
            cat_counts[s['category']] += 1
        report['category_distribution'][split_name] = dict(cat_counts)
    
    for split_name, samples in splits.items():
        diff_counts = defaultdict(int)
        for s in samples:
            diff_counts[s['difficulty']] += 1
        report['difficulty_distribution'][split_name] = dict(diff_counts)
    
    all_ids = {}
    for split_name, samples in splits.items():
        for s in samples:
            sample_id = s.get('id', s.get('scene_token', ''))
            if sample_id in all_ids:
                report['overlap_check'] = False
            all_ids[sample_id] = split_name
    
    return report


def print_split_statistics(splits: Dict[str, List[Dict]]):
    """Print detailed statistics"""
    print("\n" + "="*60)
    print("SPLIT STATISTICS")
    print("="*60)
    
    total = sum(len(s) for s in splits.values())
    
    for split_name, samples in splits.items():
        pct = 100 * len(samples) / total if total > 0 else 0
        print(f"\n{split_name.upper()} ({len(samples)} samples, {pct:.1f}%)")
        print("-"*40)
        
        cat_counts = defaultdict(int)
        diff_counts = defaultdict(int)
        safety_count = 0
        
        for s in samples:
            cat_counts[s.get('category', 'unknown')] += 1
            diff_counts[s.get('difficulty', 'unknown')] += 1
            if s.get('category') in [
                "Landing Area Detection", "Entity Detection", 
                "Obstacle Avoidance", "Weather-Edge Cases"
            ]:
                safety_count += 1
        
        print("  By Category:")
        for cat, count in sorted(cat_counts.items()):
            print(f"    {cat}: {count}")
        
        print("  By Difficulty:")
        for diff in ['easy', 'medium', 'hard']:
            print(f"    {diff}: {diff_counts.get(diff, 0)}")
        
        print(f"  Safety-Critical: {safety_count}")
