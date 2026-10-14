"""
MC-UAV-Bench Dataset Validator
"""

import re
from pathlib import Path
from typing import Dict, List
from collections import defaultdict


VALID_CAMERAS = ['CAM_FRONT', 'CAM_BACK', 'CAM_LEFT', 'CAM_RIGHT', 'CAM_BOTTOM']
VALID_TOOLS = [
    'get_camera_view', 'detect_objects', 'estimate_depth',
    'fuse_multi_camera', 'track_motion', 'assess_safety', 'plan_path'
]
VALID_CATEGORIES = [
    "Landing Area Detection", "Entity Detection", "Cross-View Reasoning",
    "Obstacle Avoidance", "Route Optimization", "Weather-Edge Cases"
]
VALID_DIFFICULTIES = ["easy", "medium", "hard"]


class DatasetValidator:
    """Validate MC-UAV-Bench dataset quality"""
    
    def __init__(self, samples: List[Dict]):
        self.samples = samples
        self.errors = []
        self.warnings = []
        
    def validate_all(self) -> Dict:
        """Run all validation checks"""
        self.errors = []
        self.warnings = []
        
        for i, sample in enumerate(self.samples):
            self._validate_sample(sample, i)
        
        report = {
            'total_samples': len(self.samples),
            'valid_samples': len(self.samples) - len(set(e['sample_idx'] for e in self.errors)),
            'error_count': len(self.errors),
            'warning_count': len(self.warnings),
            'errors': self.errors,
            'warnings': self.warnings,
            'is_valid': len(self.errors) == 0
        }
        
        return report
    
    def _validate_sample(self, sample: Dict, idx: int):
        """Validate single sample"""
        sample_id = sample.get('id', f'index_{idx}')
        
        # Required fields
        required = ['id', 'category', 'difficulty', 'question', 'tool_calls', 
                    'answer', 'required_cameras', 'image_paths']
        for field in required:
            if field not in sample or not sample[field]:
                self.errors.append({
                    'sample_idx': idx,
                    'sample_id': sample_id,
                    'type': 'missing_field',
                    'message': f"Missing required field: {field}"
                })
        
        # Category
        if sample.get('category') not in VALID_CATEGORIES:
            self.errors.append({
                'sample_idx': idx,
                'sample_id': sample_id,
                'type': 'invalid_category',
                'message': f"Invalid category: {sample.get('category')}"
            })
        
        # Difficulty
        if sample.get('difficulty') not in VALID_DIFFICULTIES:
            self.errors.append({
                'sample_idx': idx,
                'sample_id': sample_id,
                'type': 'invalid_difficulty',
                'message': f"Invalid difficulty: {sample.get('difficulty')}"
            })
        
        # Cameras
        cameras = sample.get('required_cameras', [])
        if len(cameras) < 2:
            self.warnings.append({
                'sample_idx': idx,
                'sample_id': sample_id,
                'type': 'few_cameras',
                'message': f"Only {len(cameras)} cameras"
            })
        
        for cam in cameras:
            if cam not in VALID_CAMERAS:
                self.errors.append({
                    'sample_idx': idx,
                    'sample_id': sample_id,
                    'type': 'invalid_camera',
                    'message': f"Invalid camera: {cam}"
                })
        
        # Tools
        tools = sample.get('tool_calls', [])
        if len(tools) < 4:
            self.warnings.append({
                'sample_idx': idx,
                'sample_id': sample_id,
                'type': 'few_tools',
                'message': f"Only {len(tools)} tools"
            })
        
        for tool in tools:
            tool_name = tool.get('tool', '')
            if tool_name not in VALID_TOOLS:
                self.errors.append({
                    'sample_idx': idx,
                    'sample_id': sample_id,
                    'type': 'invalid_tool',
                    'message': f"Invalid tool: {tool_name}"
                })
        
        # Tool ordering
        if not self._validate_tool_order(tools):
            self.warnings.append({
                'sample_idx': idx,
                'sample_id': sample_id,
                'type': 'tool_order',
                'message': "Suboptimal tool order"
            })
        
        # Measurements
        answer = sample.get('answer', '')
        if not self._has_measurements(answer):
            self.warnings.append({
                'sample_idx': idx,
                'sample_id': sample_id,
                'type': 'no_measurements',
                'message': "No measurements in answer"
            })
    
    def _validate_tool_order(self, tools: List[Dict]) -> bool:
        """Check tool order validity"""
        perception = {'get_camera_view', 'detect_objects', 'estimate_depth'}
        decision = {'assess_safety', 'plan_path'}
        
        last_perception_idx = -1
        first_decision_idx = len(tools)
        
        for i, tool in enumerate(tools):
            tool_name = tool.get('tool', '')
            if tool_name in perception:
                last_perception_idx = max(last_perception_idx, i)
            if tool_name in decision:
                first_decision_idx = min(first_decision_idx, i)
        
        return last_perception_idx < first_decision_idx or first_decision_idx == len(tools)
    
    def _has_measurements(self, text: str) -> bool:
        """Check for quantitative measurements"""
        patterns = [
            r'\d+\.?\d*\s*(m|meters?)\b',
            r'\d+\.?\d*\s*(sqm|sq\.?\s*m)',
            r'\d+\.?\d*\s*(°|degrees?)',
        ]
        return any(re.search(p, text.lower()) for p in patterns)
    
    def print_report(self, report: Dict):
        """Print validation report"""
        print("\n" + "="*60)
        print("DATASET VALIDATION REPORT")
        print("="*60)
        print(f"Total samples: {report['total_samples']}")
        print(f"Valid samples: {report['valid_samples']}")
        print(f"Errors: {report['error_count']}")
        print(f"Warnings: {report['warning_count']}")
        print(f"Overall valid: {'✓ YES' if report['is_valid'] else '✗ NO'}")
        
        if report['errors']:
            print("\n--- ERRORS ---")
            error_types = defaultdict(list)
            for e in report['errors']:
                error_types[e['type']].append(e)
            
            for err_type, errors in error_types.items():
                print(f"\n{err_type}: {len(errors)}")
                for e in errors[:3]:
                    print(f"  [{e['sample_id']}] {e['message']}")
                if len(errors) > 3:
                    print(f"  ... and {len(errors) - 3} more")


def validate_dataset(filepath: str) -> Dict:
    """Validate dataset file"""
    import json
    
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    samples = data.get('samples', data)
    validator = DatasetValidator(samples)
    report = validator.validate_all()
    validator.print_report(report)
    
    return report
