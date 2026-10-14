"""
MC-UAV-Bench Dataset Loading Module
"""

import json
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass
from PIL import Image

from src.data.image_root import get_image_root, resolve_image_paths


@dataclass
class UAVSample:
    """Single sample from MC-UAV-Bench"""
    id: str
    category: str
    sub_category: str
    difficulty: str
    question_type: str
    question: str
    tool_calls: List[Dict]
    answer: str
    required_cameras: List[str]
    reasoning: str
    safety_critical: bool
    spatial_measurements: Dict
    scene_token: str
    image_paths: Dict[str, str]
    quality_metrics: Dict
    
    @property
    def num_tools(self) -> int:
        return len(self.tool_calls)
    
    @property
    def num_cameras(self) -> int:
        return len(self.required_cameras)
    
    @property
    def is_safety_critical(self) -> bool:
        return self.category in [
            "Landing Area Detection",
            "Entity Detection", 
            "Obstacle Avoidance",
            "Weather-Edge Cases"
        ]


class MCUAVBenchDataset:
    """Dataset class for MC-UAV-Bench"""
    
    VALID_CAMERAS = ['CAM_FRONT', 'CAM_BACK', 'CAM_LEFT', 'CAM_RIGHT', 'CAM_BOTTOM']
    VALID_TOOLS = [
        'get_camera_view', 'detect_objects', 'estimate_depth',
        'fuse_multi_camera', 'track_motion', 'assess_safety', 'plan_path'
    ]
    
    def __init__(
        self,
        data_path: str,
        split: Optional[str] = None,
        transform=None,
        load_images: bool = True,
        image_size: int = 512,
        image_root: Optional[str] = None
    ):
        self.data_path = Path(data_path)
        # Root of the user's UAV3D v1.0 copy (--image-root or UAV3D_ROOT);
        # only needed when load_images is True.
        self.image_root = get_image_root(image_root) if load_images else None
        self.split = split
        self.transform = transform
        self.load_images = load_images
        self.image_size = image_size
        
        self.samples = self._load_data()
        self.metadata = self._load_metadata()
        
    def _load_data(self) -> List[UAVSample]:
        """Load and parse dataset"""
        with open(self.data_path, 'r') as f:
            data = json.load(f)
        
        samples = []
        raw_samples = data.get('samples', data)
        
        for s in raw_samples:
            sample = UAVSample(
                id=s.get('id', ''),
                category=s.get('category', ''),
                sub_category=s.get('sub_category', ''),
                difficulty=s.get('difficulty', ''),
                question_type=s.get('question_type', ''),
                question=s.get('question', ''),
                tool_calls=s.get('tool_calls', []),
                answer=s.get('answer', ''),
                required_cameras=s.get('required_cameras', []),
                reasoning=s.get('reasoning', ''),
                safety_critical=s.get('safety_critical', False),
                spatial_measurements=s.get('spatial_measurements', {}),
                scene_token=s.get('scene_token', ''),
                image_paths=s.get('image_paths', {}),
                quality_metrics=s.get('quality_metrics', {})
            )
            samples.append(sample)
        
        return samples
    
    def _load_metadata(self) -> Dict:
        """Load dataset metadata"""
        with open(self.data_path, 'r') as f:
            data = json.load(f)
        return data.get('metadata', {})
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict:
        sample = self.samples[idx]
        
        item = {
            'id': sample.id,
            'category': sample.category,
            'sub_category': sample.sub_category,
            'difficulty': sample.difficulty,
            'question_type': sample.question_type,
            'question': sample.question,
            'tool_calls': sample.tool_calls,
            'answer': sample.answer,
            'required_cameras': sample.required_cameras,
            'reasoning': sample.reasoning,
            'safety_critical': sample.is_safety_critical,
            'spatial_measurements': sample.spatial_measurements,
        }
        
        if self.load_images:
            images = {}
            resolved = resolve_image_paths(sample.image_paths, self.image_root)
            for cam, path in resolved.items():
                img = Image.open(path).convert('RGB')
                if self.image_size:
                    img = img.resize((self.image_size, self.image_size))
                if self.transform:
                    img = self.transform(img)
                images[cam] = img
            item['images'] = images
            item['image_paths'] = resolved
        
        return item
    
    def get_by_id(self, sample_id: str) -> Optional[Dict]:
        """Get sample by ID"""
        for idx, sample in enumerate(self.samples):
            if sample.id == sample_id:
                return self[idx]
        return None
    
    def filter_by_category(self, category: str) -> 'MCUAVBenchDataset':
        """Return filtered dataset by category"""
        filtered = MCUAVBenchDataset.__new__(MCUAVBenchDataset)
        filtered.samples = [s for s in self.samples if s.category == category]
        filtered.metadata = self.metadata
        filtered.load_images = self.load_images
        filtered.transform = self.transform
        filtered.image_size = self.image_size
        filtered.image_root = self.image_root
        filtered.data_path = self.data_path
        return filtered
    
    def filter_by_difficulty(self, difficulty: str) -> 'MCUAVBenchDataset':
        """Return filtered dataset by difficulty"""
        filtered = MCUAVBenchDataset.__new__(MCUAVBenchDataset)
        filtered.samples = [s for s in self.samples if s.difficulty == difficulty]
        filtered.metadata = self.metadata
        filtered.load_images = self.load_images
        filtered.transform = self.transform
        filtered.image_size = self.image_size
        filtered.image_root = self.image_root
        filtered.data_path = self.data_path
        return filtered
    
    def get_safety_critical_subset(self) -> 'MCUAVBenchDataset':
        """Return safety-critical subset"""
        filtered = MCUAVBenchDataset.__new__(MCUAVBenchDataset)
        filtered.samples = [s for s in self.samples if s.is_safety_critical]
        filtered.metadata = self.metadata
        filtered.load_images = self.load_images
        filtered.transform = self.transform
        filtered.image_size = self.image_size
        filtered.image_root = self.image_root
        filtered.data_path = self.data_path
        return filtered
    
    def get_statistics(self) -> Dict:
        """Compute dataset statistics"""
        from collections import defaultdict
        
        stats = {
            'total_samples': len(self.samples),
            'by_category': defaultdict(int),
            'by_difficulty': defaultdict(int),
            'by_question_type': defaultdict(int),
            'avg_tools': 0,
            'avg_cameras': 0,
            'safety_critical_count': 0
        }
        
        total_tools = 0
        total_cameras = 0
        
        for s in self.samples:
            stats['by_category'][s.category] += 1
            stats['by_difficulty'][s.difficulty] += 1
            stats['by_question_type'][s.question_type] += 1
            total_tools += s.num_tools
            total_cameras += s.num_cameras
            if s.is_safety_critical:
                stats['safety_critical_count'] += 1
        
        stats['avg_tools'] = total_tools / len(self.samples) if self.samples else 0
        stats['avg_cameras'] = total_cameras / len(self.samples) if self.samples else 0
        stats['by_category'] = dict(stats['by_category'])
        stats['by_difficulty'] = dict(stats['by_difficulty'])
        stats['by_question_type'] = dict(stats['by_question_type'])
        
        return stats


def load_dataset(path: str, **kwargs) -> MCUAVBenchDataset:
    """Convenience function to load dataset"""
    return MCUAVBenchDataset(path, **kwargs)
