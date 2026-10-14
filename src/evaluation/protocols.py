"""
MC-UAV-Bench Three-Protocol Evaluation Framework

Protocol A: End-to-End (all cameras provided)
Protocol B: Oracle (only required cameras provided)
Protocol C: Tool-Augmented (agentic tool access)
"""

from typing import Dict, List, Optional
from dataclasses import dataclass
from PIL import Image
from pathlib import Path


@dataclass
class ProtocolConfig:
    """Protocol configuration"""
    name: str
    description: str
    provide_all_cameras: bool
    provide_oracle_cameras: bool
    enable_tools: bool


# Protocol Definitions
PROTOCOL_A = ProtocolConfig(
    name="A",
    description="End-to-End: All cameras provided, model decides which to use",
    provide_all_cameras=True,
    provide_oracle_cameras=False,
    enable_tools=False
)

PROTOCOL_B = ProtocolConfig(
    name="B",
    description="Oracle: Only required cameras provided, tests pure perception",
    provide_all_cameras=False,
    provide_oracle_cameras=True,
    enable_tools=False
)

PROTOCOL_C = ProtocolConfig(
    name="C",
    description="Tool-Augmented: Model invokes tools to access cameras",
    provide_all_cameras=True,
    provide_oracle_cameras=False,
    enable_tools=True
)


class ProtocolEvaluator:
    """Evaluator for three protocols"""

    ALL_CAMERAS = ['CAM_FRONT', 'CAM_BACK', 'CAM_LEFT', 'CAM_RIGHT', 'CAM_BOTTOM']

    def __init__(self, model, tool_executor=None):
        self.model = model
        self.tool_executor = tool_executor

    def prepare_input(self, sample: Dict, protocol: ProtocolConfig) -> Dict:
        """Prepare input based on protocol"""
        prepared = {
            'question': sample['question'],
            'sample_id': sample.get('id', 'unknown'),
            'category': sample.get('category', 'unknown'),
            'ground_truth': sample
        }

        if protocol.provide_oracle_cameras:
            cameras = sample.get('required_cameras', [])
        else:
            cameras = self.ALL_CAMERAS

        prepared['images'] = {}
        prepared['available_cameras'] = cameras

        # FIX 1: Protocol C doesn't need PIL images loaded here —
        # tool_executor uses image_paths directly. Only load for A/B.
        if not protocol.enable_tools:
            for cam in cameras:
                if 'images' in sample and cam in sample['images']:
                    prepared['images'][cam] = sample['images'][cam]
                elif 'image_paths' in sample and cam in sample['image_paths']:
                    path = sample['image_paths'][cam]
                    prepared['images'][cam] = Image.open(path).convert('RGB')

        prepared['tools_enabled'] = protocol.enable_tools
        return prepared

    def run_protocol_a(self, sample: Dict) -> Dict:
        """Protocol A: End-to-End with all cameras"""
        prepared = self.prepare_input(sample, PROTOCOL_A)

        images = list(prepared['images'].values())
        cameras = list(prepared['images'].keys())

        prediction = self.model.predict(
            question=prepared['question'],
            images=images,
            camera_names=cameras,
            enable_tools=False
        )

        return {
            'protocol': 'A',
            'sample_id': prepared['sample_id'],
            'answer': prediction.answer,           # FIX 2: expose answer at top level
            'cameras_provided': cameras,
            'prediction': prediction.to_dict(),    # FIX 2: store as dict for evaluator
            'ground_truth': prepared['ground_truth']
        }

    def run_protocol_b(self, sample: Dict) -> Dict:
        """Protocol B: Oracle camera selection"""
        prepared = self.prepare_input(sample, PROTOCOL_B)

        images = list(prepared['images'].values())
        cameras = list(prepared['images'].keys())

        prediction = self.model.predict(
            question=prepared['question'],
            images=images,
            camera_names=cameras,
            enable_tools=False
        )

        return {
            'protocol': 'B',
            'sample_id': prepared['sample_id'],
            'answer': prediction.answer,           # FIX 2
            'cameras_provided': cameras,
            'prediction': prediction.to_dict(),    # FIX 2
            'ground_truth': prepared['ground_truth']
        }

    def run_protocol_c(self, sample: Dict) -> Dict:
        """Protocol C: Tool-augmented agentic"""
        if not self.tool_executor:
            raise ValueError("Tool executor required for Protocol C")

        prepared = self.prepare_input(sample, PROTOCOL_C)

        # FIX 3: reset executor and register image paths BEFORE predict_with_tools
        # Omitting these means tool_executor has stale/empty image_paths from the
        # previous sample, causing get_camera_view to return available=False for all cameras
        self.tool_executor.reset()
        self.tool_executor.set_image_paths(sample.get('image_paths', {}))

        prediction = self.model.predict_with_tools(
            question=prepared['question'],
            available_cameras=prepared['available_cameras'],
            image_paths=sample.get('image_paths', {}),
            tool_executor=self.tool_executor,
            max_iterations=5
        )

        # FIX 4: prediction is a ModelPrediction dataclass, not a dict —
        # .get() doesn't exist on dataclasses; use attribute access instead
        return {
            'protocol': 'C',
            'sample_id': prepared['sample_id'],
            'answer': prediction.answer,           # FIX 2
            'tool_trace': prediction.tool_calls,   # FIX 4: was prediction.get('tool_trace')
            'cameras_used': prediction.cameras_used,  # FIX 4: was prediction.get('cameras_used')
            'prediction': prediction.to_dict(),    # FIX 2
            'ground_truth': prepared['ground_truth']
        }

    def evaluate_sample(self, sample: Dict, protocols: List[str] = None) -> Dict:
        """Run specified protocols on a sample"""
        if protocols is None:
            protocols = ['A', 'B', 'C']

        results = {}

        if 'A' in protocols:
            results['A'] = self.run_protocol_a(sample)

        if 'B' in protocols:
            results['B'] = self.run_protocol_b(sample)

        if 'C' in protocols:
            if self.tool_executor:
                results['C'] = self.run_protocol_c(sample)
            else:
                # FIX 5: warn instead of silently skipping — silent skip means
                # Protocol C results are missing from output with no indication why
                print("  WARNING: Protocol C skipped — no tool_executor provided")

        return results

    def compute_delta_ab(self, results_a: List[Dict], results_b: List[Dict], evaluator) -> Dict:
        """
        Compute camera selection gap: ΔA→B = Acc_B - Acc_A

        Large gap = camera selection is bottleneck
        Small gap = perception/reasoning is bottleneck
        """
        # FIX 2: results now store prediction as dict, use that directly
        preds_a = [r['prediction'] for r in results_a]
        preds_b = [r['prediction'] for r in results_b]
        gts = [r['ground_truth'] for r in results_a]

        acc_a = evaluator.evaluate_dataset(preds_a, gts)['overall_accuracy']
        acc_b = evaluator.evaluate_dataset(preds_b, gts)['overall_accuracy']

        delta = acc_b - acc_a

        return {
            'accuracy_a': acc_a,
            'accuracy_b': acc_b,
            'delta_ab': delta,
            'interpretation': self._interpret_delta(delta)
        }

    def _interpret_delta(self, delta: float) -> str:
        if delta > 0.10:
            return "Large gap (>10%): Camera selection is PRIMARY bottleneck"
        elif delta > 0.05:
            return "Medium gap (5-10%): Camera selection is A bottleneck"
        else:
            return "Small gap (<5%): Camera selection is NOT the main issue"


def run_full_evaluation(
    model,
    test_samples: List[Dict],
    evaluator,
    tool_executor=None,
    protocols: List[str] = None
) -> Dict:
    """Run complete evaluation across protocols"""
    if protocols is None:
        protocols = ['A', 'B', 'C']

    protocol_eval = ProtocolEvaluator(model, tool_executor)
    all_results = {p: [] for p in protocols}

    for sample in test_samples:
        sample_results = protocol_eval.evaluate_sample(sample, protocols)
        for p, result in sample_results.items():
            all_results[p].append(result)

    final = {}
    for protocol in protocols:
        if not all_results[protocol]:
            continue
        # FIX 2: prediction is already stored as dict — pass directly to evaluator
        preds = [r['prediction'] for r in all_results[protocol]]
        gts   = [r['ground_truth'] for r in all_results[protocol]]
        final[protocol] = evaluator.evaluate_dataset(preds, gts)
        final[protocol]['raw_results'] = all_results[protocol]

    if 'A' in protocols and 'B' in protocols:
        if all_results['A'] and all_results['B']:
            final['delta_analysis'] = protocol_eval.compute_delta_ab(
                all_results['A'], all_results['B'], evaluator
            )

    return final