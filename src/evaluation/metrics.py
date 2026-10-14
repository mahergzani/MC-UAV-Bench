"""
MC-UAV-Bench Evaluation Metrics
"""

import re
import numpy as np
from typing import Dict, List
from collections import defaultdict


VALID_TOOLS = [
    'get_camera_view', 'detect_objects', 'estimate_depth',
    'fuse_multi_camera', 'track_motion', 'assess_safety', 'plan_path'
]
VALID_CAMERAS = ['CAM_FRONT', 'CAM_BACK', 'CAM_LEFT', 'CAM_RIGHT', 'CAM_BOTTOM']
PERCEPTION_TOOLS = {'get_camera_view', 'detect_objects', 'estimate_depth'}
FUSION_TOOLS = {'fuse_multi_camera'}
DECISION_TOOLS = {'assess_safety', 'plan_path'}
# Lower-cased marker of UAVAgent's "Cannot Determine Safety" answer
# (uav_agent.INDETERMINATE_MARKER).
ABSTAIN_MARKER = 'indeterminate'


class AnswerMetrics:
    """Answer-level evaluation"""

    @staticmethod
    def compute_accuracy(predictions: List[Dict], ground_truths: List[Dict]) -> float:
        correct = sum(1 for p, g in zip(predictions, ground_truths)
                      if AnswerMetrics.is_correct(p, g))
        return correct / len(predictions) if predictions else 0.0

    @staticmethod
    def is_correct(prediction: Dict, ground_truth: Dict) -> bool:
        pred = prediction.get('answer', '').lower()
        gt   = ground_truth.get('answer', '').lower()

        # FIX (re-scoring audit, 2026-09-23): the previous is_safe() defaulted
        # to False for any text containing neither 'safe' nor 'unsafe', which
        # silently scored degenerate non-answers (ERROR sentinels, "Max
        # iterations reached", or any real answer that ran out of budget
        # before stating a verdict) as correct whenever gt was 'unsafe' --
        # False == False by coincidence, not because the answer said
        # anything. Confirmed to inflate accuracy by up to +18.6 points on
        # Protocol C_tool runs and +1.3-2.4 points even on ordinary Protocol
        # A/B free-text answers. Returning None (no assertion) for a
        # degenerate answer and treating that as always-wrong closes the
        # loophole without changing scoring for any real, verdict-stating
        # answer.
        # Reviewer comment 5: UAVAgent's "Cannot Determine Safety" state is
        # an abstention -- never credited as correct, for any GT type.
        if ABSTAIN_MARKER in pred:
            return False

        def is_safe(text):
            if 'unsafe' in text or 'not safe' in text:
                return False
            if 'safe' in text:
                return True
            return None

        if 'safe' in gt or 'unsafe' in gt:
            pred_safe = is_safe(pred)
            if pred_safe is None:
                return False
            return pred_safe == is_safe(gt)

        return AnswerMetrics.keyword_match(pred, gt) > 0.5

    @staticmethod
    def keyword_match(pred: str, gt: str) -> float:
        stopwords = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
                     'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
                     'could', 'should', 'to', 'of', 'in', 'for', 'on', 'with', 'at'}

        pred_words = set(re.findall(r'\b\w+\b', pred.lower())) - stopwords
        gt_words   = set(re.findall(r'\b\w+\b', gt.lower())) - stopwords

        if not gt_words:
            return 0.0

        return len(pred_words & gt_words) / len(gt_words)

    @staticmethod
    def compute_by_category(predictions: List[Dict], ground_truths: List[Dict]) -> Dict[str, float]:
        by_cat = defaultdict(lambda: {'correct': 0, 'total': 0})

        for pred, gt in zip(predictions, ground_truths):
            cat = gt.get('category', 'unknown')
            by_cat[cat]['total'] += 1
            if AnswerMetrics.is_correct(pred, gt):
                by_cat[cat]['correct'] += 1

        return {cat: d['correct'] / d['total'] if d['total'] > 0 else 0.0
                for cat, d in by_cat.items()}

    @staticmethod
    def compute_by_difficulty(predictions: List[Dict], ground_truths: List[Dict]) -> Dict[str, float]:
        by_diff = defaultdict(lambda: {'correct': 0, 'total': 0})

        for pred, gt in zip(predictions, ground_truths):
            diff = gt.get('difficulty', 'unknown')
            by_diff[diff]['total'] += 1
            if AnswerMetrics.is_correct(pred, gt):
                by_diff[diff]['correct'] += 1

        return {diff: d['correct'] / d['total'] if d['total'] > 0 else 0.0
                for diff, d in by_diff.items()}


class ProcessMetrics:
    """Process-level evaluation (Protocol C)"""

    @staticmethod
    def compute_camera_coverage(cameras_used: List[str], cameras_required: List[str]) -> float:
        """Coverage = |Used ∩ Required| / |Required|"""
        used     = set(cameras_used)
        required = set(cameras_required)

        if not required:
            return 1.0

        return len(used & required) / len(required)

    @staticmethod
    def compute_tool_metrics(tools_used: List[str], tools_required: List[str]) -> Dict[str, float]:
        """Compute precision, recall, F1"""
        used     = set(tools_used)
        required = set(tools_required)

        if not used and not required:
            return {'precision': 1.0, 'recall': 1.0, 'f1': 1.0}

        tp        = len(used & required)
        precision = tp / len(used)     if used     else 0.0
        recall    = tp / len(required) if required else 0.0
        f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        return {'precision': precision, 'recall': recall, 'f1': f1}

    @staticmethod
    def validate_tool_order(tools_used: List[str]) -> bool:
        """Check if perception comes before decision"""
        last_perception = -1
        first_decision  = len(tools_used)

        for i, tool in enumerate(tools_used):
            if tool in PERCEPTION_TOOLS:
                last_perception = max(last_perception, i)
            if tool in DECISION_TOOLS:
                first_decision = min(first_decision, i)

        return last_perception < first_decision

    @staticmethod
    def compute_tool_order_score(tools_used: List[str]) -> float:
        """Tool ordering quality (0-1) via inversion count"""
        if len(tools_used) <= 1:
            return 1.0

        order_scores = {
            'get_camera_view': 1, 'detect_objects': 2, 'estimate_depth': 3,
            'fuse_multi_camera': 4, 'track_motion': 5, 'assess_safety': 6, 'plan_path': 7
        }

        inversions = 0
        total      = 0

        for i in range(len(tools_used)):
            for j in range(i + 1, len(tools_used)):
                total += 1
                if order_scores.get(tools_used[i], 4) > order_scores.get(tools_used[j], 4):
                    inversions += 1

        return 1.0 - (inversions / total) if total > 0 else 1.0


class SafetyMetrics:
    """Safety-level evaluation"""

    @staticmethod
    def extract_safety_decision(text: str) -> str:
        # FIX 1: check unsafe before safe — 'unsafe' contains 'safe' as substring
        text_lower = text.lower()
        if ABSTAIN_MARKER in text_lower:
            return 'abstain'
        if 'unsafe' in text_lower or 'not safe' in text_lower:
            return 'unsafe'
        elif 'safe' in text_lower:
            return 'safe'
        return 'unknown'

    @staticmethod
    def compute_safety_metrics(
        predictions: List[Dict],
        ground_truths: List[Dict],
        safety_critical_only: bool = True
    ) -> Dict[str, float]:
        """
        FNR = FN / (FN + TP)  -- Missed hazards (lower is better)
        FPR = FP / (FP + TN)  -- False alarms
        CAR = TP / (TP + FN)  -- Collision avoidance rate (= recall)
        """
        tp, tn, fp, fn = 0, 0, 0, 0
        abstain_unsafe, abstain_safe = 0, 0

        safety_cats = ["Landing Area Detection", "Entity Detection",
                       "Obstacle Avoidance", "Weather-Edge Cases"]

        for pred, gt in zip(predictions, ground_truths):
            if safety_critical_only and gt.get('category') not in safety_cats:
                continue

            pred_dec = SafetyMetrics.extract_safety_decision(pred.get('answer', ''))

            # FIX 2: use is_safe boolean from GT when available — same fix as safety_metrics.py
            if 'is_safe' in gt:
                gt_dec = 'safe' if gt['is_safe'] else 'unsafe'
            else:
                gt_dec = SafetyMetrics.extract_safety_decision(gt.get('answer', ''))

            if gt_dec == 'unknown' or pred_dec == 'unknown':
                continue

            # Abstentions are counted, not silently dropped: dropping them
            # would shrink the FNR denominator and flatter the system.
            if pred_dec == 'abstain':
                if gt_dec == 'unsafe':
                    abstain_unsafe += 1
                else:
                    abstain_safe += 1
                continue

            if gt_dec == 'unsafe':
                if pred_dec == 'unsafe':
                    tp += 1
                else:
                    fn += 1
            else:
                if pred_dec == 'safe':
                    tn += 1
                else:
                    fp += 1

        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        # FIX 3: CAR = TP/(TP+FN) not 1-FNR — same fix as safety_metrics.py, explicit formula
        car = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        acc = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
        abstained = abstain_unsafe + abstain_safe
        n_scored = tp + tn + fp + fn + abstained
        # Conservative variant: an abstention on a hazardous scene counts as
        # a missed hazard (the hazard was not flagged).
        fnr_abstain_as_miss = ((fn + abstain_unsafe) / (tp + fn + abstain_unsafe)
                               if (tp + fn + abstain_unsafe) > 0 else 0.0)

        return {
            'true_positives': tp, 'true_negatives': tn,
            'false_positives': fp, 'false_negatives': fn,
            'fnr': fnr, 'fpr': fpr, 'car': car, 'accuracy': acc,
            'abstained_on_unsafe': abstain_unsafe, 'abstained_on_safe': abstain_safe,
            'abstention_rate': abstained / n_scored if n_scored > 0 else 0.0,
            'fnr_abstain_as_miss': fnr_abstain_as_miss,
            'car_abstain_as_miss': 1.0 - fnr_abstain_as_miss if (tp + fn + abstain_unsafe) > 0 else 0.0,
        }


class MCUAVBenchEvaluator:
    """Complete evaluator"""

    def __init__(self):
        self.answer_metrics  = AnswerMetrics()
        self.process_metrics = ProcessMetrics()
        self.safety_metrics  = SafetyMetrics()

    def evaluate_single(self, prediction: Dict, ground_truth: Dict, include_process: bool = True) -> Dict:
        results = {
            'correct':       AnswerMetrics.is_correct(prediction, ground_truth),
            'keyword_match': AnswerMetrics.keyword_match(
                prediction.get('answer', ''), ground_truth.get('answer', ''))
        }

        if include_process and 'tool_calls' in prediction:
            pred_tools = self._extract_tool_names(prediction.get('tool_calls', []))
            gt_tools   = self._extract_tool_names(ground_truth.get('tool_calls', []))
            # FIX 4: cameras_used key takes priority; required_cameras is GT field not prediction field
            pred_cams  = prediction.get('cameras_used', [])
            gt_cams    = ground_truth.get('required_cameras', [])

            results['camera_coverage']  = ProcessMetrics.compute_camera_coverage(pred_cams, gt_cams)
            results['tool_metrics']     = ProcessMetrics.compute_tool_metrics(pred_tools, gt_tools)
            results['tool_order_valid'] = ProcessMetrics.validate_tool_order(pred_tools)
            results['tool_order_score'] = ProcessMetrics.compute_tool_order_score(pred_tools)

        results['safety'] = {
            'pred': SafetyMetrics.extract_safety_decision(prediction.get('answer', '')),
            'gt':   SafetyMetrics.extract_safety_decision(ground_truth.get('answer', '')),
        }
        results['safety']['match'] = results['safety']['pred'] == results['safety']['gt']

        return results

    def evaluate_dataset(self, predictions: List[Dict], ground_truths: List[Dict],
                         include_process: bool = True) -> Dict:
        # FIX 5: guard against empty inputs — _aggregate crashes with IndexError on results[0]
        if not predictions or not ground_truths:
            return {
                'total_samples': 0,
                'overall_accuracy': 0.0,
                'by_category': {},
                'by_difficulty': {},
                'safety_metrics': {}
            }

        all_results = []

        for pred, gt in zip(predictions, ground_truths):
            result = self.evaluate_single(pred, gt, include_process)
            result['sample_id'] = gt.get('id', 'unknown')
            result['category']  = gt.get('category', 'unknown')
            result['difficulty'] = gt.get('difficulty', 'unknown')
            all_results.append(result)

        aggregated = self._aggregate(all_results, predictions, ground_truths)
        aggregated['per_sample'] = all_results
        return aggregated

    def _extract_tool_names(self, tool_calls: List) -> List[str]:
        names = []
        for tool in tool_calls:
            if isinstance(tool, dict):
                names.append(tool.get('tool', ''))
            elif isinstance(tool, str):
                names.append(tool)
        return [n for n in names if n]

    def _aggregate(self, results: List[Dict], predictions: List[Dict],
                   ground_truths: List[Dict]) -> Dict:
        agg = {
            'total_samples':  len(results),
            'overall_accuracy': float(np.mean([r['correct'] for r in results])),
            'by_category':    AnswerMetrics.compute_by_category(predictions, ground_truths),
            'by_difficulty':  AnswerMetrics.compute_by_difficulty(predictions, ground_truths),
        }

        # FIX 6: guard against missing key before indexing results[0] —
        # Protocol A/B results never have 'camera_coverage', this check must
        # use .get() not direct key access to avoid KeyError
        if results[0].get('camera_coverage') is not None:
            agg['avg_camera_coverage'] = float(np.mean([r['camera_coverage'] for r in results]))
            agg['avg_tool_f1']         = float(np.mean([r['tool_metrics']['f1'] for r in results]))
            agg['avg_tool_precision']  = float(np.mean([r['tool_metrics']['precision'] for r in results]))
            agg['avg_tool_recall']     = float(np.mean([r['tool_metrics']['recall'] for r in results]))
            agg['tool_order_accuracy'] = float(np.mean([r['tool_order_valid'] for r in results]))

        agg['safety_metrics'] = SafetyMetrics.compute_safety_metrics(
            predictions, ground_truths, safety_critical_only=True)

        return agg