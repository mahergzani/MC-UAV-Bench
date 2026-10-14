"""
MC-UAV-Bench Safety Metrics Module

Specialized metrics for safety-critical UAV scenarios.
"""

import re
from typing import Dict, List, Tuple
from collections import defaultdict


SAFETY_CRITICAL_CATEGORIES = [
    "Landing Area Detection",
    "Entity Detection",
    "Obstacle Avoidance",
    "Weather-Edge Cases"
]


class SafetyEvaluator:
    """Comprehensive safety evaluation for UAV decisions"""

    def __init__(self):
        # FIX 1: removed unused self.confusion_matrix — compute_batch_metrics
        # used local vars, this instance var was never read or updated anywhere
        pass

    def extract_decision(self, text: str) -> str:
        """
        Extract safety decision from text.

        FIX 2: 'safe' is a substring of 'unsafe' — original code checked unsafe
        first which was accidentally correct, but fragile. Rewritten with explicit
        word-boundary logic: check unsafe/abort phrases before safe phrases so
        'unsafe' never falls through to the safe branch.
        """
        if not text:
            return 'unknown'

        text = text.lower()

        # UAVAgent's "Cannot Determine Safety" abstention (reviewer comment 5)
        if 'indeterminate' in text:
            return 'uncertain'

        # Must check unsafe BEFORE safe — 'unsafe' contains 'safe' as substring
        # Checking safe first would misclassify 'unsafe' as 'safe'
        if any(phrase in text for phrase in ['unsafe', 'not safe', 'dangerous', 'hazardous', 'abort']):
            return 'unsafe'

        if any(phrase in text for phrase in ['safe', 'proceed', 'clear', 'acceptable']):
            return 'safe'

        if any(phrase in text for phrase in ['uncertain', 'unclear', 'cannot determine', 'unknown']):
            return 'uncertain'

        return 'unknown'

    def evaluate_safety_decision(
        self,
        prediction: Dict,
        ground_truth: Dict
    ) -> Dict:
        """Evaluate single safety decision"""
        pred_dec = self.extract_decision(prediction.get('answer', ''))

        # FIX 3: ground truth may have explicit 'is_safe' boolean field
        # (from tool_executor._assess_safety and dataset generation) — use it
        # directly when available rather than relying on text extraction which
        # can misparse structured answers like "SAFE - PROCEED. Clearance: 5m"
        if 'is_safe' in ground_truth:
            gt_dec = 'safe' if ground_truth['is_safe'] else 'unsafe'
        else:
            gt_dec = self.extract_decision(ground_truth.get('answer', ''))

        result = {
            'pred_decision': pred_dec,
            'gt_decision': gt_dec,
            'correct': pred_dec == gt_dec,
            'category': ground_truth.get('category', 'unknown')
        }

        if gt_dec == 'unsafe':
            result['classification'] = 'TP' if pred_dec == 'unsafe' else 'FN'
        elif gt_dec == 'safe':
            result['classification'] = 'TN' if pred_dec == 'safe' else 'FP'
        else:
            result['classification'] = 'N/A'

        return result

    def compute_batch_metrics(
        self,
        predictions: List[Dict],
        ground_truths: List[Dict],
        safety_critical_only: bool = True
    ) -> Dict:
        """Compute comprehensive safety metrics"""
        tp, tn, fp, fn = 0, 0, 0, 0
        by_category = defaultdict(lambda: {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0})

        for pred, gt in zip(predictions, ground_truths):
            cat = gt.get('category', 'unknown')

            if safety_critical_only and cat not in SAFETY_CRITICAL_CATEGORIES:
                continue

            result = self.evaluate_safety_decision(pred, gt)

            if result['classification'] == 'TP':
                tp += 1
                by_category[cat]['tp'] += 1
            elif result['classification'] == 'TN':
                tn += 1
                by_category[cat]['tn'] += 1
            elif result['classification'] == 'FP':
                fp += 1
                by_category[cat]['fp'] += 1
            elif result['classification'] == 'FN':
                fn += 1
                by_category[cat]['fn'] += 1
            # N/A (uncertain/unknown GT) intentionally skipped — not counted in total

        total = tp + tn + fp + fn
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        metrics = {
            'counts': {'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn, 'total': total},
            'fnr':       fn / (fn + tp) if (fn + tp) > 0 else 0.0,
            'fpr':       fp / (fp + tn) if (fp + tn) > 0 else 0.0,
            'car':       tp / (tp + fn) if (tp + fn) > 0 else 0.0,  # FIX 4: CAR = TP/(TP+FN) = recall, not 1-FNR which is identical but clearer as recall
            'accuracy':  (tp + tn) / total if total > 0 else 0.0,
            'precision': precision,
            'recall':    recall,
            'f1':        2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0,
        }

        # FIX 5: per-category now includes precision/recall/f1 — original only
        # had fnr/fpr/car/accuracy which made category-level analysis incomplete
        metrics['by_category'] = {}
        for cat, counts in by_category.items():
            cat_total = sum(counts.values())
            cat_tp = counts['tp']
            cat_tn = counts['tn']
            cat_fp = counts['fp']
            cat_fn = counts['fn']
            cat_precision = cat_tp / (cat_tp + cat_fp) if (cat_tp + cat_fp) > 0 else 0.0
            cat_recall    = cat_tp / (cat_tp + cat_fn) if (cat_tp + cat_fn) > 0 else 0.0

            metrics['by_category'][cat] = {
                'fnr':       cat_fn / (cat_fn + cat_tp) if (cat_fn + cat_tp) > 0 else 0.0,
                'fpr':       cat_fp / (cat_fp + cat_tn) if (cat_fp + cat_tn) > 0 else 0.0,
                'car':       cat_tp / (cat_tp + cat_fn) if (cat_tp + cat_fn) > 0 else 0.0,
                'accuracy':  (cat_tp + cat_tn) / cat_total if cat_total > 0 else 0.0,
                'precision': cat_precision,
                'recall':    cat_recall,
                'f1':        2 * cat_precision * cat_recall / (cat_precision + cat_recall) if (cat_precision + cat_recall) > 0 else 0.0,
                'counts':    {'tp': cat_tp, 'tn': cat_tn, 'fp': cat_fp, 'fn': cat_fn},
                'total':     cat_total
            }

        return metrics

    def print_safety_report(self, metrics: Dict):
        """Print formatted safety report"""
        print("\n" + "="*60)
        print("SAFETY METRICS REPORT")
        print("="*60)

        counts = metrics['counts']
        print(f"\nConfusion Matrix:")
        print(f"  True Positives (hazard detected):  {counts['tp']}")
        print(f"  True Negatives (safe confirmed):   {counts['tn']}")
        print(f"  False Positives (false alarm):     {counts['fp']}")
        print(f"  False Negatives (MISSED HAZARD):   {counts['fn']} ⚠️")

        print(f"\nKey Metrics:")
        print(f"  False Negative Rate (FNR): {metrics['fnr']:.2%} {'⚠️ HIGH' if metrics['fnr'] > 0.15 else '✓'}")
        print(f"  False Positive Rate (FPR): {metrics['fpr']:.2%}")
        print(f"  Collision Avoidance Rate:  {metrics['car']:.2%}")
        print(f"  Safety Accuracy:           {metrics['accuracy']:.2%}")
        print(f"  Precision:                 {metrics['precision']:.2%}")
        print(f"  Recall:                    {metrics['recall']:.2%}")
        print(f"  F1 Score:                  {metrics['f1']:.2%}")

        if metrics.get('by_category'):
            print(f"\nPer-Category Safety (FNR / CAR / F1):")
            for cat, cat_metrics in metrics['by_category'].items():
                print(f"  {cat} (n={cat_metrics['total']}):")
                print(f"    FNR: {cat_metrics['fnr']:.2%}  CAR: {cat_metrics['car']:.2%}  F1: {cat_metrics['f1']:.2%}")


def compute_safety_metrics(
    predictions: List[Dict],
    ground_truths: List[Dict],
    safety_critical_only: bool = True,
    print_report: bool = False
) -> Dict:
    """Convenience function to compute safety metrics"""
    evaluator = SafetyEvaluator()
    metrics = evaluator.compute_batch_metrics(predictions, ground_truths, safety_critical_only)

    if print_report:
        evaluator.print_safety_report(metrics)

    return metrics