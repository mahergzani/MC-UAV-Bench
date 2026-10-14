
import sys
import json
import argparse
import time
import inspect
from pathlib import Path
from datetime import datetime
from collections import defaultdict

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import MCUAVBenchDataset
from src.evaluation.metrics import MCUAVBenchEvaluator, AnswerMetrics, ProcessMetrics, SafetyMetrics
from src.evaluation.protocols import PROTOCOL_A, PROTOCOL_B, PROTOCOL_C
from src.evaluation.safety_metrics import compute_safety_metrics
from src.tools.tool_executor import UAVToolExecutor
from src.data.image_root import resolve_samples
from PIL import Image


def load_test_data(filepath: str, image_root: str = None) -> list:
    """Load test dataset and resolve image paths against the UAV3D root.
    Raises FileNotFoundError if any camera image is missing."""
    with open(filepath, 'r') as f:
        data = json.load(f)
    return resolve_samples(data.get('samples', data), image_root)


def get_model(model_name: str, device: str = "cuda"):
    """Get model wrapper based on name"""
    model_lower = model_name.lower()
    
    if model_lower == 'dummy':
        from src.models.base_model import DummyVLM
        model = DummyVLM()
        model.load_model()
        return model
    
    elif 'gpt' in model_lower:
        from src.models.api_models import GPT4VisionWrapper
        model = GPT4VisionWrapper(model_name)
        model.load_model()
        return model
    
    elif 'claude' in model_lower:
        from src.models.api_models import ClaudeWrapper
        model = ClaudeWrapper(model_name)
        model.load_model()
        return model
    
    elif 'uavagent' in model_lower:
        # FIX (Claude, 2026-09-10): this branch must come BEFORE the qwen/llava/intern
        # check below -- 'uavagent:qwen2-vl-2b' contains 'qwen' as a substring, so the
        # original branch order silently routed it to vlm_wrapper.get_model() with the
        # literal string 'uavagent:qwen2-vl-2b', which isn't a MODEL_REGISTRY key ->
        # ValueError: Unknown model. Reordering fixes this without changing either
        # branch's own logic.
        from src.models.uav_agent import create_uav_agent
        base_model = model_name.split(':')[1] if ':' in model_name else "qwen2-vl-2b"
        model = create_uav_agent(model_name=base_model, device=device)
        model.load_model()
        return model

    elif 'qwen' in model_lower or 'llava' in model_lower or 'intern' in model_lower or 'phi' in model_lower:
        from src.models.vlm_wrapper import get_model as _get_vlm
        model = _get_vlm(model_name)
        model.load_model()
        return model

    else:
        raise ValueError(f"Unknown model: {model_name}")


def run_protocol_a(model, samples: list, evaluator) -> dict:
    """Run Protocol A: End-to-End with all cameras"""
    print("\n" + "="*60)
    print("PROTOCOL A: End-to-End (All Cameras)")
    print("="*60)
    
    predictions = []
    
    for i, sample in enumerate(samples):
        # Load all images
        images = []
        camera_names = []
        
        for cam, path in sample.get('image_paths', {}).items():
            images.append(Image.open(path).convert('RGB'))
            camera_names.append(cam)
        
        # Get prediction
        prediction = model.predict(
            question=sample['question'],
            images=images,
            camera_names=camera_names,
            enable_tools=False
        )
        
        predictions.append(prediction.to_dict())
        
        if (i + 1) % 10 == 0:
            print(f"  Processed {i+1}/{len(samples)} samples...")
    
    # Evaluate
    results = evaluator.evaluate_dataset(predictions, samples, include_process=False)
    
    print(f"\n  Overall Accuracy: {results['overall_accuracy']:.2%}")
    print(f"  By Category:")
    for cat, acc in results['by_category'].items():
        print(f"    {cat}: {acc:.2%}")
    
    return {'predictions': predictions, 'results': results}


def run_protocol_b(model, samples: list, evaluator) -> dict:
    """Run Protocol B: Oracle (only required cameras)"""
    print("\n" + "="*60)
    print("PROTOCOL B: Oracle (Required Cameras Only)")
    print("="*60)
    
    predictions = []
    
    for i, sample in enumerate(samples):
        # Load ONLY required cameras
        images = []
        camera_names = []
        
        required = sample.get('required_cameras', [])
        
        for cam in required:
            path = sample.get('image_paths', {}).get(cam)
            if path is None:
                raise FileNotFoundError(
                    f"{sample.get('id')}: required camera {cam} has no image path")
            images.append(Image.open(path).convert('RGB'))
            camera_names.append(cam)
        
        prediction = model.predict(
            question=sample['question'],
            images=images,
            camera_names=camera_names,
            enable_tools=False
        )
        
        predictions.append(prediction.to_dict())
        
        if (i + 1) % 10 == 0:
            print(f"  Processed {i+1}/{len(samples)} samples...")
    
    results = evaluator.evaluate_dataset(predictions, samples, include_process=False)
    
    print(f"\n  Overall Accuracy: {results['overall_accuracy']:.2%}")
    
    return {'predictions': predictions, 'results': results}


def run_protocol_c(model, samples: list, evaluator) -> dict:
    """Run Protocol C: Tool-Augmented (agentic)"""
    print("\n" + "="*60)
    print("PROTOCOL C: Tool-Augmented (Agentic)")
    print("="*60)
    
    # Check if model supports tool calling
    has_tool_support = hasattr(model, 'predict_with_tools')
    if not has_tool_support:
        print("  Warning: Model does not support tool calling, using standard prediction")
    
    predictions = []
    tool_executor = UAVToolExecutor()
    
    for i, sample in enumerate(samples):
        tool_executor.reset()
        # FIX (Claude, 2026-09-10): uav_agent.py's own comment says the caller
        # already calls set_image_paths() before predict_with_tools() -- this
        # function never did, so tool_executor.image_paths stayed {} for every
        # sample and get_camera_view always reported every camera unavailable,
        # even when the real image existed on disk. protocols.py's class-based
        # ProtocolEvaluator.run_protocol_c() already does this correctly; this
        # standalone duplicate in step4_run_evaluation.py was missing it.
        tool_executor.set_image_paths(sample.get('image_paths', {}))
        # FIX (Claude, 2026-09-10): thread the real sample id through so
        # tool_executor's seeding is genuinely per-sample and reproducible
        # across runs (see tool_executor.py's _stable_rng), and attach the
        # real VLM so detect_objects/estimate_depth/track_motion/
        # fuse_multi_camera/assess_safety can query the real image content
        # instead of returning random values.
        tool_executor.set_context(
            sample_id=sample.get('id', ''),
            model=getattr(model, 'base_model', None) or model
        )

        if has_tool_support:
            # FIX (Claude, 2026-09-12): `category` was being passed
            # unconditionally to every model's predict_with_tools(), but
            # only UAVAgent's override actually accepts it -- BaseVLM's
            # generic fallback (used by every plain baseline model:
            # Qwen2VLWrapper, InternVLWrapper, LLaVAWrapper, etc.) and both
            # API wrappers' predict_with_tools() do not, so Protocol C
            # crashed with TypeError on literally the first sample for any
            # non-UAVAgent model -- confirmed via a real full-451 run on
            # qwen2-vl-7b (audit/qwen2vl7b_full_sweep_report.md). This means
            # no plain baseline model's Protocol C had ever actually
            # completed in this codebase before now. Only pass `category`
            # when the target method's signature actually declares it, so
            # UAVAgent keeps getting it exactly as before and everything
            # else falls back correctly instead of crashing.
            kwargs = dict(
                question=sample['question'],
                available_cameras=['CAM_FRONT', 'CAM_BACK', 'CAM_LEFT', 'CAM_RIGHT', 'CAM_BOTTOM'],
                image_paths=sample.get('image_paths', {}),
                tool_executor=tool_executor,
                max_iterations=10,  # FIX (Claude): match paper Tmax=10, was hardcoded 5
            )
            if 'category' in inspect.signature(model.predict_with_tools).parameters:
                kwargs['category'] = sample.get('category')
            prediction = model.predict_with_tools(**kwargs)
        else:
            # Fallback to standard prediction
            images = []
            camera_names = []
            for cam, path in sample.get('image_paths', {}).items():
                images.append(Image.open(path).convert('RGB'))
                camera_names.append(cam)
            
            prediction = model.predict(
                question=sample['question'],
                images=images,
                camera_names=camera_names,
                enable_tools=True
            )
        
        predictions.append(prediction.to_dict())
        
        if (i + 1) % 10 == 0:
            print(f"  Processed {i+1}/{len(samples)} samples...")
    
    # Evaluate with process metrics
    results = evaluator.evaluate_dataset(predictions, samples, include_process=True)
    
    print(f"\n  Overall Accuracy: {results['overall_accuracy']:.2%}")
    if 'avg_camera_coverage' in results:
        print(f"  Camera Coverage: {results['avg_camera_coverage']:.2%}")
        print(f"  Tool F1: {results['avg_tool_f1']:.2%}")
    
    return {'predictions': predictions, 'results': results}


def run_protocol_c_tool(model, samples: list, evaluator) -> dict:
    """Run Protocol C_tool: genuine free-tool-choice agentic inference.

    Unlike the legacy run_protocol_c() above (kept as-is under the 'C' key,
    unchanged, so nothing already-verified gets silently overwritten), every
    model here goes through its own REAL predict_with_tools(): for
    Qwen2-VL-2B/7B and LLaVA-OV-7B this is the new shared free-tool-choice
    ReAct loop (src/models/react_tool_loop.py); for Claude/GPT-4o it's their
    existing native-function-calling implementation in api_models.py. No
    has_tool_support fallback branch -- BaseVLM.predict_with_tools()'s
    default (Protocol-A-equivalent) fallback IS the honest result for any
    model that doesn't have real tool support wired up (InternVL, Phi-4,
    LLaVA-Next, etc. -- out of scope for this task).
    """
    print("\n" + "="*60)
    print("PROTOCOL C_TOOL: Tool-Augmented (genuine free tool choice)")
    print("="*60)

    predictions = []
    tool_executor = UAVToolExecutor()

    for i, sample in enumerate(samples):
        tool_executor.reset()
        tool_executor.set_image_paths(sample.get('image_paths', {}))
        tool_executor.set_context(
            sample_id=sample.get('id', ''),
            model=getattr(model, 'base_model', None) or model
        )

        kwargs = dict(
            question=sample['question'],
            available_cameras=['CAM_FRONT', 'CAM_BACK', 'CAM_LEFT', 'CAM_RIGHT', 'CAM_BOTTOM'],
            image_paths=sample.get('image_paths', {}),
            tool_executor=tool_executor,
            max_iterations=10,  # matches UAVAgent's paper Tmax=10
        )
        if 'category' in inspect.signature(model.predict_with_tools).parameters:
            kwargs['category'] = sample.get('category')
        prediction = model.predict_with_tools(**kwargs)

        predictions.append(prediction.to_dict())

        if (i + 1) % 10 == 0:
            print(f"  Processed {i+1}/{len(samples)} samples...")

    results = evaluator.evaluate_dataset(predictions, samples, include_process=True)

    print(f"\n  Overall Accuracy: {results['overall_accuracy']:.2%}")
    if 'avg_camera_coverage' in results:
        print(f"  Camera Coverage: {results['avg_camera_coverage']:.2%}")
        print(f"  Tool F1: {results['avg_tool_f1']:.2%}")

    n = len(predictions)
    n_answered = sum(1 for p in predictions if p.get('termination_reason') == 'answered')
    n_budget = sum(1 for p in predictions if p.get('termination_reason') == 'budget_exhausted')
    n_error = sum(1 for p in predictions if p.get('termination_reason') == 'error')
    avg_tool_calls = sum(len(p.get('tool_calls', [])) for p in predictions) / n if n else 0.0
    avg_parse_failures = sum(p.get('parse_failures', 0) for p in predictions) / n if n else 0.0
    print(f"\n  Termination: answered={n_answered} budget_exhausted={n_budget} error={n_error} (n={n})")
    print(f"  Avg tool calls/episode: {avg_tool_calls:.2f}  Avg parse failures/episode: {avg_parse_failures:.2f}")

    return {'predictions': predictions, 'results': results}


def compute_delta_ab(results_a: dict, results_b: dict) -> dict:
    """Compute camera selection gap"""
    acc_a = results_a['results']['overall_accuracy']
    acc_b = results_b['results']['overall_accuracy']
    delta = acc_b - acc_a
    
    interpretation = ""
    if delta > 0.10:
        interpretation = "Large gap (>10%): Camera selection is PRIMARY bottleneck"
    elif delta > 0.05:
        interpretation = "Medium gap (5-10%): Camera selection is A bottleneck"
    else:
        interpretation = "Small gap (<5%): Camera selection is NOT the main issue"
    
    return {
        'accuracy_a': acc_a,
        'accuracy_b': acc_b,
        'delta_ab': delta,
        'interpretation': interpretation
    }


def print_final_summary(results: dict, output_path: str = None):
    """Print and optionally save final summary"""
    print("\n" + "="*70)
    print("FINAL EVALUATION SUMMARY")
    print("="*70)
    
    # Protocol comparison
    print("\n📊 PROTOCOL COMPARISON")
    print("-"*50)
    print(f"{'Protocol':<20} {'Accuracy':>12} {'Safety Acc':>12}")
    print("-"*50)
    
    for protocol in ['A', 'B', 'C', 'C_TOOL']:
        if protocol in results:
            acc = results[protocol]['results']['overall_accuracy']
            safety = results[protocol]['results'].get('safety_metrics', {}).get('accuracy', 0)
            print(f"Protocol {protocol:<14} {acc:>11.2%} {safety:>11.2%}")
    
    # Delta analysis
    if 'delta_analysis' in results:
        delta = results['delta_analysis']
        print(f"\n📈 ΔA→B ANALYSIS")
        print("-"*50)
        print(f"Protocol A Accuracy: {delta['accuracy_a']:.2%}")
        print(f"Protocol B Accuracy: {delta['accuracy_b']:.2%}")
        print(f"Camera Selection Gap: {delta['delta_ab']:+.2%}")
        print(f"Interpretation: {delta['interpretation']}")
    
    # Safety metrics
    if 'C' in results and 'safety_metrics' in results['C']['results']:
        safety = results['C']['results']['safety_metrics']
        print(f"\n🛡️ SAFETY METRICS (Protocol C)")
        print("-"*50)
        print(f"False Negative Rate (FNR): {safety.get('fnr', 0):.2%}")
        print(f"False Positive Rate (FPR): {safety.get('fpr', 0):.2%}")
        print(f"Collision Avoidance Rate: {safety.get('car', 0):.2%}")
        if 'abstention_rate' in safety:
            print(f"Abstention rate (INDETERMINATE): {safety['abstention_rate']:.2%} "
                  f"(on unsafe GT: {safety['abstained_on_unsafe']}, on safe GT: {safety['abstained_on_safe']})")
            print(f"FNR, abstain counted as miss: {safety['fnr_abstain_as_miss']:.2%}")
            print(f"CAR, abstain counted as miss: {safety['car_abstain_as_miss']:.2%}")

    # Process metrics
    if 'C' in results:
        res_c = results['C']['results']
        if 'avg_camera_coverage' in res_c:
            print(f"\n🔧 PROCESS METRICS (Protocol C)")
            print("-"*50)
            print(f"Camera Coverage: {res_c['avg_camera_coverage']:.2%}")
            print(f"Tool Precision: {res_c.get('avg_tool_precision', 0):.2%}")
            print(f"Tool Recall: {res_c.get('avg_tool_recall', 0):.2%}")
            print(f"Tool F1: {res_c['avg_tool_f1']:.2%}")
    
    # Save results
    if output_path:
        # Convert numpy values to Python types for JSON
        def convert(obj):
            import numpy as np
            if isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert(i) for i in obj]
            return obj
        
        results_clean = convert(results)
        
        with open(output_path, 'w') as f:
            json.dump(results_clean, f, indent=2)
        print(f"\n💾 Results saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Step 4: Run three-protocol evaluation')
    parser.add_argument('--model', '-m', type=str, default='dummy',
                        help='Model name (dummy, gpt-4o, qwen2-vl-2b, uavagent, etc.)')
    parser.add_argument('--test-set', '-t', type=str, default='data/processed/test_451.json',
                        help='Path to test set JSON')
    parser.add_argument('--image-root', type=str, default=None,
                        help='UAV3D v1.0 root (folder containing sweeps/); defaults to $UAV3D_ROOT')
    parser.add_argument('--protocols', '-p', type=str, default='A,B,C',
                        help='Protocols to run (comma-separated)')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='Output path for results JSON')
    parser.add_argument('--device', '-d', type=str, default='cuda',
                        help='Device (cuda/cpu)')
    parser.add_argument('--limit', '-l', type=int, default=None,
                        help='Limit number of samples (for testing)')
    
    args = parser.parse_args()
    
    protocols = [p.strip().upper() for p in args.protocols.split(',')]
    
    print("\n" + "="*70)
    print("STEP 4: THREE-PROTOCOL EVALUATION")
    print("="*70)
    print(f"Model: {args.model}")
    print(f"Test set: {args.test_set}")
    print(f"Protocols: {protocols}")
    print(f"Device: {args.device}")
    
    # Load test data
    print("\n📂 Loading test data...")
    samples = load_test_data(args.test_set, args.image_root)
    
    if args.limit:
        samples = samples[:args.limit]
        print(f"  Limited to {len(samples)} samples")
    else:
        print(f"  Loaded {len(samples)} samples")
    
    # Load model
    print(f"\n🤖 Loading model: {args.model}")
    model = get_model(args.model, args.device)
    
    # Initialize evaluator
    evaluator = MCUAVBenchEvaluator()
    
    # Run protocols
    results = {}
    start_time = time.time()
    
    if 'A' in protocols:
        results['A'] = run_protocol_a(model, samples, evaluator)
    
    if 'B' in protocols:
        results['B'] = run_protocol_b(model, samples, evaluator)
    
    if 'C' in protocols:
        results['C'] = run_protocol_c(model, samples, evaluator)

    if 'C_TOOL' in protocols:
        results['C_TOOL'] = run_protocol_c_tool(model, samples, evaluator)

    # Compute delta
    if 'A' in results and 'B' in results:
        results['delta_analysis'] = compute_delta_ab(results['A'], results['B'])
    
    elapsed = time.time() - start_time
    print(f"\n⏱️ Total time: {elapsed:.1f}s ({elapsed/len(samples):.2f}s/sample)")
    
    # Summary
    output_path = args.output
    if output_path is None:
        output_path = f"outputs/results/{args.model.replace('/', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    print_final_summary(results, output_path)
    
    print("\n" + "="*70)
    print("Evaluation complete!")
    print("="*70 + "\n")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
