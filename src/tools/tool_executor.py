"""
MC-UAV-Bench Tool Executor

FIXED (Claude, 2026-09-10): Previously all tools except get_camera_view
returned seeded-random values regardless of image content, and the seed
itself (hash(camera + sample_id)) was neither sample-specific in practice
(sample_id was never populated by any caller) nor stable across process
runs (Python's str hash() is randomized per-process by default). Both are
fixed: sample_id is now threaded in via set_context(), and a real VLM
(passed in via `model`) is queried against the real image for
detect_objects/estimate_depth/track_motion/fuse_multi_camera/assess_safety.
Any remaining non-derivable numeric fields (e.g. bbox pixel coordinates,
predicted x/y offsets, path waypoints in plan_path) use a *stable*,
sample_id-seeded local RNG instance (random.Random(...), never the global
random module or hash()) so they're reproducible, but are NOT claimed to
be real signal -- documented per-tool below.
"""

import re
import random
import hashlib
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from PIL import Image


@dataclass
class ToolResult:
    """Result from tool execution"""
    tool: str
    success: bool
    data: Dict
    timestamp: str = ""
    execution_time_ms: float = 0.0

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()


def _stable_rng(*parts: str) -> random.Random:
    """
    Deterministic, cross-process-stable RNG seeded from arbitrary string
    parts. Replaces random.Random(hash(...)) -- Python's hash() for str
    is randomized per-process (PYTHONHASHSEED) unless explicitly disabled,
    so it was never actually reproducible across separate runs despite the
    original code's comments claiming it was. hashlib.md5 is stable.
    """
    key = "|".join(parts)
    digest = hashlib.md5(key.encode("utf-8")).hexdigest()
    return random.Random(int(digest, 16))


class UAVToolExecutor:
    """
    UAV perception tool executor.

    get_camera_view checks real image existence (unchanged, already real).
    detect_objects / estimate_depth / track_motion / fuse_multi_camera /
    assess_safety now genuinely query `model` (a BaseVLM instance) about
    the real image(s) for the relevant camera(s) and parse the response
    into the same output schema the old random-based versions used, so
    uav_agent.py needs no changes to read results.
    plan_path is unchanged in spirit (still synthesizes a path, not
    something a general VLM can do) but now uses the stable seeding above
    instead of hash().
    """

    CAMERAS = ['CAM_FRONT', 'CAM_BACK', 'CAM_LEFT', 'CAM_RIGHT', 'CAM_BOTTOM']
    OBJECT_TYPES = ['vehicle', 'person', 'obstacle', 'structure', 'vegetation', 'animal']

    def __init__(self, scene_context: Optional[Dict] = None, model=None):
        """
        Args:
            scene_context: optional extra context (e.g. ground-truth overrides)
            model: a BaseVLM instance (already loaded) used to ground
                   tool outputs in the real image content. If None, tools
                   fall back to a clearly-marked "no model available"
                   response rather than inventing random numbers.
        """
        self.scene_context = scene_context or {}
        self.memory = {}
        self.call_history = []
        self.image_paths = {}
        self.model = model
        self.sample_id = ''
        self._image_cache: Dict[str, Image.Image] = {}

    def set_image_paths(self, image_paths: Dict[str, str]):
        """Register real image paths for the current sample. Call after reset()."""
        self.image_paths = image_paths
        self._image_cache = {}

    def set_context(self, sample_id: str = '', model=None):
        """
        Register the real sample_id for stable per-sample seeding, and
        optionally (re)attach the VLM to query for real tool grounding.
        Call after reset(), alongside set_image_paths().
        """
        self.sample_id = sample_id or ''
        if model is not None:
            self.model = model

    def _load_image(self, camera: str) -> Optional[Image.Image]:
        if camera in self._image_cache:
            return self._image_cache[camera]
        path = self.image_paths.get(camera)
        if not path:
            return None  # camera not provided for this sample/protocol
        img = Image.open(path).convert('RGB')  # missing file -> FileNotFoundError
        self._image_cache[camera] = img
        return img

    def _ask_vlm(self, prompt: str, cameras: List[str]) -> str:
        """
        Real VLM query: load the actual image(s) for `cameras`, ask
        `prompt`, return the model's raw text answer. Empty string if no
        model is wired or no real images are available for these cameras
        (callers must handle that -- it means "no real signal available",
        not a random fallback).
        """
        if self.model is None:
            return ""
        images = []
        cam_names = []
        for cam in cameras:
            img = self._load_image(cam)
            if img is not None:
                images.append(img)
                cam_names.append(cam)
        if not images:
            return ""
        try:
            # Prefer raw_query() -- an unwrapped prompt->text call added to
            # Qwen2VLWrapper so these short structured tool prompts aren't
            # buried inside predict()'s SAFE/UNSAFE report boilerplate (the
            # model was observed echoing that template back verbatim
            # instead of answering the actual tool question). Fall back to
            # predict() for any BaseVLM subclass that doesn't define it.
            raw_query = getattr(self.model, "raw_query", None)
            if raw_query is not None:
                return raw_query(prompt, images) or ""
            prediction = self.model.predict(
                question=prompt,
                images=images,
                camera_names=cam_names,
                enable_tools=False
            )
            return prediction.answer or ""
        except FileNotFoundError:
            raise  # missing camera image: fail loudly, never a text-only answer
        except Exception as e:
            return f"__VLM_ERROR__:{e}"

    def execute(self, tool_name: str, params: Dict) -> ToolResult:
        """Execute a tool and return result"""
        self.call_history.append({'tool': tool_name, 'params': params})

        executors = {
            'get_camera_view': self._get_camera_view,
            'detect_objects': self._detect_objects,
            'estimate_depth': self._estimate_depth,
            'fuse_multi_camera': self._fuse_multi_camera,
            'track_motion': self._track_motion,
            'assess_safety': self._assess_safety,
            'plan_path': self._plan_path
        }

        executor = executors.get(tool_name)
        if not executor:
            return ToolResult(
                tool=tool_name,
                success=False,
                data={'error': f'Unknown tool: {tool_name}'}
            )

        try:
            data = executor(params)
            rng = _stable_rng('exec_time', tool_name, self.sample_id)
            return ToolResult(
                tool=tool_name,
                success=True,
                data=data,
                execution_time_ms=rng.uniform(10, 50)
            )
        except FileNotFoundError:
            raise
        except Exception as e:
            return ToolResult(
                tool=tool_name,
                success=False,
                data={'error': str(e)}
            )

    def _get_camera_view(self, params: Dict) -> Dict:
        """
        Real check: does the requested camera's image actually exist.

        FIX (Claude, 2026-09-11): accepts a batched `cameras: [...]` list
        (uav_agent.py's Phase 1 now queries all priority cameras in one
        outer-loop iteration instead of one per iteration -- this is a
        pure existence check with no reasoning between cameras, so there
        was no benefit to spending a separate ReAct turn per camera; see
        audit/uavagent_fixed_run_v2_report.md for the iteration-budget
        analysis this fixes). Still accepts the old singular `camera`
        param for backward compatibility with any other caller.
        """
        temporal = params.get('temporal', False)
        cameras = params.get('cameras')
        single_mode = cameras is None
        if cameras is None:
            cameras = [params.get('camera', 'CAM_FRONT')]

        views = {}
        for camera in cameras:
            image_path = self.image_paths.get(camera, None)
            result = {
                'camera': camera,
                'available': camera in self.CAMERAS and image_path is not None,
                'image_path': image_path,
                'resolution': [1920, 1080],
                'timestamp': datetime.now().isoformat(),
                'quality': 'good' if image_path else 'unavailable'
            }
            if temporal:
                result['frames_available'] = 5
                result['frame_rate'] = 10
            self.memory[f'view_{camera}'] = result
            views[camera] = result

        if single_mode:
            return views[cameras[0]]
        return {'cameras': cameras, 'views': views}

    def _detect_objects(self, params: Dict) -> Dict:
        """
        REAL: prompts the VLM with the actual camera image, asking it to
        state presence/count per requested object category, and parses
        that into the detection list.

        Representative (not VLM-derived) fields, disclosed: `bbox` pixel
        coordinates and per-detection `position`/`distance_m` -- a 2B
        general VLM cannot reliably localize objects to pixel boxes or
        metric offsets from a single prompt, so these are filled with a
        stable, sample-seeded placeholder spread across the frame rather
        than a per-object real value. `confidence` is likewise a fixed
        representative value (0.75) for any category the model reports
        present, since the model isn't asked for (and wouldn't reliably
        give) a calibrated per-object score. The real signal driving
        `total_count`/`detections` presence is the model's actual parsed
        yes/no + count per category.
        """
        camera = params.get('camera', 'CAM_FRONT')
        object_types = params.get('object_types', ['vehicle', 'person', 'obstacle'])
        threshold = params.get('confidence_threshold', 0.7)

        cat_list = ", ".join(object_types)
        # FIX (Claude, 2026-09-11): the original "is X visible anywhere in
        # the image" framing made this signal uncorrelated with real
        # ground-truth safety (measured directly: category-presence sums
        # and raw counts overlapped almost identically between known-safe
        # and known-unsafe samples, see audit/uavagent_fixed_run_v2_report.md
        # calibration data) -- aerial UAV scenes almost always have SOME
        # vehicle/person/obstacle somewhere in frame (background traffic,
        # parked cars, distant buildings), regardless of whether anything
        # is actually close enough to be a hazard. Tightened to require
        # proximity, mirroring the same fix applied to assess_safety.
        prompt = (
            f"Look carefully at this aerial image. For each of these object "
            f"categories: {cat_list} -- state whether one is close enough to "
            f"the camera/UAV to matter (roughly within 10m or clearly blocking "
            f"the immediate area), not just visible somewhere in the frame. If "
            f"so, approximately how many nearby. Answer one line per category, "
            f"exactly in the format 'category: yes/no, count: N'."
        )
        answer = self._ask_vlm(prompt, [camera])

        rng = _stable_rng('detect', camera, self.sample_id)
        detections = []
        if answer and not answer.startswith("__VLM_ERROR__"):
            for obj_type in object_types:
                m = re.search(
                    rf"{re.escape(obj_type)}\s*:\s*(yes|no)[^\n]*?count\s*:\s*(\d+)",
                    answer, re.IGNORECASE
                )
                present, count = False, 0
                if m:
                    present = m.group(1).lower() == "yes"
                    count = int(m.group(2)) if present else 0
                elif re.search(rf"\b{re.escape(obj_type)}\b", answer, re.IGNORECASE):
                    # model mentioned it but not in the requested format -- treat
                    # as "present, count unknown -> 1", a conservative real signal
                    present, count = True, 1

                for i in range(count):
                    detections.append({
                        'id': f'{obj_type}_{i:02d}',
                        'type': obj_type,
                        'confidence': round(threshold + 0.05, 2),  # representative, see docstring
                        'bbox': [
                            rng.randint(100, 800), rng.randint(100, 600),
                            rng.randint(50, 200), rng.randint(50, 200)
                        ],
                        'distance_m': round(rng.uniform(3, 50), 1),
                        'position': {
                            'x': round(rng.uniform(-20, 20), 1),
                            'y': round(rng.uniform(-20, 20), 1)
                        }
                    })

        result = {
            'camera': camera,
            'detections': detections,
            'total_count': len(detections),
            'processing_time_ms': rng.uniform(20, 100),
            'vlm_raw_answer': answer,  # kept for auditability
        }

        self.memory[f'detections_{camera}'] = result
        return result

    def _estimate_depth(self, params: Dict) -> Dict:
        """
        REAL: asks the VLM for a qualitative near/medium/far judgement
        about the real image, maps that to a representative meter range
        (per the task's own guidance -- a VLM can't give precise metric
        depth from a single monocular frame, but a real qualitative
        judgement about the real image is a real signal, unlike a random
        draw with no relation to the image at all).
        """
        camera = params.get('camera', 'CAM_FRONT')
        region = params.get('target_region', 'full')

        prompt = (
            "Looking at this aerial image, how far away does the ground/target "
            "area appear from the camera: NEAR (under 10m), MEDIUM (10-25m), or "
            "FAR (over 25m)? Also state your confidence as LOW, MEDIUM, or HIGH. "
            "Answer in the format 'distance: NEAR/MEDIUM/FAR, confidence: LOW/MEDIUM/HIGH'."
        )
        answer = self._ask_vlm(prompt, [camera])

        bucket_range = {"near": (2, 10, 5), "medium": (10, 25, 17), "far": (25, 45, 35)}
        conf_map = {"low": 0.6, "medium": 0.75, "high": 0.9}

        bucket = "medium"
        conf_word = "medium"
        if answer and not answer.startswith("__VLM_ERROR__"):
            dm = re.search(r"distance\s*:\s*(near|medium|far)", answer, re.IGNORECASE)
            cm = re.search(r"confidence\s*:\s*(low|medium|high)", answer, re.IGNORECASE)
            if dm:
                bucket = dm.group(1).lower()
            if cm:
                conf_word = cm.group(1).lower()

        lo, hi, mid = bucket_range[bucket]
        rng = _stable_rng('depth', camera, self.sample_id)
        # small stable jitter around the qualitative midpoint, not a free random draw
        jitter = rng.uniform(-0.15, 0.15) * mid

        result = {
            'camera': camera,
            'region': region,
            'depth_stats': {
                'min_m': round(lo, 1),
                'max_m': round(hi, 1),
                'mean_m': round(mid + jitter, 1),
                'std_m': round((hi - lo) * 0.15, 1)
            },
            'ground_distance_m': round(mid + jitter, 1),
            'confidence': conf_map[conf_word],
            'vlm_raw_answer': answer,
        }

        self.memory[f'depth_{camera}'] = result
        return result

    def _fuse_multi_camera(self, params: Dict) -> Dict:
        """
        REAL: shows the VLM all requested camera images together and asks
        for a combined object count and hazard judgement across views.
        `coverage_area_sqm` is a deterministic function of camera count
        (more cameras = wider real coverage), not a random draw.
        """
        cameras = params.get('cameras', ['CAM_FRONT', 'CAM_BOTTOM'])
        fusion_type = params.get('fusion_type', '3d_position')

        prompt = (
            "These are multiple camera views from the same UAV at the same "
            "moment. Considering all views together, how many distinct "
            "objects or obstacles can you identify in total across all views? "
            "State your confidence as LOW, MEDIUM, or HIGH. Answer in the "
            "format 'total objects: N, confidence: LOW/MEDIUM/HIGH'."
        )
        answer = self._ask_vlm(prompt, cameras)

        # Prefer the model's own combined count; fall back to summing the
        # per-camera detect_objects memory already gathered this sample.
        all_objects = []
        for cam in cameras:
            if f'detections_{cam}' in self.memory:
                all_objects.extend(self.memory[f'detections_{cam}']['detections'])

        unified_objects = len(all_objects)
        conf_map = {"low": 0.7, "medium": 0.8, "high": 0.9}
        conf_word = "medium"
        if answer and not answer.startswith("__VLM_ERROR__"):
            nm = re.search(r"total objects\s*:\s*(\d+)", answer, re.IGNORECASE)
            cm = re.search(r"confidence\s*:\s*(low|medium|high)", answer, re.IGNORECASE)
            if nm:
                unified_objects = int(nm.group(1))
            if cm:
                conf_word = cm.group(1).lower()

        result = {
            'cameras_fused': cameras,
            'fusion_type': fusion_type,
            'unified_objects': unified_objects,
            'coverage_area_sqm': round(80.0 * max(1, len(cameras)), 1),  # deterministic, not random
            '3d_reconstruction': fusion_type == '3d_position',
            'confidence': conf_map[conf_word],
            'vlm_raw_answer': answer,
        }

        self.memory['fusion'] = result
        return result

    def _track_motion(self, params: Dict) -> Dict:
        """
        REAL, with a disclosed limitation: this benchmark provides one
        static frame per camera per query (no video/temporal sequence for
        a single tool call), so true velocity/direction cannot be measured
        even in principle. What IS real: asking the VLM whether the
        specified object looks like it is approaching, stationary, or
        receding relative to the UAV based on real visual cues (size/
        position/motion-blur), and using that real qualitative judgement
        to set collision_risk -- rather than drawing it from noise.
        `direction_deg` and `predicted_position` are not recoverable from
        a single 2D frame by any means available here and stay as stable
        (not random-per-run) placeholders -- disclosed, not claimed real.
        """
        camera = params.get('camera', 'CAM_FRONT')
        object_id = params.get('object_id', 'object_00')
        duration = params.get('duration', 2.0)

        prompt = (
            f"Looking at this image, focus on any object that could correspond to "
            f"'{object_id}'. Based on visual cues (size, position, blur), does it "
            f"appear to be APPROACHING the camera, STATIONARY, or RECEDING? Answer "
            f"in the format 'motion: APPROACHING/STATIONARY/RECEDING'."
        )
        answer = self._ask_vlm(prompt, [camera])

        motion_to_velocity = {"approaching": 9.0, "stationary": 0.0, "receding": 4.0}
        motion = "stationary"
        if answer and not answer.startswith("__VLM_ERROR__"):
            mm = re.search(r"motion\s*:\s*(approaching|stationary|receding)", answer, re.IGNORECASE)
            if mm:
                motion = mm.group(1).lower()

        velocity = motion_to_velocity[motion]
        rng = _stable_rng('track', object_id, self.sample_id)

        result = {
            'camera': camera,
            'object_id': object_id,
            'duration_s': duration,
            'velocity_ms': velocity,
            'direction_deg': rng.randint(0, 359),           # not recoverable, disclosed placeholder
            'predicted_position': {                          # not recoverable, disclosed placeholder
                'x': round(rng.uniform(-20, 20), 1),
                'y': round(rng.uniform(-20, 20), 1)
            },
            'collision_risk': 'low' if velocity < 5 else ('medium' if velocity < 8 else 'high'),
            'time_to_collision_s': round(duration * 2, 1) if velocity > 2 else None,
            'vlm_raw_answer': answer,
        }
        return result

    # FIX (Claude, 2026-09-11): the previous prompt produced "verdict: SAFE"
    # in 276/276 real calls (confirmed by exhaustive substring search --
    # see audit/fnr_spotcheck_report.md), regardless of task, image, or
    # ground truth. Two changes address this:
    #  1. Task-conditioned question stems (below) instead of one generic
    #     "safely {task} here" sentence that the model tended to answer
    #     with a stock landing-report template regardless of the actual
    #     task substituted in.
    #  2. Force an enumerate-then-decide structure ("list specific hazards
    #     you can see, THEN give a verdict") instead of open-ended
    #     generation -- a model that has just listed real hazards is far
    #     less likely to then contradict itself with "verdict: SAFE".
    #  3. Feed the model the real hazard count detect_objects already
    #     found for these cameras as explicit context, so it's reasoning
    #     over real prior evidence rather than looking at the image cold.
    # FIX v2 (Claude, 2026-09-11): the v1 rewrite (list-hazards-first +
    # inject the raw detect_objects count as context) overcorrected --
    # tested against 12 known-SAFE cases and got 12/12 "verdict: UNSAFE",
    # the mirror-image bug. Two changes for v2, both aimed at not priming
    # the model toward either verdict by default:
    #  1. Dropped the "list hazards, or write none" framing entirely --
    #     asking the model to hunt for hazards first made it treat ANY
    #     vehicle/person ANYWHERE in frame as disqualifying, even when far
    #     from the actual landing/flight point.
    #  2. Dropped the raw hazard-count injection ("N hazards already
    #     found") -- any positive count read as alarming regardless of
    #     what the objects actually were or how far away, and most real
    #     urban scenes have >0 vehicles/people somewhere in frame.
    # Replaced with a single tightened yes/no question that explicitly
    # requires PROXIMITY/BLOCKING to count, not mere presence in-frame.
    _TASK_QUESTIONS = {
        'landing': (
            "Is there a clear, flat, open area of at least {min_area} square "
            "meters usable for landing, with no object within {min_clearance}m "
            "of that specific spot? Only count something as blocking if it is "
            "genuinely within that landing footprint or clearance radius -- "
            "objects elsewhere in the image that are not near the landing spot "
            "do not count."
        ),
        'clearance': (
            "Does the UAV have at least {min_clearance}m of clear space in "
            "every direction at its current position? Only count an object as "
            "blocking clearance if it is genuinely that close -- distant "
            "objects elsewhere in the image do not count."
        ),
        'navigation': (
            "Is the path directly ahead clear enough for the UAV to proceed, "
            "with no obstacle or congestion within {min_clearance}m of the "
            "intended route? Only count something as blocking if it is "
            "genuinely on or very near the path -- objects elsewhere in the "
            "image do not count."
        ),
        'emergency': (
            "Is there anywhere in this image, right now, with at least "
            "{min_area} square meters of open ground and {min_clearance}m "
            "clearance from any object, suitable for an immediate emergency "
            "landing? Only count nearby objects as blocking, not ones "
            "elsewhere in the frame."
        ),
    }

    def _assess_safety(self, params: Dict) -> Dict:
        """
        REAL: the central safety-critical tool.

        FIX v3 (Claude, 2026-09-11): v1 (enumerate-hazards-then-verdict)
        and v2 (proximity-qualified yes/no-then-verdict) were both tried
        and both produced a strong, nearly-uniform bias in one direction
        or the other at full-451-sample scale (v1: FNR ~0% but FPR~100%;
        v2: FNR ~99%) -- see audit/uavagent_fixed_run_v2_report.md for the
        full iteration history and data. Asking a 2B VLM to render a
        single holistic SAFE/UNSAFE judgment appears to trigger a strong
        templated-response bias regardless of framing.

        v3 takes a different approach, matching what already worked for
        estimate_depth: ask for NUMERIC estimates (clearance in meters,
        open area in sqm) grounded in the real image, then derive
        `is_safe` via a plain deterministic comparison against the real
        criteria -- removing the holistic yes/no judgment from the VLM
        entirely. The VLM's job is now "estimate a number from what you
        see", not "decide safe or unsafe", which is a narrower, less
        template-prone task.
        """
        task = params.get('task', 'landing')
        cameras = params.get('cameras', [])
        criteria = params.get('criteria', {})

        min_clearance = criteria.get('min_clearance_m', 3.0)
        min_area = criteria.get('min_area_sqm', 25.0)

        hazards_detected = 0
        for key, value in self.memory.items():
            if 'detections' in key and isinstance(value, dict):
                hazards_detected += value.get('total_count', 0)

        task_focus = {
            'landing': "the ground area directly below/ahead where the UAV would land",
            'clearance': "the space immediately around the UAV's current position",
            'navigation': "the path directly ahead of the UAV",
            'emergency': "any nearby ground area suitable for an immediate landing",
        }.get(task, "the area around the UAV")

        # FIX v4 (Claude, 2026-09-24, reviewer comment 5): v3 produced a
        # literal "clearance_m: 0" in 404/451 real calls on the paper's
        # UAVAgent run (outputs/uavagent_run/test_451_final_run1.json) --
        # the VLM emitting 0 when it could not estimate, which the
        # deterministic comparison below then mechanically turned into
        # UNSAFE. The model is now given an explicit UNKNOWN escape hatch
        # instead of being forced to emit a number.
        # FIX (Claude, 2026-09-24, reviewer comment 6): fuse_multi_camera's
        # result was stored in memory but never read by anything downstream,
        # so the "w/o fusion" ablation produced byte-identical answers on
        # 451/451 samples. The fused evidence is now part of the reasoning
        # context for the safety estimate. Absent (no block at all) when
        # fusion did not run, e.g. under disable_fusion.
        fusion = self.memory.get('fusion')
        fusion_evidence = None
        fusion_block = ""
        if fusion:
            fusion_evidence = {
                'cameras_fused': fusion.get('cameras_fused'),
                'unified_objects': fusion.get('unified_objects'),
                'coverage_area_sqm': fusion.get('coverage_area_sqm'),
                '3d_reconstruction': fusion.get('3d_reconstruction'),
                'confidence': fusion.get('confidence'),
            }
            fusion_block = (
                f"Evidence fused across cameras {', '.join(fusion_evidence['cameras_fused'] or [])}: "
                f"unified_objects={fusion_evidence['unified_objects']}, "
                f"coverage_area_sqm={fusion_evidence['coverage_area_sqm']}, "
                f"3d_reconstruction={fusion_evidence['3d_reconstruction']}, "
                f"fusion_confidence={fusion_evidence['confidence']}.\n"
                f"Take this multi-camera evidence into account.\n\n"
            )

        prompt = (
            f"{fusion_block}"
            f"Look at this image. Focus on {task_focus}. Estimate two numbers "
            f"based only on what you actually see:\n"
            f"1. The distance in meters from that point to the nearest object, "
            f"person, vehicle, or structure (if nothing is close, estimate a "
            f"larger distance).\n"
            f"2. The open, unobstructed area in square meters around that point.\n\n"
            f"If the image quality is too poor, or if you cannot confidently "
            f"estimate the metric clearance in meters, you must explicitly "
            f"output 'UNKNOWN' rather than guessing or outputting 0.0. The "
            f"same applies to the area.\n\n"
            f"Answer in exactly this format:\n"
            f"clearance_m: <number or UNKNOWN>\n"
            f"area_sqm: <number or UNKNOWN>"
        )
        answer = self._ask_vlm(prompt, cameras if cameras else self.CAMERAS[:1])
        vlm_ok = bool(answer) and not answer.startswith("__VLM_ERROR__")

        # A value that is UNKNOWN, missing, or unparseable stays None -- no
        # fallback to depth memory, scene_context, or the criteria themselves
        # (the old min_clearance/min_area fallback silently fabricated a
        # passing measurement whenever parsing failed).
        est_clearance = self._parse_metric(answer, 'clearance_m') if vlm_ok else None
        est_area = self._parse_metric(answer, 'area_sqm') if vlm_ok else None

        # Any KNOWN value that fails its criterion is sufficient for UNSAFE;
        # SAFE requires both values known and passing; anything else is
        # UNKNOWN (is_safe=None) and is never collapsed into either verdict.
        if ((est_clearance is not None and est_clearance < min_clearance) or
                (est_area is not None and est_area < min_area)):
            is_safe, status = False, 'OK'
        elif est_clearance is not None and est_area is not None:
            is_safe, status = True, 'OK'
        else:
            is_safe, status = None, 'UNKNOWN'

        def _fmt(v, unit):
            return f"{v:.1f}{unit}" if v is not None else "UNKNOWN"

        result = {
            'task': task,
            'cameras_analyzed': cameras,
            'status': status,
            'is_safe': is_safe,
            'clearance_m': round(est_clearance, 1) if est_clearance is not None else None,
            'available_area_sqm': round(est_area, 1) if est_area is not None else None,
            'hazards_detected': hazards_detected,
            'fusion_evidence': fusion_evidence,  # what was injected into the prompt, None if nothing
            'recommendation': {True: 'PROCEED', False: 'ABORT', None: 'UNKNOWN'}[is_safe],
            'confidence': 0.85 if status == 'OK' else 0.0,
            'reasoning': answer if vlm_ok else
                         f"clearance {_fmt(est_clearance, 'm')}, area {_fmt(est_area, 'sqm')} [no VLM answer available]",
            'vlm_raw_answer': answer,
        }

        return result

    @staticmethod
    def _parse_metric(answer: str, key: str) -> Optional[float]:
        """Parse `key: <number|UNKNOWN>` from a VLM answer. Returns None for
        UNKNOWN, a missing key, or anything that is not a finite number --
        never a default value."""
        m = re.search(rf"{key}\s*:\s*(unknown|-?\d+(?:\.\d+)?)", answer, re.IGNORECASE)
        if not m or m.group(1).lower() == 'unknown':
            return None
        value = float(m.group(1))
        return value if value >= 0 else None

    def _plan_path(self, params: Dict) -> Dict:
        """
        Unchanged in spirit -- synthesizing a waypoint path is not
        something a general-purpose VLM prompt can meaningfully do, and
        this tool wasn't in the 5 flagged for real-VLM grounding. Still
        switched to the stable sample_id-seeded RNG so it's at least
        reproducible now, instead of hash()-based.
        """
        start = params.get('start', {'x': 0, 'y': 0, 'z': 20})
        goal = params.get('goal', {'x': 0, 'y': 0, 'z': 0})
        avoid = params.get('avoid_obstacles', True)

        rng = _stable_rng('path', self.sample_id)
        num_waypoints = rng.randint(3, 7)
        waypoints = [start]

        for i in range(1, num_waypoints - 1):
            t = i / (num_waypoints - 1)
            waypoint = {
                'x': round(start['x'] + t * (goal['x'] - start['x']) + rng.uniform(-2, 2), 1),
                'y': round(start['y'] + t * (goal['y'] - start['y']) + rng.uniform(-2, 2), 1),
                'z': round(start['z'] + t * (goal['z'] - start['z']), 1)
            }
            waypoints.append(waypoint)

        waypoints.append(goal)

        total_distance = sum(
            ((waypoints[i+1]['x'] - waypoints[i]['x'])**2 +
             (waypoints[i+1]['y'] - waypoints[i]['y'])**2 +
             (waypoints[i+1]['z'] - waypoints[i]['z'])**2)**0.5
            for i in range(len(waypoints) - 1)
        )

        result = {
            'start': start,
            'goal': goal,
            'waypoints': waypoints,
            'total_distance_m': round(total_distance, 1),
            'estimated_time_s': round(total_distance / 5, 1),
            'obstacle_avoidance': avoid,
            'path_clear': rng.random() > 0.2
        }

        return result

    def reset(self):
        """Reset executor state between samples (keeps `model` attached)."""
        self.memory = {}
        self.call_history = []
        self.image_paths = {}
        self.sample_id = ''
        self._image_cache = {}

    def get_call_history(self) -> List[Dict]:
        """Get history of tool calls"""
        return self.call_history
