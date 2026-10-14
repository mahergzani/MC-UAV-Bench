"""
Base Model Classes for MC-UAV-Bench Evaluation

All model wrappers should inherit from BaseVLM.
"""

import re
import random
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from PIL import Image


@dataclass
class ModelPrediction:
    """Standard prediction output format"""
    answer: str
    confidence: float = 0.0
    tool_calls: List[Dict] = None
    cameras_used: List[str] = None
    reasoning: str = ""
    raw_output: str = ""
    # Protocol C (tool-augmented) episode bookkeeping — "" / 0 for anything
    # that doesn't run a tool episode (Protocol A/B, or Protocol C without
    # this instrumentation), so this is additive and doesn't affect existing
    # callers or serialized output shape.
    termination_reason: str = ""  # "answered" / "budget_exhausted" / "error"
    parse_failures: int = 0

    def __post_init__(self):
        if self.tool_calls is None:
            self.tool_calls = []
        if self.cameras_used is None:
            self.cameras_used = []

    def to_dict(self) -> Dict:
        return {
            'answer': self.answer,
            'confidence': self.confidence,
            'tool_calls': self.tool_calls,
            'cameras_used': self.cameras_used,
            'reasoning': self.reasoning,
            'termination_reason': self.termination_reason,
            'parse_failures': self.parse_failures
        }


class BaseVLM(ABC):
    """Abstract base class for Vision-Language Models"""

    def __init__(self, model_name: str, device: str = "cuda"):
        self.model_name = model_name
        self.device = device
        self.model = None
        self.processor = None

    @abstractmethod
    def load_model(self):
        """Load model and processor"""
        pass

    @abstractmethod
    def predict(
        self,
        question: str,
        images: List[Image.Image],
        camera_names: List[str],
        enable_tools: bool = False
    ) -> ModelPrediction:
        """
        Generate prediction for a single sample.

        Args:
            question: The question to answer
            images: List of PIL images
            camera_names: Names of cameras corresponding to images
            enable_tools: Whether tool calling is enabled

        Returns:
            ModelPrediction with answer and metadata
        """
        pass

    def predict_with_tools(
        self,
        question: str,
        available_cameras: List[str],
        image_paths: Dict[str, str],
        tool_executor,
        max_iterations: int = 5
    ) -> ModelPrediction:
        """
        Default fallback for models without native tool support.
        Loads all available images and calls predict() directly.
        Override in subclasses for ReAct-style tool use (e.g. UAVAgent).
        """
        images = []
        cameras = []
        for cam, path in image_paths.items():
            if cam in available_cameras:
                images.append(Image.open(path).convert('RGB'))
                cameras.append(cam)

        return self.predict(question, images, cameras, enable_tools=False)

    def build_prompt(
        self,
        question: str,
        camera_names: List[str],
        include_tool_instructions: bool = False
    ) -> str:
        """Build prompt for the model"""
        camera_list = ", ".join(camera_names)

        prompt = f"""You are an AI assistant for UAV (drone) autonomous systems. 
You have access to {len(camera_names)} camera views: {camera_list}.

Analyze the provided camera images and answer the following question about UAV safety and navigation.

Question: {question}

Provide your answer in the following format:
1. First, describe what you observe in the relevant cameras
2. Then provide your safety assessment
3. Include specific measurements (distances in meters, areas in sqm) when relevant
4. End with a clear decision: SAFE or UNSAFE with justification

Answer:"""

        if include_tool_instructions:
            prompt += """

You can use the following tools:
- get_camera_view(camera): Get image from specific camera
- detect_objects(camera, types): Detect objects in camera view
- estimate_depth(camera): Estimate distances in camera view
- assess_safety(task, cameras): Assess safety for the task

Call tools using: <tool>tool_name(params)</tool>
When done, provide final answer in: <answer>Your final answer</answer>"""

        return prompt

    def parse_response(self, response: str) -> ModelPrediction:
        """Parse model response into structured prediction"""
        answer = response.strip()

        tool_calls = []
        tool_pattern = r'<tool>(\w+)\((.*?)\)</tool>'
        for match in re.finditer(tool_pattern, response):
            tool_calls.append({
                'tool': match.group(1),
                'params': match.group(2)
            })

        answer_match = re.search(r'<answer>(.*?)</answer>', response, re.DOTALL)
        if answer_match:
            answer = answer_match.group(1).strip()

        cameras_used = []
        for cam in ['CAM_FRONT', 'CAM_BACK', 'CAM_LEFT', 'CAM_RIGHT', 'CAM_BOTTOM']:
            if cam in response:
                cameras_used.append(cam)

        return ModelPrediction(
            answer=answer,
            tool_calls=tool_calls,
            cameras_used=cameras_used,
            raw_output=response
        )

    def __repr__(self):
        return f"{self.__class__.__name__}(model={self.model_name})"


class DummyVLM(BaseVLM):
    """Dummy model for testing — returns deterministic predictions"""

    def __init__(self):
        super().__init__("dummy", device="cpu")

    def load_model(self):
        pass

    def predict(
        self,
        question: str,
        images: List[Image.Image],
        camera_names: List[str],
        enable_tools: bool = False
    ) -> ModelPrediction:
        # FIX: seed per question so same input always gives same output
        # non-deterministic random makes step2 dummy checks non-reproducible
        rng = random.Random(hash(question))
        safety = rng.choice(['SAFE', 'UNSAFE'])
        distance = rng.randint(5, 30)
        area = rng.randint(20, 100)

        answer = f"{safety} - Clear area of {area} sqm detected. Nearest obstacle at {distance}m."

        # FIX: filter out empty dicts — when camera_names is empty the original code
        # produced [{'tool': 'get_camera_view', 'params': ...}, {}, {}] which breaks
        # Protocol C tool_f1 metric calculation downstream
        tool_calls = []
        if camera_names:
            tool_calls = [
                {'tool': 'get_camera_view', 'params': {'camera': camera_names[0]}},
                {'tool': 'detect_objects',  'params': {'camera': camera_names[0]}},
                {'tool': 'assess_safety',   'params': {'task': 'landing'}}
            ]

        return ModelPrediction(
            answer=answer,
            confidence=rng.uniform(0.6, 0.95),
            cameras_used=camera_names[:2] if camera_names else [],
            tool_calls=tool_calls
        )

    def predict_with_tools(
        self,
        question: str,
        available_cameras: List[str],
        image_paths: Dict[str, str],
        tool_executor,
        max_iterations: int = 5
    ) -> ModelPrediction:
        # FIX: override predict_with_tools so DummyVLM returns proper tool_calls
        # and cameras_used for Protocol C metrics — base fallback returns empty both
        cameras = [c for c in available_cameras if c in image_paths][:2]
        result = self.predict(question, [], cameras, enable_tools=False)
        result.cameras_used = cameras
        return result