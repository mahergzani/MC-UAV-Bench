"""
API-Based Model Wrappers

Supports: GPT-4o, Claude 3.5 Sonnet
"""

import os
import json
import base64
import io
from typing import Dict, List, Optional
from PIL import Image

from .base_model import BaseVLM, ModelPrediction


class GPT4VisionWrapper(BaseVLM):
    """Wrapper for GPT-4o API"""

    def __init__(
        self,
        model_name: str = "gpt-4o",
        api_key: Optional[str] = None
    ):
        super().__init__(model_name, device="api")
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.client = None

    def load_model(self):
        if not self.api_key:
            raise ValueError(
                "OpenAI API key not found. Set the OPENAI_API_KEY environment variable:\n"
                "  export OPENAI_API_KEY=your_key_here"
            )
        from openai import OpenAI
        self.client = OpenAI(api_key=self.api_key)
        print(f"✓ OpenAI client initialized for {self.model_name}")

    def _encode_image(self, image: Image.Image) -> str:
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")

    def predict(
        self,
        question: str,
        images: List[Image.Image],
        camera_names: List[str],
        enable_tools: bool = False
    ) -> ModelPrediction:

        if self.client is None:
            self.load_model()

        content = []
        for img, cam in zip(images, camera_names):
            base64_img = self._encode_image(img)
            content.append({"type": "text", "text": f"[{cam}]:"})
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_img}",
                    "detail": "high"
                }
            })

        content.append({"type": "text", "text": self.build_prompt(question, camera_names, enable_tools)})

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": content}],
            max_tokens=1024,
            temperature=0.0
        )
        return self.parse_response(response.choices[0].message.content)

    def predict_with_tools(
        self,
        question: str,
        available_cameras: List[str],
        image_paths: Dict[str, str],
        tool_executor,
        max_iterations: int = 5
    ) -> ModelPrediction:

        if self.client is None:
            self.load_model()

        # FIX: use relative import consistent with rest of codebase
        from ..tools.tool_registry import get_openai_tools_schema

        images = {}
        for cam, path in image_paths.items():
            if cam in available_cameras:
                images[cam] = Image.open(path).convert('RGB')

        content = []
        for cam, img in images.items():
            base64_img = self._encode_image(img)
            content.append({"type": "text", "text": f"[{cam}]:"})
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{base64_img}"}
            })
        content.append({
            "type": "text",
            "text": self.build_prompt(question, list(images.keys()), True)
        })

        messages = [{"role": "user", "content": content}]
        tools = get_openai_tools_schema()
        tool_trace = []
        cameras_used = set()

        for iteration in range(max_iterations):
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                tools=tools,
                tool_choice="auto",
                max_tokens=1024,
                temperature=0.0
            )

            msg = response.choices[0].message

            if msg.tool_calls is None:
                prediction = self.parse_response(msg.content or "")
                prediction.tool_calls = tool_trace
                prediction.cameras_used = list(cameras_used)
                return prediction

            # FIX: must append msg as dict, not object — OpenAI SDK returns
            # ChatCompletionMessage object which is not JSON-serialisable for
            # subsequent API calls; convert to dict format the API expects
            messages.append(msg)

            for tool_call in msg.tool_calls:
                func_name = tool_call.function.name
                func_args = json.loads(tool_call.function.arguments)

                if 'camera' in func_args:
                    cameras_used.add(func_args['camera'])
                if 'cameras' in func_args:
                    cameras_used.update(func_args['cameras'])

                result = tool_executor.execute(func_name, func_args)

                tool_trace.append({
                    'tool': func_name,
                    'params': func_args,
                    'result': result.data if result.success else {'error': 'failed'}
                })

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(result.data)
                })

        # Max iterations reached — return best answer from last non-tool response
        return ModelPrediction(
            answer="Max iterations reached",
            tool_calls=tool_trace,
            cameras_used=list(cameras_used)
        )


class ClaudeWrapper(BaseVLM):
    """Wrapper for Claude API"""

    def __init__(
        self,
        model_name: str = "claude-3-5-sonnet-20241022",
        api_key: Optional[str] = None
    ):
        super().__init__(model_name, device="api")
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self.client = None

    def load_model(self):
        if not self.api_key:
            raise ValueError(
                "Anthropic API key not found. Set the ANTHROPIC_API_KEY environment variable:\n"
                "  export ANTHROPIC_API_KEY=your_key_here"
            )
        import anthropic
        self.client = anthropic.Anthropic(api_key=self.api_key)
        print(f"✓ Anthropic client initialized for {self.model_name}")

    def _encode_image(self, image: Image.Image) -> tuple:
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        return "image/jpeg", base64.b64encode(buffered.getvalue()).decode("utf-8")

    def predict(
        self,
        question: str,
        images: List[Image.Image],
        camera_names: List[str],
        enable_tools: bool = False
    ) -> ModelPrediction:

        if self.client is None:
            self.load_model()

        content = []
        for img, cam in zip(images, camera_names):
            media_type, base64_img = self._encode_image(img)
            content.append({"type": "text", "text": f"[{cam}]:"})
            content.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": media_type,
                    "data": base64_img
                }
            })

        content.append({"type": "text", "text": self.build_prompt(question, camera_names, enable_tools)})

        response = self.client.messages.create(
            model=self.model_name,
            max_tokens=1024,
            messages=[{"role": "user", "content": content}]
        )
        return self.parse_response(response.content[0].text)

    def predict_with_tools(
        self,
        question: str,
        available_cameras: List[str],
        image_paths: Dict[str, str],
        tool_executor,
        max_iterations: int = 5
    ) -> ModelPrediction:
        """
        FIX: ClaudeWrapper was missing predict_with_tools entirely.
        Without this, Protocol C silently fell back to BaseVLM.predict_with_tools
        which loads all images at once with no agentic tool calling — Protocol C
        results for Claude were identical to Protocol A.
        """
        if self.client is None:
            self.load_model()

        from ..tools.tool_registry import get_all_tools

        # Build Anthropic tool schema from registry
        tools = []
        for tool in get_all_tools():
            schema = tool.to_openai_schema()['function']
            tools.append({
                "name": schema['name'],
                "description": schema['description'],
                "input_schema": schema['parameters']
            })

        # Load initial images
        images = {}
        for cam, path in image_paths.items():
            if cam in available_cameras:
                images[cam] = Image.open(path).convert('RGB')

        content = []
        for cam, img in images.items():
            media_type, base64_img = self._encode_image(img)
            content.append({"type": "text", "text": f"[{cam}]:"})
            content.append({
                "type": "image",
                "source": {"type": "base64", "media_type": media_type, "data": base64_img}
            })
        content.append({
            "type": "text",
            "text": self.build_prompt(question, list(images.keys()), True)
        })

        messages = [{"role": "user", "content": content}]
        tool_trace = []
        cameras_used = set(images.keys())

        for iteration in range(max_iterations):
            response = self.client.messages.create(
                model=self.model_name,
                max_tokens=1024,
                tools=tools,
                messages=messages
            )

            # Check stop reason
            if response.stop_reason == "end_turn":
                text = next((b.text for b in response.content if hasattr(b, 'text')), "")
                prediction = self.parse_response(text)
                prediction.tool_calls = tool_trace
                prediction.cameras_used = list(cameras_used)
                prediction.termination_reason = "answered"
                return prediction

            # Process tool use blocks
            tool_use_blocks = [b for b in response.content if b.type == "tool_use"]
            if not tool_use_blocks:
                text = next((b.text for b in response.content if hasattr(b, 'text')), "")
                prediction = self.parse_response(text)
                prediction.tool_calls = tool_trace
                prediction.cameras_used = list(cameras_used)
                prediction.termination_reason = "answered"
                return prediction

            # Append assistant turn
            messages.append({"role": "assistant", "content": response.content})

            # Execute each tool and build user tool_result turn
            tool_results = []
            for block in tool_use_blocks:
                func_name = block.name
                func_args = block.input

                if 'camera' in func_args:
                    cameras_used.add(func_args['camera'])
                if 'cameras' in func_args:
                    cameras_used.update(func_args['cameras'])

                result = tool_executor.execute(func_name, func_args)

                tool_trace.append({
                    'tool': func_name,
                    'params': func_args,
                    'result': result.data if result.success else {'error': 'failed'}
                })

                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": json.dumps(result.data)
                })

            messages.append({"role": "user", "content": tool_results})

        return ModelPrediction(
            answer="Max iterations reached",
            tool_calls=tool_trace,
            cameras_used=list(cameras_used),
            termination_reason="budget_exhausted"
        )


def get_api_model(model_name: str, **kwargs) -> BaseVLM:
    """Factory function for API models"""
    model_lower = model_name.lower()

    if 'gpt' in model_lower:
        return GPT4VisionWrapper(model_name, **kwargs)
    elif 'claude' in model_lower:
        return ClaudeWrapper(model_name, **kwargs)
    else:
        raise ValueError(f"Unknown API model: {model_name}. Supported: gpt-*, claude-*")