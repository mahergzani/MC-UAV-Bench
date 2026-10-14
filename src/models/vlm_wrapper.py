"""
VLM Wrappers for MC-UAV-Bench Evaluation
"""

import os
import base64
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional
from PIL import Image

from .base_model import BaseVLM, ModelPrediction


class Qwen2VLWrapper(BaseVLM):

    def __init__(self, model_name="Qwen/Qwen2-VL-2B-Instruct", device="cuda", quantize=True):
        super().__init__(model_name, device)
        self.quantize = quantize

    def load_model(self):
        from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
        import torch

        print(f"Loading {self.model_name}...")

        load_kwargs = {"device_map": "cuda:0", "trust_remote_code": True}
        if self.quantize:
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16
            )
        else:
            load_kwargs["torch_dtype"] = torch.float16

        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            self.model_name, **load_kwargs
        )
        self.processor = AutoProcessor.from_pretrained(
            self.model_name,
            min_pixels=64 * 28 * 28,
            max_pixels=640 * 28 * 28
        )
        print(f"✓ Loaded {self.model_name}")

    # def predict(self, question, images, camera_names, enable_tools=False):
    #     import torch
    #     import tempfile
    #     import os
    #     from qwen_vl_utils import process_vision_info

    #     if self.model is None:
    #         self.load_model()

    #     if not images:
    #         return ModelPrediction(answer="No images available", raw_output="")

    #     # Build 28-aligned combined image — both dims must be divisible by 28
    #     # 224 = 8×28, 896 = 32×28
    #     CELL_H, CELL_W = 224, 224
    #     resized = [img.convert('RGB').resize((CELL_W, CELL_H)) for img in images]
    #     combined = Image.new('RGB', (CELL_W * len(resized), CELL_H))
    #     x = 0
    #     for img in resized:
    #         combined.paste(img, (x, 0))
    #         x += CELL_W
    #     if combined.width > 896:
    #         # FIX: explicit resize to 28-aligned dims — never use thumbnail() which
    #         # produces non-28-aligned dimensions and causes cudaErrorIllegalAddress
    #         combined = combined.resize((896, 224))

    #     tmp = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
    #     combined.save(tmp.name, quality=95)
    #     tmp.close()

    #     try:
    #         cam_labels = ", ".join(camera_names)
    #         messages = [{
    #             "role": "user",
    #             "content": [
    #                 {
    #                     "type": "image",
    #                     # FIX: Path.as_uri() always produces correct file:///path format
    #                     # file:// (two slashes) is malformed and causes process_vision_info
    #                     # to silently return empty list → tensor b (0) mismatch → CUDA crash
    #                     "image": Path(tmp.name).as_uri(),
    #                     "resized_height": combined.height,
    #                     "resized_width": combined.width
    #                 },
    #                 {
    #                     "type": "text",
    #                     "text": f"Cameras shown left to right: {cam_labels}\n{self.build_prompt(question, camera_names, enable_tools)}"
    #                 }
    #             ]
    #         }]

    #         text = self.processor.apply_chat_template(
    #             messages, tokenize=False, add_generation_prompt=True
    #         )

    #         # Verify template inserted vision tokens — if not, pixel_values will be empty
    #         assert "<|vision_start|>" in text, \
    #             "Chat template did not insert vision tokens — image URI was not recognized"

    #         image_inputs, video_inputs = process_vision_info(messages)

    #         assert image_inputs and len(image_inputs) > 0, \
    #             f"process_vision_info returned empty — URI format issue: {Path(tmp.name).as_uri()}"

    #         inputs = self.processor(
    #             text=[text],
    #             images=image_inputs,
    #             videos=video_inputs if video_inputs else None,
    #             return_tensors="pt"
    #         ).to(self.model.device)  # FIX: model.device not hardcoded cuda:0

    #         # Verify token count alignment before hitting CUDA
    #         thw = inputs.get('image_grid_thw')
    #         if thw is not None:
    #             expected = int(thw[0][0] * thw[0][1] * thw[0][2])
    #             actual = inputs['pixel_values'].shape[0]
    #             assert expected == actual, \
    #                 f"Token mismatch: image_grid_thw says {expected}, pixel_values has {actual} — dimensions not 28-aligned"

    #         with torch.no_grad():
    #             outputs = self.model.generate(
    #                 **inputs,
    #                 max_new_tokens=512,
    #                 do_sample=False
    #             )

    #         response = self.processor.decode(
    #             outputs[0][inputs.input_ids.shape[1]:],
    #             skip_special_tokens=True
    #         )

    #     finally:
    #         os.unlink(tmp.name)

    #     return self.parse_response(response)


    # def predict(self, question, images, camera_names, enable_tools=False):
    #     import torch
    #     import tempfile
    #     import os
    #     from qwen_vl_utils import process_vision_info
    #     from transformers import GenerationConfig

    #     if self.model is None:
    #         self.load_model()

    #     if not images:
    #         return ModelPrediction(answer="No images available", raw_output="")

    #     # Build 28-aligned combined image
    #     CELL_H, CELL_W = 224, 224
    #     resized = [img.convert('RGB').resize((CELL_W, CELL_H)) for img in images]
    #     combined = Image.new('RGB', (CELL_W * len(resized), CELL_H))
    #     x = 0
    #     for img in resized:
    #         combined.paste(img, (x, 0))
    #         x += CELL_W
    #     if combined.width > 896:
    #         combined = combined.resize((896, 224))

    #     tmp = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
    #     combined.save(tmp.name, quality=95)
    #     tmp.close()

    #     try:
    #         cam_labels = ", ".join(camera_names)
    #         messages = [{
    #             "role": "user",
    #             "content": [
    #                 {
    #                     "type": "image",
    #                     "image": Path(tmp.name).as_uri(),
    #                 },
    #                 {
    #                     "type": "text",
    #                     "text": f"Cameras shown left to right: {cam_labels}\n{self.build_prompt(question, camera_names, enable_tools)}"
    #                 }
    #             ]
    #         }]

    #         text = self.processor.apply_chat_template(
    #             messages, tokenize=False, add_generation_prompt=True
    #         )

    #         image_inputs, video_inputs = process_vision_info(messages)

    #         inputs = self.processor(
    #             text=[text],
    #             images=image_inputs,
    #             videos=video_inputs if video_inputs else None,
    #             return_tensors="pt"
    #         )

    #         # FIX: device_map="auto" spreads layers across GPUs — self.model.device
    #         # is unreliable. Explicitly move each tensor to cuda:0 (where visual
    #         # encoder lives) instead of using .to(self.model.device)
    #         visual_device = next(self.model.visual.parameters()).device
    #         lm_device     = next(self.model.model.parameters()).device

            
    #         # device = "cuda:0"
    #         input_ids      = inputs["input_ids"].to(device)
    #         attention_mask = inputs["attention_mask"].to(device)
    #         pixel_values   = inputs["pixel_values"].to(device, dtype=torch.float16)
    #         image_grid_thw = inputs["image_grid_thw"].to(device)

    #         print(f"  visual_device: {visual_device}, lm_device: {lm_device}")
    #         print(f"  pixel_values:   {pixel_values.shape}")
    #         print(f"  image_grid_thw: {image_grid_thw}")

    #         with torch.no_grad():
    #             outputs = self.model.generate(
    #                 input_ids=input_ids,
    #                 attention_mask=attention_mask,
    #                 pixel_values=pixel_values,
    #                 image_grid_thw=image_grid_thw,
    #                 max_new_tokens=512,
    #                 # FIX: override model's generation_config which sets temperature/
    #                 # top_p/top_k forcing _sample path — pass neutral values so
    #                 # do_sample=False actually takes effect
    #                 generation_config=GenerationConfig(
    #                     do_sample=False,
    #                     temperature=1.0,
    #                     # top_p=1.0,
    #                     # top_k=50,
    #                 )
    #             )

    #         response = self.processor.decode(
    #             outputs[0][input_ids.shape[1]:],
    #             skip_special_tokens=True
    #         )

    #     finally:
    #         os.unlink(tmp.name)

    #     return self.parse_response(response)

    def raw_query(self, prompt, images):
        """Unwrapped prompt->text call, no build_prompt()/parse_response()
        templating — used by tool_executor.py's internal tool-grounding
        queries and by the ReAct tool loop, so short structured prompts
        (e.g. "category: yes/no, count: N") aren't buried inside predict()'s
        SAFE/UNSAFE report boilerplate."""
        import torch
        import tempfile
        import os
        from pathlib import Path
        from qwen_vl_utils import process_vision_info
        from transformers import GenerationConfig

        if self.model is None:
            self.load_model()

        if not images:
            return ""

        # EXPERIMENT (Claude, 2026-09-24, reviewer comment 5): QWEN_NATIVE_RES=1
        # skips the 224x224-per-camera shrink and tiles the cameras at native
        # size (UAV3D frames are 800x450). The processor's own
        # max_pixels=640*28*28 cap still applies, so a single camera reaches
        # the model unshrunk and a multi-camera strip is downscaled only to
        # that cap. Off by default: predict() also routes through here, so
        # enabling it globally would change the published Protocol A/B numbers.
        if os.environ.get('QWEN_NATIVE_RES') == '1':
            rgb = [img.convert('RGB') for img in images]
            combined = Image.new('RGB', (sum(i.width for i in rgb), max(i.height for i in rgb)))
            x = 0
            for img in rgb:
                combined.paste(img, (x, 0))
                x += img.width
        else:
            CELL_H, CELL_W = 224, 224
            resized = [img.convert('RGB').resize((CELL_W, CELL_H)) for img in images]
            combined = Image.new('RGB', (CELL_W * len(resized), CELL_H))
            x = 0
            for img in resized:
                combined.paste(img, (x, 0))
                x += CELL_W
            if combined.width > 896:
                combined = combined.resize((896, 224))

        tmp = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
        combined.save(tmp.name, quality=95)
        tmp.close()

        try:
            messages = [{
                "role": "user",
                "content": [
                    {"type": "image", "image": Path(tmp.name).as_uri()},
                    {"type": "text", "text": prompt}
                ]
            }]

            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            image_inputs, video_inputs = process_vision_info(messages)

            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs if video_inputs else None,
                return_tensors="pt"
            )

            # transformers has moved the vision tower between a top-level
            # `.visual` and a nested `.model.visual` across versions; this
            # instance (currently installed transformers) has it nested.
            # Handle both rather than hardcoding the layout of one version.
            visual_module = getattr(self.model, "visual", None)
            if visual_module is None:
                visual_module = getattr(getattr(self.model, "model", None), "visual", None)
            if visual_module is None:
                raise AttributeError("Could not locate the Qwen2VL vision encoder ('visual') on this model instance")
            visual_device = next(visual_module.parameters()).device
            lm_device     = visual_device

            input_ids      = inputs["input_ids"].to(lm_device)
            attention_mask = inputs["attention_mask"].to(lm_device)
            pixel_values   = inputs["pixel_values"].to(visual_device, dtype=torch.float16)
            image_grid_thw = inputs["image_grid_thw"].to(visual_device)

            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    pixel_values=pixel_values,
                    image_grid_thw=image_grid_thw,
                    max_new_tokens=512,
                    generation_config=GenerationConfig(
                        do_sample=False,
                        max_new_tokens=512,
                    )
                )

            response = self.processor.decode(
                outputs[0][input_ids.shape[1]:],
                skip_special_tokens=True
            )

        finally:
            os.unlink(tmp.name)

        return response

    def predict(self, question, images, camera_names, enable_tools=False):
        if not images:
            return ModelPrediction(answer="No images available", raw_output="")
        cam_labels = ", ".join(camera_names)
        prompt = f"Cameras shown left to right: {cam_labels}\n{self.build_prompt(question, camera_names, enable_tools)}"
        response = self.raw_query(prompt, images)
        return self.parse_response(response)

    def predict_with_tools(self, question, available_cameras, image_paths, tool_executor, max_iterations=10):
        from .react_tool_loop import run_react_tool_loop
        return run_react_tool_loop(
            self, question, available_cameras, image_paths, tool_executor, max_iterations
        )

# class Phi4Wrapper(BaseVLM):

#     def __init__(self, device="cuda", quantize=True):
#         super().__init__("microsoft/Phi-4-multimodal-instruct", device)
#         self.quantize = quantize

#     def load_model(self):
#         from transformers import AutoModelForCausalLM, AutoProcessor, BitsAndBytesConfig
#         import torch

#         print(f"Loading {self.model_name}...")
#         bnb_config = BitsAndBytesConfig(
#             load_in_4bit=True,
#             bnb_4bit_compute_dtype=torch.float16
#         )

#         self.model = Qwen2VLForConditionalGeneration.from_pretrained(
#             self.model_name,
#             quantization_config=bnb_config,
#             # FIX: force single GPU — device_map="auto" was spreading visual encoder
#             # (cuda:1) and LM (cuda:4) across GPUs causing Accelerate hook interference
#             # in cu_seqlens/rotary_pos_emb internal tensor operations
#             device_map="cuda:0",
#             torch_dtype=torch.float16,
#         )

#         self.processor = AutoProcessor.from_pretrained(
#             self.model_name, 
#             min_pixels=64 * 28 * 28,
#             max_pixels=640 * 28 * 28
#         )
#         print(f"✓ Loaded {self.model_name}")
#         load_kwargs = {"trust_remote_code": True, "torch_dtype": torch.float16, "device_map": "auto"}
#         if self.quantize:
#             load_kwargs["quantization_config"] = BitsAndBytesConfig(
#                 load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16
#             )

#         self.model = AutoModelForCausalLM.from_pretrained(self.model_name, **load_kwargs)
#         self.model.eval()
#         print(f"✓ Loaded {self.model_name}")

#     def predict(self, question, images, camera_names, enable_tools=False):
#         import torch

#         if self.model is None:
#             self.load_model()

#         if not images:
#             return ModelPrediction(answer="No images available", raw_output="")

#         content = []
#         for img, cam in zip(images, camera_names):
#             content.append({"type": "image", "image": img})
#             content.append({"type": "text", "text": f"[{cam}]"})

#         prompt = self.build_prompt(question, camera_names, enable_tools)
#         content.append({"type": "text", "text": prompt})

#         messages = [{"role": "user", "content": content}]

#         text = self.processor.tokenizer.apply_chat_template(
#             messages, tokenize=False, add_generation_prompt=True
#         )

#         inputs = self.processor(
#             text=text,
#             images=images,
#             return_tensors="pt"
#         ).to(self.model.device)

#         with torch.no_grad():
#             output_ids = self.model.generate(
#                 **inputs,
#                 max_new_tokens=512,
#                 do_sample=False,
#                 temperature=None,   # explicitly unset — suppresses _sample routing
#                 top_p=None,
#                 top_k=None,
#             )

#         generated = output_ids[0][inputs["input_ids"].shape[1]:]
#         response = self.processor.tokenizer.decode(generated, skip_special_tokens=True)
#         return self.parse_response(response)

class Phi4Wrapper(BaseVLM):
    HF_MODEL_ID = "microsoft/Phi-4-multimodal-instruct"

    def __init__(self, device="cuda:0", quantize=False, local_model_path=None):
        super().__init__(self.HF_MODEL_ID, device)
        self.quantize = quantize
        # Use local path only if explicitly provided and it exists on disk;
        # fall back to the HuggingFace hub ID so the wrapper works on any machine.
        if local_model_path and Path(local_model_path).exists():
            self.model_path = local_model_path
        else:
            self.model_path = self.HF_MODEL_ID

    def load_model(self):
        from transformers import AutoModelForCausalLM, AutoProcessor, BitsAndBytesConfig, GenerationConfig
        import torch
        
    
        
        print(f"Loading Phi-4-multimodal...")

        load_kwargs = {
            "trust_remote_code": True,
            "torch_dtype": torch.float16,
            "_attn_implementation": "eager",
        }

        if self.quantize:
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16
            )
            load_kwargs["device_map"] = "cuda:0"
        else:
            # no device_map — .to() directly avoids accelerate hook issues
            pass

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path, **load_kwargs
        )

        if not self.quantize:
            self.model = self.model.to("cuda:0")

        self.model.generation_config = GenerationConfig(do_sample=False, num_beams=1)
        self.model.eval()

        self.processor = AutoProcessor.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            num_crops=4  # Phi-4 specific — controls image tile count per image
        )

        print(f"✓ Loaded Phi-4-multimodal")

    # def predict(self, question, images, camera_names, enable_tools=False):
    #     import torch
    #     from transformers import GenerationConfig

    #     if self.model is None:
    #         self.load_model()
    #     if not images:
    #         return ModelPrediction(answer="No images available", raw_output="")

    #     # Phi-4 multimodal uses <|image_N|> placeholder tokens
    #     content = ""
    #     for i, (img, cam) in enumerate(zip(images, camera_names), 1):
    #         content += f"<|image_{i}|>\n[{cam}]\n"

    #     prompt = self.build_prompt(question, camera_names, enable_tools)
    #     content += prompt

    #     messages = [
    #         {"role": "user", "content": content}
    #     ]

    #     text = self.processor.tokenizer.apply_chat_template(
    #         messages,
    #         tokenize=False,
    #         add_generation_prompt=True
    #     )

    #     inputs = self.processor(
    #         text=text,
    #         images=images,
    #         return_tensors="pt"
    #     )
    #     inputs = {k: v.to("cuda:0") for k, v in inputs.items() if v is not None}

    #     with torch.no_grad():
    #         output_ids = self.model.generate(
    #             **inputs,
    #         generation_config=GenerationConfig(
    #         do_sample=False,
    #         num_beams=1,
    #         max_new_tokens=512,
    #         )
    #         )

    #     generated = output_ids[0][inputs["input_ids"].shape[1]:]
    #     response = self.processor.tokenizer.decode(generated, skip_special_tokens=True)
    #     return self.parse_response(response)

    def predict(self, question, images, camera_names, enable_tools=False):
        import torch

        if self.model is None:
            self.load_model()
        if not images:
            return ModelPrediction(answer="No images available", raw_output="")

        content = ""
        for i, (img, cam) in enumerate(zip(images, camera_names), 1):
            content += f"<|image_{i}|>\n[{cam}]\n"

        prompt = self.build_prompt(question, camera_names, enable_tools)
        content += prompt

        messages = [{"role": "user", "content": content}]

        text = self.processor.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = self.processor(text=text, images=images, return_tensors="pt")
        
        inputs = {
            k: (v if k == "input_mode" else v.to("cuda:0"))
            for k, v in inputs.items() if v is not None}
        # inputs["input_mode"] = torch.tensor([1], dtype=torch.long)
        
        with torch.no_grad():
            generated = inputs["input_ids"].clone()
            past_key_values = None

            for _ in range(512):
                model_inputs = {
                    "input_ids": generated if past_key_values is None else generated[:, -1:],
                    "attention_mask": torch.ones_like(generated),
                    "past_key_values": past_key_values,
                    "use_cache": True,
                    "input_mode": 1,
                }
                if past_key_values is None:
                    if "pixel_values" in inputs:
                        model_inputs["pixel_values"] = inputs["pixel_values"]
                    if "image_sizes" in inputs:
                        model_inputs["image_sizes"] = inputs["image_sizes"]

                outputs = self.model(**model_inputs)
                past_key_values = outputs.past_key_values
                next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                generated = torch.cat([generated, next_token], dim=-1)

                if next_token.item() == self.processor.tokenizer.eos_token_id:
                    break

        generated_tokens = generated[0][inputs["input_ids"].shape[1]:]
        response = self.processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        return self.parse_response(response)

class GPT4OWrapper(BaseVLM):

    def __init__(self, model_name="gpt-4o", api_key=None):
        super().__init__(model_name, device="api")
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.client = None  # FIX: explicit None init — consistent with other wrappers

    def load_model(self):
        from openai import OpenAI
        self.client = OpenAI(api_key=self.api_key)
        print(f"✓ Initialized OpenAI client for {self.model_name}")

    def _image_to_base64(self, image):
        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")

    def predict(self, question, images, camera_names, enable_tools=False):
        if self.client is None:  # FIX: consistent None check instead of hasattr
            self.load_model()

        content = []
        for img, cam in zip(images, camera_names):
            b64 = self._image_to_base64(img)
            content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
            content.append({"type": "text", "text": f"[{cam}]"})

        prompt = self.build_prompt(question, camera_names, enable_tools)
        content.append({"type": "text", "text": prompt})

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": content}],
            max_tokens=512,
            temperature=0.0
        )
        return self.parse_response(response.choices[0].message.content)


class LLaVAWrapper(BaseVLM):

    def __init__(self, model_name="llava-hf/llava-v1.6-mistral-7b-hf", device="cuda", quantize=True):
        super().__init__(model_name, device)
        self.quantize = quantize

    def load_model(self):
        from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration, BitsAndBytesConfig
        import torch

        print(f"Loading {self.model_name}...")
        self.processor = LlavaNextProcessor.from_pretrained(self.model_name)

        load_kwargs = {"device_map": "auto"}
        if self.quantize:
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16
            )
        else:
            load_kwargs["torch_dtype"] = torch.float16

        self.model = LlavaNextForConditionalGeneration.from_pretrained(self.model_name, **load_kwargs)
        print(f"✓ Loaded {self.model_name}")

    def predict(self, question, images, camera_names, enable_tools=False):
        if self.model is None:
            self.load_model()

        if not images:
            return ModelPrediction(answer="No images available", raw_output="")

        if len(images) > 1:
            widths, heights = zip(*(img.size for img in images))
            combined = Image.new('RGB', (sum(widths), max(heights)))
            x = 0
            for img in images:
                combined.paste(img, (x, 0))
                x += img.width
            images = [combined]

        prompt = self.build_prompt(question, camera_names, enable_tools)
        conversation = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
        text = self.processor.apply_chat_template(conversation, add_generation_prompt=True)

        inputs = self.processor(text=text, images=images[0], return_tensors="pt").to(self.model.device)

        import torch
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=512, do_sample=False)

        response = self.processor.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        return self.parse_response(response)


class LLaVAOVWrapper(BaseVLM):
    """Wrapper for LLaVA-OneVision models (lmms-lab/llava-onevision-qwen2-*).

    LLaVA-OV is architecturally distinct from LLaVA-Next: it uses Qwen2 as the
    language backbone and natively supports multi-image input without grid-stitching.
    """

    def __init__(self, model_name="lmms-lab/llava-onevision-qwen2-7b-ov", device="cuda", quantize=True):
        super().__init__(model_name, device)
        self.quantize = quantize

    def load_model(self):
        from transformers import LlavaOnevisionForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
        import torch

        print(f"Loading {self.model_name}...")
        self.processor = AutoProcessor.from_pretrained(self.model_name)

        load_kwargs = {"device_map": "cuda:0"}
        if self.quantize:
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16
            )
        else:
            load_kwargs["torch_dtype"] = torch.float16

        self.model = LlavaOnevisionForConditionalGeneration.from_pretrained(
            self.model_name, **load_kwargs
        )
        print(f"✓ Loaded {self.model_name}")

    def raw_query(self, prompt, images):
        """Unwrapped prompt->text call, no build_prompt()/parse_response()
        templating — used by tool_executor.py's internal tool-grounding
        queries and by the ReAct tool loop."""
        import torch

        if self.model is None:
            self.load_model()

        if not images:
            return ""

        # LLaVA-OV uses one {"type": "image"} content block per image, followed by text.
        # The processor matches these blocks positionally to the images list.
        content = [{"type": "image"} for _ in images]
        content.append({"type": "text", "text": prompt})

        conversation = [{"role": "user", "content": content}]
        text = self.processor.apply_chat_template(conversation, add_generation_prompt=True)

        inputs = self.processor(
            text=[text],
            images=images,
            return_tensors="pt"
        ).to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=512, do_sample=False)

        return self.processor.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        )

    def predict(self, question, images, camera_names, enable_tools=False):
        if not images:
            return ModelPrediction(answer="No images available", raw_output="")
        cam_labels = " | ".join(camera_names)
        prompt = f"Cameras (left-to-right): {cam_labels}\n{self.build_prompt(question, camera_names, enable_tools)}"
        response = self.raw_query(prompt, images)
        return self.parse_response(response)

    def predict_with_tools(self, question, available_cameras, image_paths, tool_executor, max_iterations=10):
        from .react_tool_loop import run_react_tool_loop
        return run_react_tool_loop(
            self, question, available_cameras, image_paths, tool_executor, max_iterations
        )


class InternVLWrapper(BaseVLM):

    def __init__(self, model_name="OpenGVLab/InternVL2-2B", device="cuda", quantize=True):
        super().__init__(model_name, device)
        self.quantize = quantize

    def load_model(self):
        from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig
        import torch

        print(f"Loading {self.model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)

        # No device_map in the non-quantized path: transformers' newer
        # device_map-loading code (caching_allocator_warmup) calls
        # model.all_tied_weights_keys, which this trust_remote_code
        # InternVLChatModel class doesn't define -- crashes at load time
        # regardless of quantization. Mirrors Phi4Wrapper's device_map-free
        # + explicit .to() pattern, which sidesteps that code path.
        load_kwargs = {"trust_remote_code": True}
        if self.quantize:
            load_kwargs["device_map"] = "cuda:0"
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16
            )
        else:
            load_kwargs["torch_dtype"] = torch.float16

        self.model = AutoModel.from_pretrained(self.model_name, **load_kwargs)
        if not self.quantize:
            self.model = self.model.to("cuda:0")
        print(f"✓ Loaded {self.model_name}")

    def predict(self, question, images, camera_names, enable_tools=False):
        import torch
        from torchvision import transforms

        if self.model is None:
            self.load_model()

        if not images:
            return ModelPrediction(answer="No images available", raw_output="")

        transform = transforms.Compose([
            transforms.Resize((448, 448)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        pixel_values = torch.stack([transform(img) for img in images]).to(self.model.device)
        prompt = self.build_prompt(question, camera_names, enable_tools)

        response = self.model.chat(
            self.tokenizer,
            pixel_values=pixel_values,
            question=prompt,
            generation_config={"max_new_tokens": 512, "do_sample": False}
        )
        return self.parse_response(response)


# Model registry
# FIX: added Claude and Gemini API models so get_model() works for all supported models
MODEL_REGISTRY = {
    "qwen2-vl-2b":          lambda: Qwen2VLWrapper("Qwen/Qwen2-VL-2B-Instruct"),
    "qwen2-vl-7b":          lambda: Qwen2VLWrapper("Qwen/Qwen2-VL-7B-Instruct"),
    "phi-4":                lambda: Phi4Wrapper(),
    # quantize=False: transformers' 4-bit-quantizer path calls
    # model.all_tied_weights_keys, which InternVL's trust_remote_code model
    # class doesn't define (only _tied_weights_keys) -- a library/remote-code
    # version mismatch, not a logic bug here. fp16 sidesteps the quantizer
    # entirely. InternVL2-8B is left quantized (still broken) since fp16
    # (~16GB) doesn't fit in this instance's free VRAM.
    "internvl2-2b":         lambda: InternVLWrapper("OpenGVLab/InternVL2-2B", quantize=False),
    "internvl2-8b":         lambda: InternVLWrapper("OpenGVLab/InternVL2-8B"),
    "llava-next-7b":        lambda: LLaVAWrapper("llava-hf/llava-v1.6-mistral-7b-hf"),
    # llava-hf's -hf conversion, not the original lmms-lab checkpoint: the
    # lmms-lab repo's config declares model_type "llava" while this code
    # instantiates LlavaOnevisionForConditionalGeneration (expects
    # "llava_onevision"), so almost every weight (vision tower, projector,
    # embed_tokens) came back MISSING/randomly-initialized instead of loaded
    # -- would have scored a near-random model as "LLaVA-OV-7B" even if the
    # resulting OOM hadn't crashed it first. The -hf repo is the actual
    # transformers-native conversion for this class.
    "llava-ov-7b":          lambda: LLaVAOVWrapper("llava-hf/llava-onevision-qwen2-7b-ov-hf"),
    "gpt-4o":               lambda: GPT4OWrapper("gpt-4o"),
    "gpt-4o-mini":          lambda: GPT4OWrapper("gpt-4o-mini"),
    # FIX: Claude models were missing — UAVAgent.load_model and step3 both call
    # get_model() which would raise ValueError for any claude model name
    "claude-3-5-sonnet":    lambda: _get_claude_wrapper("claude-3-5-sonnet-20241022"),
    "claude-3-haiku":       lambda: _get_claude_wrapper("claude-3-haiku-20240307"),
}


def _get_claude_wrapper(model_name: str) -> BaseVLM:
    """Lazy import Claude wrapper to avoid hard dependency on anthropic package"""
    from .api_models import ClaudeWrapper
    return ClaudeWrapper(model_name)


def get_model(model_name: str) -> BaseVLM:
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[model_name]()


def list_models():
    return list(MODEL_REGISTRY.keys())