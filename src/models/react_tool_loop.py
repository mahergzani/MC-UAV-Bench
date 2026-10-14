"""
Shared ReAct-style prompted tool loop for open-weight VLMs without native
function calling (Qwen2-VL-2B/7B, LLaVA-OV-7B).

Design note: this mirrors the free-tool-choice design ClaudeWrapper/
GPT4VisionWrapper's predict_with_tools() already use in api_models.py --
same 7-tool registry (src/tools/tool_registry.py), same budget, same
"model decides when it's done" termination. It does NOT replay UAVAgent's
own scripted 4-phase pipeline (get_camera_view -> detect_objects ->
estimate_depth -> fuse_multi_camera -> assess_safety, then a hardcoded
Python final-answer template) -- UAVAgent's phases are a deterministic
state machine, not genuine agentic tool selection by its base model, so
replaying them wouldn't exercise tool-use capability for any model,
UAVAgent included. This loop gives every open-weight baseline the same
real free tool choice Claude/GPT-4o already get, expressed as one strict
JSON action per turn since these models have no native tool-calling API.
"""

import json
from typing import Dict, List

from ..tools.tool_registry import get_all_tools
from .base_model import ModelPrediction

MAX_PARSE_FAILURES = 2
# Matches the sentinel ClaudeWrapper/GPT4VisionWrapper's predict_with_tools
# already return on budget exhaustion -- same policy, not a new one.
BUDGET_EXHAUSTED_ANSWER = "Max iterations reached"

def _tool_description_block() -> str:
    lines = []
    for tool in get_all_tools():
        params = []
        for p in tool.parameters:
            req = "required" if p.required else f"optional, default={p.default!r}"
            enum = f", one of {p.enum}" if p.enum else ""
            params.append(f"{p.name} ({p.type}{enum}, {req}): {p.description}")
        param_block = "; ".join(params) if params else "no parameters"
        lines.append(f'- "{tool.name}": {tool.description} Parameters: {param_block}')
    return "\n".join(lines)


def _build_system_prompt(question: str, camera_names: List[str], max_iterations: int) -> str:
    cam_list = ", ".join(camera_names)
    tools_block = _tool_description_block()
    return f"""You are an AI assistant for UAV (drone) autonomous systems.
You have access to {len(camera_names)} camera views: {cam_list} (shown above, in that order).

Question: {question}

You can use these tools to investigate further before answering:
{tools_block}
- "submit_answer": End the episode with your final answer. Parameters: answer (string, required): your complete answer, including a clear SAFE or UNSAFE decision with justification and specific measurements (distances in meters, areas in sqm) when relevant.

You have at most {max_iterations} turns total (tool calls + final answer combined).

On EVERY turn, respond with EXACTLY ONE JSON object and nothing else -- no
markdown fences, no extra text before or after it. Format:
{{"tool": "<tool_name>", "args": {{...}}}}

Example tool call: {{"tool": "detect_objects", "args": {{"camera": "CAM_FRONT", "object_types": ["vehicle", "person"], "confidence_threshold": 0.7}}}}
Example final answer: {{"tool": "submit_answer", "args": {{"answer": "SAFE - ..."}}}}

Respond with your first action now."""


def _parse_action(raw_text: str):
    """Extract and parse exactly one JSON action object from raw model
    output. Returns (action_dict, None) on success, (None, error) on
    failure. Strict: no guessing at intent from malformed output.

    Uses json.JSONDecoder.raw_decode() from the first '{', not a greedy
    first-'{'-to-last-'}' regex -- models occasionally emit one extra
    trailing brace or trailing text after an otherwise-valid object (e.g.
    '{"tool": "x", "args": {...}}}'), which a greedy regex would include in
    the matched span and fail to parse even though a real, valid JSON
    object is right there at the start.
    """
    if not raw_text or not raw_text.strip():
        return None, "empty response"
    start = raw_text.find('{')
    if start == -1:
        return None, "no JSON object found in response"
    try:
        action, _ = json.JSONDecoder().raw_decode(raw_text, start)
    except json.JSONDecodeError as e:
        return None, f"JSON parse error: {e}"
    if not isinstance(action, dict) or "tool" not in action:
        return None, "parsed JSON missing required 'tool' key"
    if not isinstance(action.get("args"), dict):
        action["args"] = {}
    return action, None


def run_react_tool_loop(
    model,
    question: str,
    available_cameras: List[str],
    image_paths: Dict[str, str],
    tool_executor,
    max_iterations: int = 10,
) -> ModelPrediction:
    """Genuine free-tool-choice ReAct loop for a model that only exposes
    raw_query(prompt, images) -> str (no native function calling).

    All available cameras' images are loaded once up front and re-sent
    every turn -- these wrappers have no persistent chat/session state
    between calls, so each turn is architecturally a fresh forward pass
    over the full simulated transcript. This matches what the
    native-tool-calling models (Claude/GPT-4o) see: every camera image
    visible from turn one, tools available as optional follow-up queries.
    """
    from PIL import Image

    images = []
    camera_names = []
    for cam in available_cameras:
        path = image_paths.get(cam)
        if not path:
            continue
        images.append(Image.open(path).convert('RGB'))
        camera_names.append(cam)

    if not images:
        return ModelPrediction(
            answer="No images available",
            termination_reason="error",
        )

    transcript = _build_system_prompt(question, camera_names, max_iterations)

    tool_trace = []
    cameras_used = set()
    consecutive_parse_failures = 0
    total_parse_failures = 0
    valid_tools = {t.name for t in get_all_tools()}

    def _fail(raw_text: str, note: str):
        """Record one parse-class failure; returns a terminal ModelPrediction
        if the consecutive-failure budget is exhausted, else None (caller
        continues the loop after this appends `note` to the transcript)."""
        nonlocal consecutive_parse_failures, total_parse_failures
        total_parse_failures += 1
        consecutive_parse_failures += 1
        print(f"  [react_tool_loop] parse failure {consecutive_parse_failures}/{MAX_PARSE_FAILURES}: {note} "
              f"-- raw output: {raw_text[:300]!r}")
        if consecutive_parse_failures >= MAX_PARSE_FAILURES:
            return ModelPrediction(
                answer=f"ERROR: {consecutive_parse_failures} consecutive tool-call parse failures",
                tool_calls=tool_trace,
                cameras_used=list(cameras_used),
                termination_reason="error",
                parse_failures=total_parse_failures,
            )
        return None

    for iteration in range(max_iterations):
        raw_text = model.raw_query(transcript, images)
        action, err = _parse_action(raw_text)

        if err is not None:
            terminal = _fail(raw_text, err)
            if terminal is not None:
                return terminal
            transcript += (
                f"\n\nAssistant: {raw_text}\n\n"
                f"Your last response could not be parsed ({err}). "
                f"Respond with EXACTLY ONE JSON object, nothing else."
            )
            continue

        tool_name = action["tool"]
        args = action["args"]

        if tool_name == "submit_answer":
            answer = args.get("answer", "")
            if not answer:
                terminal = _fail(raw_text, "submit_answer with empty 'answer'")
                if terminal is not None:
                    return terminal
                transcript += (
                    f"\n\nAssistant: {raw_text}\n\n"
                    f"submit_answer requires a non-empty 'answer' argument. Try again."
                )
                continue
            consecutive_parse_failures = 0
            return ModelPrediction(
                answer=answer,
                tool_calls=tool_trace,
                cameras_used=list(cameras_used),
                termination_reason="answered",
                parse_failures=total_parse_failures,
            )

        if tool_name not in valid_tools:
            terminal = _fail(raw_text, f"unknown tool {tool_name!r}")
            if terminal is not None:
                return terminal
            transcript += (
                f"\n\nAssistant: {raw_text}\n\n"
                f"Unknown tool {tool_name!r}. Valid tools: {sorted(valid_tools)} or 'submit_answer'."
            )
            continue

        consecutive_parse_failures = 0

        if 'camera' in args:
            cameras_used.add(args['camera'])
        if isinstance(args.get('cameras'), list):
            cameras_used.update(args['cameras'])

        result = tool_executor.execute(tool_name, args)
        tool_trace.append({
            'step': iteration + 1,
            'tool': tool_name,
            'params': args,
            'result': result.data if result.success else {'error': 'failed'}
        })

        observation = result.data if result.success else {'error': 'tool execution failed'}
        transcript += (
            f"\n\nAssistant: {raw_text}\n\n"
            f"Observation ({tool_name}): {json.dumps(observation)}\n\n"
            f"What is your next action?"
        )

    return ModelPrediction(
        answer=BUDGET_EXHAUSTED_ANSWER,
        tool_calls=tool_trace,
        cameras_used=list(cameras_used),
        termination_reason="budget_exhausted",
        parse_failures=total_parse_failures,
    )
