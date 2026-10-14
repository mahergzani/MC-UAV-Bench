"""
MC-UAV-Bench Tool Registry

Defines the 7 perception and reasoning tools for UAV autonomous systems.
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field


@dataclass
class ToolParameter:
    """Tool parameter definition"""
    name: str
    type: str
    description: str
    required: bool = True
    default: Any = None
    enum: Optional[List[str]] = None
    items: Optional[Dict] = None       # FIX 1: needed for array types in OpenAI schema
    properties: Optional[Dict] = None  # FIX 1: needed for object types in OpenAI schema


@dataclass
class Tool:
    """Tool definition"""
    name: str
    description: str
    category: str  # perception, fusion, motion, decision
    parameters: List[ToolParameter] = field(default_factory=list)
    
    def to_openai_schema(self) -> Dict:
        """Convert to OpenAI function calling format"""
        properties = {}
        required = []
        
        for param in self.parameters:
            prop = {"type": param.type, "description": param.description}

            if param.enum:
                prop["enum"] = param.enum
            if param.default is not None:
                prop["default"] = param.default

            # FIX 1: array params require "items", object params require "properties"
            # Without these, OpenAI function calling rejects or misparsed the schema
            if param.type == "array":
                prop["items"] = param.items if param.items else {"type": "string"}
            if param.type == "object":
                prop["properties"] = param.properties if param.properties else {}

            properties[param.name] = prop
            if param.required:
                required.append(param.name)
        
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required
                }
            }
        }


# Define all 7 tools
TOOL_REGISTRY = {
    "get_camera_view": Tool(
        name="get_camera_view",
        description="Retrieve image from specified camera. Use to examine specific viewpoints.",
        category="perception",
        parameters=[
            ToolParameter(
                name="camera",
                type="string",
                description="Camera to retrieve",
                enum=["CAM_FRONT", "CAM_BACK", "CAM_LEFT", "CAM_RIGHT", "CAM_BOTTOM"]
            ),
            ToolParameter(
                name="temporal",
                type="boolean",
                description="Include temporal history for motion analysis",
                required=False,
                default=False
            )
        ]
    ),
    
    "detect_objects": Tool(
        name="detect_objects",
        description="Detect objects in camera view. Returns bounding boxes, types, and distances.",
        category="perception",
        parameters=[
            ToolParameter(
                name="camera",
                type="string",
                description="Camera to analyze",
                enum=["CAM_FRONT", "CAM_BACK", "CAM_LEFT", "CAM_RIGHT", "CAM_BOTTOM"]
            ),
            ToolParameter(
                name="object_types",
                type="array",
                description="Types to detect: vehicle, person, obstacle, structure, vegetation",
                required=False,
                default=["vehicle", "person", "obstacle"],
                items={"type": "string", "enum": ["vehicle", "person", "obstacle", "structure", "vegetation"]}  # FIX 1
            ),
            ToolParameter(
                name="confidence_threshold",
                type="number",
                description="Minimum detection confidence (0-1)",
                required=False,
                default=0.7
            )
        ]
    ),
    
    "estimate_depth": Tool(
        name="estimate_depth",
        description="Estimate depth/distance in camera view. Returns depth map statistics.",
        category="perception",
        parameters=[
            ToolParameter(
                name="camera",
                type="string",
                description="Camera to analyze",
                enum=["CAM_FRONT", "CAM_BACK", "CAM_LEFT", "CAM_RIGHT", "CAM_BOTTOM"]
            ),
            ToolParameter(
                name="target_region",
                type="string",
                description="Region of interest: center, full, or bbox coordinates",
                required=False,
                default="full"
            )
        ]
    ),
    
    "fuse_multi_camera": Tool(
        name="fuse_multi_camera",
        description="Fuse information from multiple cameras into unified representation.",
        category="fusion",
        parameters=[
            ToolParameter(
                name="cameras",
                type="array",
                description="List of cameras to fuse",
                required=False,                                            # FIX 2: was required=True, causes schema validation error if omitted
                default=["CAM_FRONT", "CAM_BOTTOM"],                       # FIX 2: sensible default
                items={"type": "string", "enum": ["CAM_FRONT", "CAM_BACK", "CAM_LEFT", "CAM_RIGHT", "CAM_BOTTOM"]}  # FIX 1
            ),
            ToolParameter(
                name="fusion_type",
                type="string",
                description="Type of fusion",
                enum=["3d_position", "occupancy_grid", "bird_eye_view"],
                required=False,
                default="3d_position"
            )
        ]
    ),
    
    "track_motion": Tool(
        name="track_motion",
        description="Track object motion across frames. Returns velocity and trajectory.",
        category="motion",
        parameters=[
            ToolParameter(
                name="camera",
                type="string",
                description="Camera for tracking",
                enum=["CAM_FRONT", "CAM_BACK", "CAM_LEFT", "CAM_RIGHT", "CAM_BOTTOM"]
            ),
            ToolParameter(
                name="object_id",
                type="string",
                description="ID of object to track (from detect_objects)"
            ),
            ToolParameter(
                name="duration",
                type="number",
                description="Tracking duration in seconds",
                required=False,
                default=2.0
            )
        ]
    ),
    
    "assess_safety": Tool(
        name="assess_safety",
        description="Assess safety for specific task. Returns safety verdict and reasoning.",
        category="decision",
        parameters=[
            ToolParameter(
                name="task",
                type="string",
                description="Task to assess",
                enum=["landing", "clearance", "navigation", "emergency"]
            ),
            ToolParameter(
                name="cameras",
                type="array",
                description="Cameras that have been analyzed",
                required=False,                  # FIX 2: was implicitly required, agent may not always pass this
                default=[],
                items={"type": "string", "enum": ["CAM_FRONT", "CAM_BACK", "CAM_LEFT", "CAM_RIGHT", "CAM_BOTTOM"]}  # FIX 1
            ),
            ToolParameter(
                name="criteria",
                type="object",
                description="Safety criteria (min_clearance_m, max_slope_deg, min_area_sqm)",
                required=False,
                properties={                     # FIX 1: object type needs properties field
                    "min_clearance_m": {"type": "number"},
                    "max_slope_deg":   {"type": "number"},
                    "min_area_sqm":    {"type": "number"}
                }
            )
        ]
    ),
    
    "plan_path": Tool(
        name="plan_path",
        description="Plan path from current position to goal. Returns waypoints.",
        category="decision",
        parameters=[
            ToolParameter(
                name="start",
                type="object",
                description="Start position {x, y, z}",
                properties={"x": {"type": "number"}, "y": {"type": "number"}, "z": {"type": "number"}}  # FIX 1
            ),
            ToolParameter(
                name="goal",
                type="object",
                description="Goal position {x, y, z}",
                properties={"x": {"type": "number"}, "y": {"type": "number"}, "z": {"type": "number"}}  # FIX 1
            ),
            ToolParameter(
                name="avoid_obstacles",
                type="boolean",
                description="Whether to avoid detected obstacles",
                required=False,
                default=True
            )
        ]
    )
}


def get_tool(name: str) -> Optional[Tool]:
    """Get tool by name"""
    return TOOL_REGISTRY.get(name)


def get_tools_by_category(category: str) -> List[Tool]:
    """Get all tools in a category"""
    return [t for t in TOOL_REGISTRY.values() if t.category == category]


def get_all_tools() -> List[Tool]:
    """Get all tools"""
    return list(TOOL_REGISTRY.values())


def get_tool_names() -> List[str]:
    """Get all tool names"""
    return list(TOOL_REGISTRY.keys())


def get_openai_tools_schema() -> List[Dict]:
    """Get all tools in OpenAI function calling format"""
    return [tool.to_openai_schema() for tool in TOOL_REGISTRY.values()]


# Tool categories for ordering validation
PERCEPTION_TOOLS = {'get_camera_view', 'detect_objects', 'estimate_depth'}
FUSION_TOOLS = {'fuse_multi_camera'}
MOTION_TOOLS = {'track_motion'}
DECISION_TOOLS = {'assess_safety', 'plan_path'}

# Valid tool ordering: perception → fusion → motion → decision
TOOL_ORDER = {
    'get_camera_view': 1,
    'detect_objects': 2,
    'estimate_depth': 3,
    'fuse_multi_camera': 4,
    'track_motion': 5,
    'assess_safety': 6,
    'plan_path': 7
}