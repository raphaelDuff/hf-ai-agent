# tools/__init__.py - Tools Module Initialization and Registry
"""
Universal Tools Module for GAIA Agent

This module provides access to all universal tools used by the GAIA agent system.
Each tool follows the Universal Tool design pattern:
- Domain-agnostic data extraction
- Standardized input/output format
- Claude handles all domain-specific reasoning
- No specialized tools for specific domains

Available Tools:
- UniversalImageAnalyzer: Visual content analysis
- UniversalAudioProcessor: Audio transcription and analysis
- UniversalVideoAnalyzer: Video content extraction
- YouTubeContentProcessor: YouTube video processing
- WebResearchEngine: Web information gathering
"""

import logging
from typing import Dict, Type, Optional, List, Any
from .base_tool import UniversalTool, ToolRegistry

# Tool imports with error handling
try:
    from .image_analyzer import UniversalImageAnalyzer

    IMAGE_ANALYZER_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Image Analyzer not available: {e}")
    IMAGE_ANALYZER_AVAILABLE = False
    UniversalImageAnalyzer = None

try:
    from .audio_processor import UniversalAudioProcessor

    AUDIO_PROCESSOR_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Audio Processor not available: {e}")
    AUDIO_PROCESSOR_AVAILABLE = False
    UniversalAudioProcessor = None

try:
    from .video_analyzer import UniversalVideoAnalyzer

    VIDEO_ANALYZER_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Video Analyzer not available: {e}")
    VIDEO_ANALYZER_AVAILABLE = False
    UniversalVideoAnalyzer = None

try:
    from .youtube_processor import YouTubeContentProcessor

    YOUTUBE_PROCESSOR_AVAILABLE = True
except ImportError as e:
    logging.warning(f"YouTube Processor not available: {e}")
    YOUTUBE_PROCESSOR_AVAILABLE = False
    YouTubeContentProcessor = None

try:
    from .web_researcher import WebResearchEngine

    WEB_RESEARCHER_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Web Research Engine not available: {e}")
    WEB_RESEARCHER_AVAILABLE = False
    WebResearchEngine = None

# Module-level logger
logger = logging.getLogger(__name__)

# Tool availability mapping
TOOL_AVAILABILITY = {
    "image_analyzer": IMAGE_ANALYZER_AVAILABLE,
    "audio_processor": AUDIO_PROCESSOR_AVAILABLE,
    "video_analyzer": VIDEO_ANALYZER_AVAILABLE,
    "youtube_processor": YOUTUBE_PROCESSOR_AVAILABLE,
    "web_researcher": WEB_RESEARCHER_AVAILABLE,
}

# Tool class mapping
TOOL_CLASSES = {
    "image_analyzer": UniversalImageAnalyzer,
    "audio_processor": UniversalAudioProcessor,
    "video_analyzer": UniversalVideoAnalyzer,
    "youtube_processor": YouTubeContentProcessor,
    "web_researcher": WebResearchEngine,
}


class ToolFactory:
    """Factory class for creating and managing universal tools"""

    def __init__(self):
        self.registry = ToolRegistry()
        self._initialized_tools = {}
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    def create_tool(
        self, tool_name: str, config: Dict[str, Any] = None
    ) -> Optional[UniversalTool]:
        """
        Create a tool instance by name

        Args:
            tool_name: Name of the tool to create
            config: Optional configuration for the tool

        Returns:
            Tool instance or None if not available
        """
        if tool_name not in TOOL_CLASSES:
            self.logger.error(f"Unknown tool: {tool_name}")
            return None

        if not TOOL_AVAILABILITY[tool_name]:
            self.logger.error(f"Tool not available: {tool_name}")
            return None

        tool_class = TOOL_CLASSES[tool_name]
        if tool_class is None:
            self.logger.error(f"Tool class not loaded: {tool_name}")
            return None

        try:
            # Special handling for tools that require configuration
            if tool_name == "web_researcher":
                api_key = config.get("tavily_api_key") if config else None
                tool = tool_class(api_key=api_key)
            elif tool_name == "youtube_processor":
                api_key = config.get("youtube_api_key") if config else None
                tool = tool_class(youtube_api_key=api_key)
            else:
                tool = tool_class()

            # Apply configuration if provided
            if config:
                tool.configure(config)

            self.logger.info(f"Created tool: {tool_name}")
            return tool

        except Exception as e:
            self.logger.error(f"Failed to create tool {tool_name}: {str(e)}")
            return None

    def create_all_available_tools(
        self, config: Dict[str, Any] = None
    ) -> Dict[str, UniversalTool]:
        """
        Create all available tools

        Args:
            config: Optional configuration dictionary

        Returns:
            Dictionary of tool name -> tool instance
        """
        tools = {}

        for tool_name in TOOL_CLASSES.keys():
            if TOOL_AVAILABILITY[tool_name]:
                tool_config = config.get(tool_name, {}) if config else {}
                tool = self.create_tool(tool_name, tool_config)
                if tool:
                    tools[tool_name] = tool
                    self.registry.register_tool(tool)

        self.logger.info(f"Created {len(tools)} tools: {list(tools.keys())}")
        return tools

    def get_tool_info(self, tool_name: str) -> Dict[str, Any]:
        """Get information about a specific tool"""
        return {
            "name": tool_name,
            "available": TOOL_AVAILABILITY.get(tool_name, False),
            "class": (
                TOOL_CLASSES.get(tool_name).__name__
                if TOOL_CLASSES.get(tool_name)
                else None
            ),
            "capabilities": self._get_tool_capabilities(tool_name),
        }

    def _get_tool_capabilities(self, tool_name: str) -> List[str]:
        """Get capabilities of a tool without instantiating it"""
        capabilities_map = {
            "image_analyzer": [
                "visual_description",
                "text_extraction",
                "spatial_analysis",
                "color_analysis",
                "object_detection",
                "composition_analysis",
            ],
            "audio_processor": [
                "transcription",
                "speaker_analysis",
                "audio_quality_analysis",
                "temporal_analysis",
                "frequency_analysis",
                "background_sound_detection",
            ],
            "video_analyzer": [
                "frame_extraction",
                "audio_extraction",
                "scene_detection",
                "motion_analysis",
                "metadata_extraction",
                "temporal_analysis",
                "object_tracking",
            ],
            "youtube_processor": [
                "youtube_download",
                "metadata_extraction",
                "subtitle_extraction",
                "audio_extraction",
                "video_processing",
                "chapter_detection",
            ],
            "web_researcher": [
                "web_search",
                "content_extraction",
                "source_verification",
                "temporal_filtering",
                "multi_source_synthesis",
            ],
        }

        return capabilities_map.get(tool_name, [])

    def get_system_info(self) -> Dict[str, Any]:
        """Get comprehensive system information"""
        total_tools = len(TOOL_CLASSES)
        available_tools = sum(TOOL_AVAILABILITY.values())

        tool_details = {}
        for tool_name in TOOL_CLASSES.keys():
            tool_details[tool_name] = self.get_tool_info(tool_name)

        return {
            "total_tools": total_tools,
            "available_tools": available_tools,
            "availability_rate": (available_tools / total_tools) * 100,
            "tool_details": tool_details,
            "missing_dependencies": self._get_missing_dependencies(),
        }

    def _get_missing_dependencies(self) -> Dict[str, List[str]]:
        """Identify missing dependencies for unavailable tools"""
        missing_deps = {}

        if not IMAGE_ANALYZER_AVAILABLE:
            missing_deps["image_analyzer"] = [
                "opencv-python",
                "Pillow",
                "numpy",
                "pytesseract",
                "easyocr",
            ]

        if not AUDIO_PROCESSOR_AVAILABLE:
            missing_deps["audio_processor"] = [
                "librosa",
                "soundfile",
                "scipy",
                "openai-whisper",
                "SpeechRecognition",
            ]

        if not VIDEO_ANALYZER_AVAILABLE:
            missing_deps["video_analyzer"] = [
                "opencv-python",
                "numpy",
                "librosa",
                "soundfile",
            ]

        if not YOUTUBE_PROCESSOR_AVAILABLE:
            missing_deps["youtube_processor"] = ["yt-dlp", "opencv-python", "librosa"]

        if not WEB_RESEARCHER_AVAILABLE:
            missing_deps["web_researcher"] = ["beautifulsoup4", "aiohttp"]

        return missing_deps


# Global factory instance
tool_factory = ToolFactory()


# Convenience functions
def create_tool(
    tool_name: str, config: Dict[str, Any] = None
) -> Optional[UniversalTool]:
    """Create a single tool instance"""
    return tool_factory.create_tool(tool_name, config)


def create_all_tools(config: Dict[str, Any] = None) -> Dict[str, UniversalTool]:
    """Create all available tools"""
    return tool_factory.create_all_available_tools(config)


def get_available_tools() -> List[str]:
    """Get list of available tool names"""
    return [name for name, available in TOOL_AVAILABILITY.items() if available]


def get_tool_info(tool_name: str) -> Dict[str, Any]:
    """Get information about a specific tool"""
    return tool_factory.get_tool_info(tool_name)


def get_system_info() -> Dict[str, Any]:
    """Get comprehensive system information"""
    return tool_factory.get_system_info()


def check_dependencies() -> bool:
    """Check if all required dependencies are available"""
    system_info = get_system_info()
    return system_info["availability_rate"] == 100.0


def print_system_status():
    """Print detailed system status"""
    info = get_system_info()

    print("🔧 GAIA Agent Tools System Status")
    print("=" * 40)
    print(
        f"Available Tools: {info['available_tools']}/{info['total_tools']} ({info['availability_rate']:.1f}%)"
    )
    print()

    for tool_name, details in info["tool_details"].items():
        status = "✅ Available" if details["available"] else "❌ Unavailable"
        print(f"{tool_name}: {status}")
        if details["capabilities"]:
            print(f"  Capabilities: {', '.join(details['capabilities'][:3])}...")

    if info["missing_dependencies"]:
        print("\n📦 Missing Dependencies:")
        for tool, deps in info["missing_dependencies"].items():
            print(f"  {tool}: {', '.join(deps)}")
        print("\nInstall missing dependencies with:")
        print("pip install -r requirements.txt")

    print(
        f"\n✅ System Ready: {'Yes' if info['availability_rate'] == 100 else 'Partial'}"
    )


# Export public interface
__all__ = [
    "UniversalTool",
    "ToolRegistry",
    "ToolFactory",
    "create_tool",
    "create_all_tools",
    "get_available_tools",
    "get_tool_info",
    "get_system_info",
    "check_dependencies",
    "print_system_status",
    "tool_factory",
    "TOOL_AVAILABILITY",
]

# Auto-initialize logging for the tools module
logging.getLogger(__name__).info(
    f"Tools module initialized - {sum(TOOL_AVAILABILITY.values())}/{len(TOOL_AVAILABILITY)} tools available"
)

# Conditional exports based on availability
if IMAGE_ANALYZER_AVAILABLE:
    __all__.append("UniversalImageAnalyzer")

if AUDIO_PROCESSOR_AVAILABLE:
    __all__.append("UniversalAudioProcessor")

if VIDEO_ANALYZER_AVAILABLE:
    __all__.append("UniversalVideoAnalyzer")

if YOUTUBE_PROCESSOR_AVAILABLE:
    __all__.append("YouTubeContentProcessor")

if WEB_RESEARCHER_AVAILABLE:
    __all__.append("WebResearchEngine")
