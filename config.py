# config.py - Configuration Management
import os
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
from pathlib import Path
import json
import logging


@dataclass
class SystemConfig:
    """System configuration for GAIA Agent"""

    # API Keys
    anthropic_api_key: str = field(
        default_factory=lambda: os.getenv("ANTHROPIC_API_KEY", "")
    )
    tavily_api_key: str = field(default_factory=lambda: os.getenv("TAVILY_API_KEY", ""))
    openai_api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    youtube_api_key: str = field(
        default_factory=lambda: os.getenv("YOUTUBE_API_KEY", "")
    )

    # Performance Settings
    max_tool_calls: int = 3
    timeout_seconds: int = 30
    max_concurrent_tasks: int = 5

    # Logging Configuration
    log_level: str = "INFO"
    log_dir: str = "logs"
    enable_performance_logging: bool = True

    # Output Configuration
    output_dir: str = "outputs"
    save_intermediate_results: bool = True

    # Tool-Specific Settings
    tool_config: Dict[str, Any] = field(
        default_factory=lambda: {
            "image_analyzer": {
                "max_image_size_mb": 50,
                "supported_formats": [".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff"],
                "ocr_engines": ["tesseract", "easyocr"],
            },
            "audio_processor": {
                "max_audio_duration_minutes": 30,
                "supported_formats": [".wav", ".mp3", ".m4a", ".flac"],
                "sample_rate": 16000,
                "transcription_engines": ["whisper", "speech_recognition"],
            },
            "video_analyzer": {
                "max_video_duration_minutes": 10,
                "supported_formats": [".mp4", ".avi", ".mov", ".mkv"],
                "frame_extraction_interval": 1.0,
            },
            "web_researcher": {
                "max_results_per_query": 10,
                "max_content_length": 5000,
                "request_timeout": 10,
                "credible_domains": [
                    "wikipedia.org",
                    "britannica.com",
                    "edu",
                    "gov",
                    "org",
                    "reuters.com",
                    "ap.org",
                    "bbc.com",
                    "cnn.com",
                    "npr.org",
                ],
            },
        }
    )

    # Claude Configuration
    claude_config: Dict[str, Any] = field(
        default_factory=lambda: {
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 4000,
            "temperature": 0.1,
            "timeout": 60,
        }
    )

    # Performance Thresholds
    performance_thresholds: Dict[str, float] = field(
        default_factory=lambda: {
            "max_response_time": 120.0,
            "min_success_rate": 0.8,
            "max_error_rate": 0.2,
            "max_cpu_usage": 0.8,
            "max_memory_usage": 0.8,
        }
    )

    def validate(self) -> bool:
        """Validate configuration"""
        required_keys = [self.anthropic_api_key]
        return all(key for key in required_keys)

    def create_directories(self):
        """Create necessary directories"""
        for directory in [self.log_dir, self.output_dir]:
            Path(directory).mkdir(exist_ok=True)

    @classmethod
    def from_file(cls, config_file: str) -> "SystemConfig":
        """Load configuration from JSON file"""
        if Path(config_file).exists():
            with open(config_file, "r") as f:
                config_data = json.load(f)
            return cls(**config_data)
        else:
            # Create default config file
            config = cls()
            config.save_to_file(config_file)
            return config

    def save_to_file(self, config_file: str):
        """Save configuration to JSON file"""
        config_dict = {
            "max_tool_calls": self.max_tool_calls,
            "timeout_seconds": self.timeout_seconds,
            "max_concurrent_tasks": self.max_concurrent_tasks,
            "log_level": self.log_level,
            "log_dir": self.log_dir,
            "output_dir": self.output_dir,
            "tool_config": self.tool_config,
            "claude_config": self.claude_config,
            "performance_thresholds": self.performance_thresholds,
        }

        with open(config_file, "w") as f:
            json.dump(config_dict, f, indent=2)


# Environment Setup
def setup_environment():
    """Setup environment variables and configuration"""

    # Check for .env file
    env_file = Path(".env")
    if env_file.exists():
        try:
            from dotenv import load_dotenv

            load_dotenv()
            print("✓ Loaded environment variables from .env file")
        except ImportError:
            print(
                "Warning: python-dotenv not installed. Install with: pip install python-dotenv"
            )

    # Validate API keys
    required_env_vars = {
        "ANTHROPIC_API_KEY": "Claude API access",
        "TAVILY_API_KEY": "Web research (optional but recommended)",
    }

    missing_vars = []
    for var, description in required_env_vars.items():
        if not os.getenv(var):
            missing_vars.append(f"  - {var}: {description}")

    if missing_vars:
        print("⚠️  Missing environment variables:")
        print("\n".join(missing_vars))
        print(
            "\nCreate a .env file with these variables or set them in your environment."
        )
        return False

    print("✓ All required environment variables are set")
    return True


# Logging Setup
def setup_logging(config: SystemConfig):
    """Setup comprehensive logging"""

    # Create log directory
    Path(config.log_dir).mkdir(exist_ok=True)

    # Configure root logger
    logging.basicConfig(
        level=getattr(logging, config.log_level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(Path(config.log_dir) / "gaia_agent.log"),
            logging.StreamHandler(),
        ],
    )

    # Configure specific loggers
    loggers = [
        "gaia_agent",
        "tools.image_analyzer",
        "tools.audio_processor",
        "tools.video_analyzer",
        "tools.web_researcher",
        "monitoring.logger",
    ]

    for logger_name in loggers:
        logger = logging.getLogger(logger_name)
        logger.setLevel(getattr(logging, config.log_level))

    print(f"✓ Logging configured (level: {config.log_level})")


# Dependency Checker
def check_dependencies():
    """Check if required dependencies are installed"""

    required_packages = {
        "anthropic": "Claude API client",
        "langgraph": "Workflow orchestration",
        "aiohttp": "HTTP client",
        "aiofiles": "Async file operations",
        "numpy": "Numerical processing",
        "pillow": "Image processing",
        "opencv-python": "Computer vision",
        "librosa": "Audio processing",
        "beautifulsoup4": "Web scraping",
        "pytest": "Testing framework",
        "psutil": "System monitoring",
    }

    optional_packages = {
        "whisper": "Audio transcription",
        "easyocr": "OCR engine",
        "speech_recognition": "Speech recognition",
        "soundfile": "Audio file handling",
        "scipy": "Scientific computing",
        "scikit-learn": "Machine learning",
        "python-dotenv": "Environment variable loading",
    }

    missing_required = []
    missing_optional = []

    # Check required packages
    for package, description in required_packages.items():
        try:
            __import__(package.replace("-", "_"))
        except ImportError:
            missing_required.append(f"  - {package}: {description}")

    # Check optional packages
    for package, description in optional_packages.items():
        try:
            __import__(package.replace("-", "_"))
        except ImportError:
            missing_optional.append(f"  - {package}: {description}")

    if missing_required:
        print("❌ Missing required packages:")
        print("\n".join(missing_required))
        print("\nInstall with: pip install -r requirements.txt")
        return False

    print("✓ All required packages are installed")

    if missing_optional:
        print("\n⚠️  Optional packages not installed (some features may be limited):")
        print("\n".join(missing_optional))

    return True


# Performance Optimization
def optimize_system_settings():
    """Optimize system settings for performance"""

    recommendations = []

    try:
        import psutil

        # Check available memory
        memory = psutil.virtual_memory()
        if memory.available < 2 * 1024 * 1024 * 1024:  # Less than 2GB
            recommendations.append(
                "⚠️  Low available memory (< 2GB). Consider closing other applications."
            )

        # Check CPU cores
        cpu_count = psutil.cpu_count()
        if cpu_count < 4:
            recommendations.append(
                "⚠️  Limited CPU cores. Consider reducing max_concurrent_tasks in config."
            )

        # Check disk space
        disk = psutil.disk_usage("/")
        if disk.free < 1 * 1024 * 1024 * 1024:  # Less than 1GB
            recommendations.append(
                "⚠️  Low disk space (< 1GB). Clean up logs and outputs regularly."
            )

    except ImportError:
        recommendations.append(
            "⚠️  psutil not available. Install for system monitoring."
        )

    if recommendations:
        print("\nSystem Performance Recommendations:")
        print("\n".join(recommendations))
    else:
        print("✓ System appears optimally configured")


# Project Initialization
def initialize_project(config_file: str = "config.json"):
    """Initialize the GAIA agent project"""

    print("🚀 Initializing GAIA Agent Project")
    print("=" * 40)

    # Check environment
    if not setup_environment():
        return False

    # Check dependencies
    if not check_dependencies():
        return False

    # Load or create configuration
    config = SystemConfig.from_file(config_file)

    # Create directories
    config.create_directories()
    print(f"✓ Created directories: {config.log_dir}, {config.output_dir}")

    # Setup logging
    setup_logging(config)

    # System optimization
    optimize_system_settings()

    print("\n✅ Project initialization complete!")
    print(f"Configuration saved to: {config_file}")
    print("You can now run the GAIA agent.")

    return True


# Development Helper Functions
def create_sample_env_file():
    """Create a sample .env file"""
    env_content = """# GAIA Agent Environment Variables

# Required: Claude API Key from Anthropic
ANTHROPIC_API_KEY=your_anthropic_api_key_here

# Optional: Tavily API Key for web research
TAVILY_API_KEY=your_tavily_api_key_here

# Optional: OpenAI API Key (for Whisper transcription)
OPENAI_API_KEY=your_openai_api_key_here

# System Configuration
GAIA_LOG_LEVEL=INFO
GAIA_OUTPUT_DIR=outputs
GAIA_LOG_DIR=logs
"""

    with open(".env.example", "w") as f:
        f.write(env_content)

    print("✓ Created .env.example file")
    print("Copy it to .env and add your API keys")


def run_system_diagnostics():
    """Run comprehensive system diagnostics"""

    print("🔍 Running System Diagnostics")
    print("=" * 40)

    # Environment check
    print("\n1. Environment Variables:")
    api_keys = ["ANTHROPIC_API_KEY", "TAVILY_API_KEY", "OPENAI_API_KEY"]
    for key in api_keys:
        value = os.getenv(key)
        status = "✓ Set" if value else "❌ Missing"
        print(f"   {key}: {status}")

    # Dependencies check
    print("\n2. Dependencies:")
    check_dependencies()

    # System resources
    print("\n3. System Resources:")
    try:
        import psutil

        memory = psutil.virtual_memory()
        disk = psutil.disk_usage("/")

        print(f"   CPU cores: {psutil.cpu_count()}")
        print(
            f"   Memory: {memory.available / 1024**3:.1f}GB available / {memory.total / 1024**3:.1f}GB total"
        )
        print(f"   Disk space: {disk.free / 1024**3:.1f}GB available")
        print(f"   CPU usage: {psutil.cpu_percent()}%")
        print(f"   Memory usage: {memory.percent}%")

    except ImportError:
        print("   psutil not available for system monitoring")

    # Configuration check
    print("\n4. Configuration:")
    config_file = "config.json"
    if Path(config_file).exists():
        print(f"   ✓ Configuration file exists: {config_file}")
        config = SystemConfig.from_file(config_file)
        print(f"   Validation: {'✓ Valid' if config.validate() else '❌ Invalid'}")
    else:
        print(f"   ❌ Configuration file missing: {config_file}")

    # Directory structure
    print("\n5. Directory Structure:")
    for directory in ["logs", "outputs", "tools", "tests", "monitoring"]:
        exists = Path(directory).exists()
        status = "✓ Exists" if exists else "❌ Missing"
        print(f"   {directory}/: {status}")

    print("\n✅ Diagnostics complete!")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        command = sys.argv[1]

        if command == "init":
            initialize_project()
        elif command == "check":
            run_system_diagnostics()
        elif command == "env":
            create_sample_env_file()
        else:
            print("Usage: python config.py [init|check|env]")
            print("  init  - Initialize project")
            print("  check - Run diagnostics")
            print("  env   - Create sample .env file")
    else:
        initialize_project()
