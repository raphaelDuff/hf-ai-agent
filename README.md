# GAIA Benchmark AI Agent

An advanced AI agent designed to answer questions from the GAIA benchmark using Claude 4 Sonnet with comprehensive multimodal capabilities.

## Overview

This project implements a sophisticated AI agent that can process and answer complex questions from the GAIA (General AI Assistant) benchmark. The agent leverages Claude 4 Sonnet for reasoning and multiple specialized tools for handling various types of media and data sources.

## Features

### Core Capabilities
- **Multimodal Processing**: Images, audio, video, and YouTube content analysis
- **Web Research**: Real-time information gathering using Tavily API
- **Intelligent Routing**: Automatic classification of input to determine optimal processing strategy
- **LangGraph Workflow**: Structured processing pipeline with conditional routing
- **Comprehensive Logging**: Detailed execution tracking and performance metrics

### Supported Media Types
- **Images**: JPG, PNG, GIF, BMP, TIFF, WebP
- **Audio**: MP3, WAV, M4A, FLAC, OGG, AAC
- **Video**: MP4, AVI, MOV, MKV, WebM, FLV
- **YouTube**: Full video content analysis and transcription
- **Web URLs**: Dynamic content extraction and analysis

### Processing Routes
1. **Media Only**: Pure multimodal analysis without external research
2. **Research Needed**: Web research for current/recent information
3. **Direct Reasoning**: Claude-only processing for knowledge-based questions
4. **Multimodal**: Combined media processing with web research

## Architecture

### Core Components

- **`app.py`**: Main application with LangGraph workflow orchestration
- **`config.py`**: Comprehensive configuration management and system setup
- **`tools/`**: Universal tool implementations for different media types
- **`monitoring/`**: Performance monitoring and logging utilities
- **`utils/`**: Data formatting, validation, and performance utilities

### Universal Tools Pattern

The agent uses a "Universal Tools" architecture where:
- Tools extract raw data without domain-specific reasoning
- Claude performs all domain-specific analysis and reasoning
- Consistent interfaces across all tools
- Built-in error handling, timeouts, and metrics tracking

## Installation

### Prerequisites
- Python 3.8+
- Required API keys (see Configuration section)

### Setup

1. **Clone the repository**
```bash
git clone <repository-url>
cd hf-monster-3rd
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Configure environment**
```bash
# Copy the example environment file
python config.py env

# Edit .env with your API keys
cp .env.example .env
# Add your API keys to .env
```

4. **Initialize the project**
```bash
python config.py init
```

## Configuration

### Required API Keys

Create a `.env` file with the following keys:

```env
# Required: Claude API Key from Anthropic
ANTHROPIC_API_KEY=your_anthropic_api_key_here

# Optional but recommended: Tavily API Key for web research
TAVILY_API_KEY=your_tavily_api_key_here

# Optional: OpenAI API Key for Whisper transcription
OPENAI_API_KEY=your_openai_api_key_here

# Optional: YouTube API Key for enhanced video processing
YOUTUBE_API_KEY=your_youtube_api_key_here
```

### System Configuration

The system can be configured via `config.json` or environment variables:

```json
{
  "max_tool_calls": 3,
  "timeout_seconds": 240,
  "log_level": "INFO",
  "output_dir": "outputs"
}
```

## Usage

### Basic Usage

```python
import asyncio
from app import BasicAgent, SystemConfig

async def main():
    # Initialize configuration
    config = SystemConfig()
    
    # Create agent
    agent = BasicAgent(config)
    
    # Process a question
    result = await agent.process_question(
        task_id="example_001",
        question="What animal is shown in this image?",
        media_files=["path/to/image.jpg"]
    )
    
    print(f"Answer: {result['model_answer']}")

if __name__ == "__main__":
    asyncio.run(main())
```

### Command Line Usage

```bash
# Run system diagnostics
python config.py check

# Initialize project setup
python config.py init

# Create sample environment file
python config.py env
```

## Project Structure

```
hf-monster-3rd/
├── app.py                      # Main application and workflow
├── config.py                   # Configuration management
├── requirements.txt            # Python dependencies
├── .env                       # Environment variables (create from .env.example)
├── tools/                     # Universal tools for media processing
│   ├── __init__.py
│   ├── base_tool.py          # Base class for all tools
│   ├── image_analyzer.py     # Image processing and OCR
│   ├── audio_processor.py    # Audio transcription and analysis
│   ├── video_analyzer.py     # Video frame extraction and analysis
│   ├── web_researcher.py     # Web search and content extraction
│   └── youtube_processor.py  # YouTube content processing
├── monitoring/               # Logging and performance tracking
│   ├── __init__.py
│   └── logger.py
├── utils/                   # Utility functions
│   ├── __init__.py
│   ├── data_formatting.py
│   ├── performance.py
│   └── validation.py
├── outputs/                 # Results and processed data
├── logs/                   # Application logs
└── README.md              # This file
```

## Examples

### Image Analysis
```python
result = await agent.process_question(
    task_id="image_001",
    question="What text is visible in this image?",
    media_files=["document.png"]
)
```

### YouTube Video Analysis
```python
result = await agent.process_question(
    task_id="youtube_001",
    question="Summarize the main points discussed in this video: https://youtube.com/watch?v=example"
)
```

### Research Question
```python
result = await agent.process_question(
    task_id="research_001",
    question="What were the latest developments in AI research in 2024?"
)
```

## Performance Monitoring

The agent includes comprehensive performance monitoring:

- **Execution metrics**: Response times, success rates, tool usage
- **Health monitoring**: Tool status and system resource usage
- **Detailed logging**: Full execution traces for debugging

```python
# Get performance metrics
metrics = agent.get_performance_metrics()
print(f"Success rate: {metrics['success_rate']}%")
print(f"Average execution time: {metrics['average_execution_time']}s")
```

## Development

### Adding New Tools

1. Create a new tool class inheriting from `UniversalTool`
2. Implement the `execute()` method
3. Register the tool in the agent initialization

```python
from tools.base_tool import UniversalTool

class CustomTool(UniversalTool):
    def __init__(self):
        super().__init__("Custom Tool")
        self.capabilities = ["custom_processing"]
    
    async def execute(self, input_data):
        # Your processing logic here
        raw_output = process_data(input_data)
        
        return self._standardize_output(
            raw_output=raw_output,
            metadata={"processing_type": "custom"}
        )
```

### Testing

```bash
# Install test dependencies
pip install pytest pytest-asyncio

# Run tests
pytest tests/
```

## Troubleshooting

### Common Issues

1. **Missing API Keys**
   ```bash
   python config.py check
   ```

2. **Import Errors**
   ```bash
   pip install -r requirements.txt
   ```

3. **Permission Issues**
   - Ensure proper file permissions for logs and outputs directories
   - Check API key validity

### System Diagnostics

```bash
python config.py check
```

This will verify:
- Environment variables
- Dependencies
- System resources
- Configuration validity
- Directory structure

## Dependencies

### Core Requirements
- `anthropic`: Claude API client
- `langgraph`: Workflow orchestration
- `aiohttp`, `aiofiles`: Async HTTP and file operations

### Media Processing
- `Pillow`, `opencv-python`: Image processing
- `librosa`, `soundfile`: Audio processing
- `yt-dlp`: YouTube content downloading

### Optional Enhancements
- `openai-whisper`: Advanced audio transcription
- `easyocr`: Enhanced OCR capabilities
- `beautifulsoup4`: Web content parsing

## Contributing

1. Fork the repository
2. Create a feature branch
3. Follow the Universal Tools pattern for new capabilities
4. Add comprehensive tests
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For issues and questions:
1. Check the troubleshooting section
2. Run system diagnostics: `python config.py check`
3. Review logs in the `logs/` directory
4. Open an issue with detailed error information

## Changelog

### v1.0.0
- Initial release with full GAIA benchmark support
- Universal Tools architecture
- LangGraph workflow implementation
- Comprehensive multimodal capabilities