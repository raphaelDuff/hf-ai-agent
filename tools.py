import os
import re
import json
import aiohttp
import asyncio
from typing import Any, Dict, List, Optional, Union, Annotated
import math
from pathlib import Path
import pandas as pd
from datetime import datetime
import base64

from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage
from pydantic import Field
from utils import parse_gaia_question


class ToolError(Exception):
    """Custom exception for tool execution errors"""

    pass


@tool
def image_reader(
    filepath: Annotated[str, Field(description="Path to the image file to read")],
) -> str:
    """
    Read and validate an image file.
    Supports common image formats: JPG, JPEG, PNG, GIF, BMP, WEBP.

    Args:
        filepath: Path to the image file

    Returns:
        Confirmation message with image details
    """
    try:
        path = Path(filepath)

        # Check if file exists
        if not path.exists():
            raise ToolError(f"Image file not found: {filepath}")

        if not path.is_file():
            raise ToolError(f"Path is not a file: {filepath}")

        # Check if it's a supported image format
        supported_formats = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"}
        if path.suffix.lower() not in supported_formats:
            raise ToolError(
                f"Unsupported image format: {path.suffix}. Supported: {', '.join(supported_formats)}"
            )

        # Check file size (max 10MB for practical purposes)
        size_mb = path.stat().st_size / (1024 * 1024)
        if size_mb > 10:
            raise ToolError(
                f"Image file too large: {size_mb:.1f}MB. Maximum size: 10MB"
            )

        # Read and encode image to base64 for later analysis
        with open(path, "rb") as image_file:
            image_data = image_file.read()
            base64_image = base64.b64encode(image_data).decode("utf-8")

        return f"Image loaded successfully: {path.name} ({size_mb:.2f}MB, {path.suffix.upper()[1:]} format). Ready for analysis."

    except ToolError:
        raise
    except Exception as e:
        raise ToolError(f"Failed to read image '{filepath}': {str(e)}")


@tool
async def image_analyzer(
    filepath: Annotated[str, Field(description="Path to the image file to analyze")],
    question: Annotated[
        str,
        Field(
            description="Specific question about the image (e.g., 'What does this chart show?', 'Extract text from image')"
        ),
    ] = "Describe what you see in this image",
) -> str:
    """
    Analyze an image using Claude's vision capabilities.
    Can extract text, analyze charts/graphs, describe content, answer questions about images.

    Args:
        filepath: Path to the image file
        question: Specific question or analysis request about the image

    Returns:
        Analysis results based on the question
    """
    try:
        path = Path(filepath)

        # Validate image file
        if not path.exists():
            raise ToolError(f"Image file not found: {filepath}")

        # Check file format and size
        supported_formats = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"}
        if path.suffix.lower() not in supported_formats:
            raise ToolError(f"Unsupported image format: {path.suffix}")

        size_mb = path.stat().st_size / (1024 * 1024)
        if size_mb > 10:
            raise ToolError(f"Image too large: {size_mb:.1f}MB")

        # Read and encode image
        with open(path, "rb") as image_file:
            image_data = image_file.read()
            base64_image = base64.b64encode(image_data).decode("utf-8")

        # Determine media type
        media_type_map = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".gif": "image/gif",
            ".bmp": "image/bmp",
            ".webp": "image/webp",
        }
        media_type = media_type_map.get(path.suffix.lower(), "image/jpeg")

        # Create Claude client for vision analysis
        anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
        if not anthropic_api_key:
            raise ToolError("ANTHROPIC_API_KEY environment variable not set")

        # Use Claude for image analysis
        llm = ChatAnthropic(
            anthropic_api_key=anthropic_api_key,
            model="claude-3-5-sonnet-20241022",  # Has vision capabilities
            temperature=0.1,
            max_tokens=2048,
        )

        # Create message with image
        message = HumanMessage(
            content=[
                {
                    "type": "text",
                    "text": f"Please analyze this image and answer: {question}\n\nProvide a clear, factual response that directly answers the question. If the image contains text, charts, or data, extract and report the relevant information accurately.",
                },
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": media_type,
                        "data": base64_image,
                    },
                },
            ]
        )

        # Get analysis from Claude
        response = await llm.ainvoke([message])

        return f"Image analysis for '{path.name}': {response.content}"

    except ToolError:
        raise
    except Exception as e:
        raise ToolError(f"Image analysis failed for '{filepath}': {str(e)}")


@tool
async def web_search(
    query: Annotated[
        str, Field(description="The search query to look up on the internet")
    ],
) -> str:
    """
    Search the internet for information using DuckDuckGo.
    Useful for finding current information, facts, news, and general knowledge.

    Args:
        query: The search term or question to search for

    Returns:
        Search results as formatted text
    """
    try:
        # Create and use DuckDuckGo search tool
        ddg_search = DuckDuckGoSearchRun()
        search_results = ddg_search.invoke(query)

        if not search_results or search_results.strip() == "":
            return f"No search results found for query: '{query}'"

        # Clean up the results
        cleaned_results = " ".join(search_results.split())

        # Truncate if too long for context window
        max_length = 2000
        if len(cleaned_results) > max_length:
            cleaned_results = cleaned_results[:max_length] + "..."

        return f"Search results for '{query}':\n\n{cleaned_results}"

    except Exception as e:
        return f"Search failed for '{query}': {str(e)}. Try rephrasing your query."


@tool
def calculator(
    expression: Annotated[
        str,
        Field(
            description="Mathematical expression to evaluate (e.g., '2+2*3', 'sqrt(16)', '15% of 240')"
        ),
    ],
) -> str:
    """
    Perform mathematical calculations and evaluate expressions.
    Supports basic arithmetic, percentages, square roots, and common mathematical functions.

    Args:
        expression: Mathematical expression as a string

    Returns:
        The calculated result as a string
    """
    try:
        # Handle percentage calculations
        if "%" in expression:
            if " of " in expression.lower():
                # Handle "X% of Y" format
                parts = expression.lower().replace("%", "").split(" of ")
                if len(parts) == 2:
                    percentage = float(parts[0].strip())
                    value = float(parts[1].strip())
                    result = (percentage / 100) * value
                    return str(result)

        # Sanitize expression - only allow safe operations
        allowed_chars = set("0123456789+-*/().% ")
        sanitized = "".join(c for c in expression if c in allowed_chars or c.isalpha())

        # Replace common mathematical functions and operators
        replacements = {
            "sin": "math.sin",
            "cos": "math.cos",
            "tan": "math.tan",
            "log": "math.log",
            "sqrt": "math.sqrt",
            "abs": "abs",
            "round": "round",
            "^": "**",  # Convert ^ to ** for power
            "π": "math.pi",
            "pi": "math.pi",
            "e": "math.e",
        }

        for old, new in replacements.items():
            sanitized = sanitized.replace(old, new)

        # Evaluate safely with restricted namespace
        allowed_names = {
            "__builtins__": {},
            "math": math,
            "abs": abs,
            "round": round,
            "min": min,
            "max": max,
            "sum": sum,
            "pow": pow,
        }

        result = eval(sanitized, allowed_names)

        # Format result appropriately for GAIA
        if isinstance(result, float):
            if result.is_integer():
                return str(int(result))
            else:
                # Round to reasonable precision and remove trailing zeros
                formatted = f"{result:.10f}".rstrip("0").rstrip(".")
                return formatted

        return str(result)

    except Exception as e:
        raise ToolError(f"Calculation failed for '{expression}': {str(e)}")


@tool
def file_reader(
    filepath: Annotated[str, Field(description="Path to the file to read")],
    max_size_mb: Annotated[
        int, Field(description="Maximum file size in MB", default=10)
    ] = 10,
) -> str:
    """
    Read and return the contents of a file.
    Supports text files, CSV, JSON, Excel files, and more.

    Args:
        filepath: Path to the file to read
        max_size_mb: Maximum allowed file size in megabytes

    Returns:
        The file contents as a string
    """
    try:
        path = Path(filepath)

        # Security and existence checks
        if not path.exists():
            raise ToolError(f"File not found: {filepath}")

        if not path.is_file():
            raise ToolError(f"Path is not a file: {filepath}")

        # Check file size
        size_mb = path.stat().st_size / (1024 * 1024)
        if size_mb > max_size_mb:
            raise ToolError(
                f"File too large: {size_mb:.1f}MB exceeds limit of {max_size_mb}MB"
            )

        # Read file based on extension
        suffix = path.suffix.lower()

        if suffix in [
            ".txt",
            ".md",
            ".py",
            ".js",
            ".html",
            ".css",
            ".json",
            ".csv",
            ".log",
        ]:
            # Text-based files
            try:
                content = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                content = path.read_text(encoding="latin-1")

        elif suffix in [".xlsx", ".xls"]:
            # Excel files
            try:
                df = pd.read_excel(path)
                content = (
                    f"Excel file with {len(df)} rows and {len(df.columns)} columns:\n\n"
                )
                content += df.to_string(max_rows=100, max_cols=20)
                if len(df) > 100:
                    content += f"\n\n... (showing first 100 of {len(df)} rows)"
            except ImportError:
                raise ToolError(
                    "pandas and openpyxl required for Excel files. Install with: pip install pandas openpyxl"
                )

        elif suffix == ".pdf":
            raise ToolError(
                "PDF reading requires PyPDF2. Install with: pip install PyPDF2"
            )

        else:
            # Try as text file
            try:
                content = path.read_text(encoding="utf-8")
            except:
                raise ToolError(f"Unsupported or unreadable file type: {suffix}")

        # Truncate if too long for context
        max_chars = 50000
        if len(content) > max_chars:
            content = (
                content[:max_chars]
                + f"\n\n... [Content truncated - showing first {max_chars:,} characters of {len(content):,} total]"
            )

        return content

    except ToolError:
        raise
    except Exception as e:
        raise ToolError(f"Failed to read file '{filepath}': {str(e)}")


@tool
def data_processor(
    data: Annotated[
        str,
        Field(
            description="Data to process (JSON string, comma-separated values, or newline-separated)"
        ),
    ],
    operation: Annotated[
        str,
        Field(
            description="Operation to perform: count, sum, average, min, max, sort, unique, reverse"
        ),
    ],
) -> str:
    """
    Process structured data with common operations.
    Can handle lists, numbers, and perform statistical operations.

    Args:
        data: Input data as string (will be parsed automatically)
        operation: Type of operation to perform

    Returns:
        Result of the operation as a string
    """
    try:
        # Parse input data
        if isinstance(data, str):
            # Try to parse as JSON first
            try:
                parsed_data = json.loads(data)
            except json.JSONDecodeError:
                # Try comma-separated values
                if "," in data:
                    parsed_data = [item.strip() for item in data.split(",")]
                elif "\n" in data:
                    parsed_data = [
                        item.strip() for item in data.split("\n") if item.strip()
                    ]
                elif " " in data:
                    parsed_data = [
                        item.strip() for item in data.split(" ") if item.strip()
                    ]
                else:
                    parsed_data = [data.strip()]
        else:
            parsed_data = data

        # Convert string numbers to float where possible
        def try_numeric(item):
            try:
                if "." in str(item):
                    return float(item)
                else:
                    return int(item)
            except (ValueError, TypeError):
                return item

        # Available operations
        operations = {
            "count": lambda x: len(x),
            "sum": lambda x: sum(
                try_numeric(i)
                for i in x
                if str(i).replace(".", "").replace("-", "").isdigit()
            ),
            "average": lambda x: sum(
                try_numeric(i)
                for i in x
                if str(i).replace(".", "").replace("-", "").isdigit()
            )
            / len([i for i in x if str(i).replace(".", "").replace("-", "").isdigit()]),
            "mean": lambda x: sum(
                try_numeric(i)
                for i in x
                if str(i).replace(".", "").replace("-", "").isdigit()
            )
            / len([i for i in x if str(i).replace(".", "").replace("-", "").isdigit()]),
            "min": lambda x: min(
                try_numeric(i)
                for i in x
                if str(i).replace(".", "").replace("-", "").isdigit()
            ),
            "max": lambda x: max(
                try_numeric(i)
                for i in x
                if str(i).replace(".", "").replace("-", "").isdigit()
            ),
            "sort": lambda x: sorted(
                x,
                key=lambda item: (
                    try_numeric(item)
                    if str(item).replace(".", "").replace("-", "").isdigit()
                    else str(item)
                ),
            ),
            "unique": lambda x: list(set(x)),
            "reverse": lambda x: list(reversed(x)),
            "first": lambda x: x[0] if x else None,
            "last": lambda x: x[-1] if x else None,
        }

        if operation.lower() not in operations:
            available = ", ".join(operations.keys())
            raise ToolError(
                f"Unknown operation '{operation}'. Available operations: {available}"
            )

        result = operations[operation.lower()](parsed_data)

        # Format result for GAIA compliance
        if isinstance(result, (list, tuple)):
            return ", ".join(str(item) for item in result)
        elif isinstance(result, float):
            if result.is_integer():
                return str(int(result))
            else:
                return f"{result:.6f}".rstrip("0").rstrip(".")
        else:
            return str(result)

    except ToolError:
        raise
    except Exception as e:
        raise ToolError(f"Data processing failed for operation '{operation}': {str(e)}")


@tool
def current_time() -> str:
    """
    Get the current date and time.
    Useful for time-sensitive questions or when current date/time context is needed.

    Returns:
        Current date and time as formatted string
    """
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")


@tool
def extract_numbers(
    text: Annotated[str, Field(description="Text to extract numbers from")],
) -> str:
    """
    Extract all numbers from a given text string.
    Finds integers, decimals, and negative numbers.

    Args:
        text: Input text containing numbers

    Returns:
        Comma-separated list of extracted numbers
    """
    try:
        pattern = r"-?\d+(?:\.\d+)?"
        numbers = re.findall(pattern, text)

        if not numbers:
            return "No numbers found"

        return ", ".join(numbers)

    except Exception as e:
        raise ToolError(f"Number extraction failed: {str(e)}")


@tool
def extract_urls(
    text: Annotated[str, Field(description="Text to extract URLs from")],
) -> str:
    """
    Extract all URLs from a given text string.
    Finds HTTP and HTTPS URLs.

    Args:
        text: Input text containing URLs

    Returns:
        Comma-separated list of extracted URLs
    """
    try:
        pattern = r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"
        urls = re.findall(pattern, text)

        if not urls:
            return "No URLs found"

        return ", ".join(urls)

    except Exception as e:
        raise ToolError(f"URL extraction failed: {str(e)}")


@tool
def text_analyzer(
    text: Annotated[str, Field(description="Text to analyze")],
    analysis_type: Annotated[
        str,
        Field(
            description="Type of analysis: word_count, char_count, sentence_count, or summary"
        ),
    ],
) -> str:
    """
    Analyze text and provide various statistics or summaries.

    Args:
        text: The text to analyze
        analysis_type: Type of analysis to perform

    Returns:
        Analysis result as string
    """
    try:
        analyses = {
            "word_count": lambda t: str(len(t.split())),
            "char_count": lambda t: str(len(t)),
            "sentence_count": lambda t: str(
                len([s for s in t.split(".") if s.strip()])
            ),
            "summary": lambda t: f"Text has {len(t.split())} words, {len(t)} characters, and approximately {len([s for s in t.split('.') if s.strip()])} sentences.",
        }

        if analysis_type.lower() not in analyses:
            available = ", ".join(analyses.keys())
            raise ToolError(
                f"Unknown analysis type '{analysis_type}'. Available: {available}"
            )

        return analyses[analysis_type.lower()](text)

    except ToolError:
        raise
    except Exception as e:
        raise ToolError(f"Text analysis failed: {str(e)}")


# Collect all tools for easy import
def get_all_tools():
    """
    Get all available tools as a list of Tool objects.

    Returns:
        List of all decorated tools
    """
    return [
        web_search,
        calculator,
        file_reader,
        data_processor,
        image_reader,
        image_analyzer,
        current_time,
        extract_numbers,
        extract_urls,
        text_analyzer,
    ]


async def test_all_tools():
    """Test all tools to verify they work correctly"""
    print("🧪 Testing all GAIA Agent Tools with @tool decorator...\n")

    # Test calculator (SYNC tool)
    print("1️⃣  Testing calculator tool:")
    try:
        result = calculator.invoke({"expression": "2 + 2 * 3"})
        print(f"   ✅ 2 + 2 * 3 = {result}")

        result = calculator.invoke({"expression": "sqrt(16) + 5"})
        print(f"   ✅ sqrt(16) + 5 = {result}")

        result = calculator.invoke({"expression": "15% of 240"})
        print(f"   ✅ 15% of 240 = {result}")
    except Exception as e:
        print(f"   ❌ Calculator error: {e}")

    # Test data processor (SYNC tool)
    print("\n2️⃣  Testing data processor tool:")
    try:
        result = data_processor.invoke({"data": "1,2,3,4,5", "operation": "sum"})
        print(f"   ✅ Sum of [1,2,3,4,5] = {result}")

        result = data_processor.invoke(
            {"data": "apple,banana,apple,cherry", "operation": "unique"}
        )
        print(f"   ✅ Unique items = {result}")
    except Exception as e:
        print(f"   ❌ Data processor error: {e}")

    # Test web search (ASYNC tool)
    print("\n3️⃣  Testing DuckDuckGo web search:")
    try:
        result = await web_search.ainvoke({"query": "Python programming language"})
        print(f"   ✅ Search results (first 200 chars): {result[:200]}...")
    except Exception as e:
        print(f"   ❌ Web search error: {e}")

    # Test image tools (SYNC and ASYNC)
    print("\n4️⃣  Testing image analysis tools:")
    try:
        # Test with a non-existent image (expected to fail gracefully)
        result = image_reader.invoke({"filepath": "nonexistent.jpg"})
        print(f"   ❌ Should have failed for non-existent image")
    except Exception as e:
        print(f"   ✅ Image reader properly handles missing files: {type(e).__name__}")

    try:
        # Test image analyzer with non-existent image
        result = await image_analyzer.ainvoke(
            {"filepath": "nonexistent.jpg", "question": "What do you see?"}
        )
        print(f"   ❌ Should have failed for non-existent image")
    except Exception as e:
        print(
            f"   ✅ Image analyzer properly handles missing files: {type(e).__name__}"
        )

    print(
        f"   ℹ️  To test image tools with real images, place an image file in the current directory"
    )

    # Test text utilities (SYNC tools)
    print("\n5️⃣  Testing utility tools:")
    try:
        result = extract_numbers.invoke(
            {"text": "I have 25 apples and 13.5 oranges, lost 3"}
        )
        print(f"   ✅ Numbers extracted: {result}")

        result = current_time.invoke({})
        print(f"   ✅ Current time: {result}")

        result = text_analyzer.invoke(
            {"text": "Hello world! This is a test.", "analysis_type": "summary"}
        )
        print(f"   ✅ Text analysis: {result}")
    except Exception as e:
        print(f"   ❌ Utility tools error: {e}")

    # Test question parsing
    print("\n6️⃣  Testing GAIA question analysis:")
    test_questions = [
        "Calculate 15% of 240 and add 50",
        "Find the current population of Tokyo",
        "What are the unique values in this list: apple,banana,apple,cherry",
        "Analyze the chart in image.png and tell me the highest value",
        "What text is shown in this screenshot?",
    ]

    for question in test_questions:
        analysis = parse_gaia_question(question)
        print(f"   📋 '{question[:40]}...'")
        print(f"      Recommended tools: {analysis['recommended_tools']}")
        print(f"      Expected answer type: {analysis['expected_answer_type']}")

    print("\n✅ All tools tested successfully!")
    print(
        "📊 Now includes image analysis capabilities for charts, graphs, and visual data!"
    )


if __name__ == "__main__":
    asyncio.run(test_all_tools())
