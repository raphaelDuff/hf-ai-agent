# utils/data_formatting.py - Data Formatting Utilities
import json
import csv
import xml.etree.ElementTree as ET
from typing import Dict, Any, List
from datetime import datetime
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class DataFormatter:
    """Utility class for formatting and converting data between different formats"""

    @staticmethod
    def format_gaia_response(
        task_id: str,
        model_answer: str,
        reasoning_trace: str,
        tools_used: List[str],
        claude_reasoning_steps: List[str],
        execution_time: float = 0.0,
        success: bool = True,
        error: str = None,
    ) -> Dict[str, Any]:
        """Format response in GAIA benchmark format"""

        response = {
            "task_id": task_id,
            "model_answer": model_answer,
            "reasoning_trace": reasoning_trace,
            "tools_used": tools_used,
            "claude_reasoning_steps": claude_reasoning_steps,
            "execution_time": round(execution_time, 3),
            "success": success,
            "timestamp": datetime.now().isoformat(),
        }

        if error:
            response["error"] = error

        return response

    @staticmethod
    def format_tool_output(
        tool_name: str,
        raw_output: Any,
        metadata: Dict[str, Any] = None,
        success: bool = True,
        execution_time: float = 0.0,
    ) -> Dict[str, Any]:
        """Format tool output in standardized format"""

        formatted_output = {
            "tool_name": tool_name,
            "raw_output": str(raw_output) if raw_output is not None else "",
            "success": success,
            "execution_time": round(execution_time, 3),
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {},
        }

        return formatted_output

    @staticmethod
    def format_claude_prompt(
        question: str,
        tool_outputs: List[Dict[str, Any]],
        task_id: str = None,
        context: str = None,
    ) -> str:
        """Format comprehensive prompt for Claude reasoning"""

        prompt_parts = []

        if task_id:
            prompt_parts.append(f"Task ID: {task_id}")

        prompt_parts.append(f"Question: {question}")

        if context:
            prompt_parts.append(f"Context: {context}")

        if tool_outputs:
            prompt_parts.append("\nAvailable Data from Tools:")
            for i, output in enumerate(tool_outputs, 1):
                prompt_parts.append(f"\nTool {i}: {output.get('tool_name', 'Unknown')}")
                if output.get("success", True):
                    prompt_parts.append(
                        f"Output: {output.get('raw_output', 'No output')}"
                    )
                    if output.get("metadata"):
                        prompt_parts.append(
                            f"Metadata: {json.dumps(output['metadata'], indent=2)}"
                        )
                else:
                    prompt_parts.append(
                        f"Error: {output.get('error', 'Unknown error')}"
                    )
        else:
            prompt_parts.append(
                "\nNo tool outputs available - use your knowledge to answer directly."
            )

        prompt_parts.extend(
            [
                "\nInstructions:",
                "1. Analyze all available data in the context of the question",
                "2. Apply logical reasoning, inference, and domain expertise",
                "3. Cross-reference information for consistency",
                "4. Provide a precise, factual answer that matches expected format",
                "5. If information is insufficient, state limitations clearly",
                "6. Show your step-by-step reasoning process",
                "",
                "IMPORTANT: The final answer must be precise and factual. GAIA benchmark requires exact matching.",
                "",
                "Format your response as:",
                "REASONING: [Your detailed step-by-step analysis]",
                "ANSWER: [Final precise answer only]",
            ]
        )

        return "\n".join(prompt_parts)

    @staticmethod
    def sanitize_filename(filename: str) -> str:
        """Sanitize filename for safe file system usage"""
        import re

        # Remove or replace invalid characters
        sanitized = re.sub(r'[<>:"/\\|?*]', "_", filename)

        # Remove multiple underscores
        sanitized = re.sub(r"_+", "_", sanitized)

        # Trim and limit length
        sanitized = sanitized.strip("_")[:200]

        return sanitized if sanitized else "unnamed_file"

    @staticmethod
    def extract_answer_from_reasoning(reasoning_text: str) -> str:
        """Extract final answer from Claude's reasoning response"""

        # Look for explicit ANSWER: marker
        lines = reasoning_text.split("\n")
        for line in lines:
            if line.strip().startswith("ANSWER:"):
                return line.split("ANSWER:", 1)[1].strip()

        # Fallback patterns
        answer_patterns = [
            r"(?i)(?:final answer|answer|conclusion):\s*(.+)",
            r"(?i)(?:therefore|thus|so),?\s*(.+)",
            r"(?i)the answer is\s*(.+)",
        ]

        for pattern in answer_patterns:
            import re

            match = re.search(pattern, reasoning_text)
            if match:
                return match.group(1).strip()

        # Last resort: use last non-empty line
        for line in reversed(lines):
            if line.strip():
                return line.strip()

        return "Unable to determine answer"

    @staticmethod
    def convert_to_csv(data: List[Dict[str, Any]], output_path: str) -> bool:
        """Convert list of dictionaries to CSV format"""
        try:
            if not data:
                logger.warning("No data to convert to CSV")
                return False

            with open(output_path, "w", newline="", encoding="utf-8") as csvfile:
                fieldnames = data[0].keys()
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

                writer.writeheader()
                for row in data:
                    # Convert complex objects to strings
                    clean_row = {}
                    for key, value in row.items():
                        if isinstance(value, (dict, list)):
                            clean_row[key] = json.dumps(value)
                        else:
                            clean_row[key] = str(value) if value is not None else ""
                    writer.writerow(clean_row)

            logger.info(
                f"Successfully converted {len(data)} records to CSV: {output_path}"
            )
            return True

        except Exception as e:
            logger.error(f"Failed to convert to CSV: {str(e)}")
            return False

    @staticmethod
    def convert_to_json(data: Any, output_path: str, indent: int = 2) -> bool:
        """Convert data to JSON format"""
        try:
            with open(output_path, "w", encoding="utf-8") as jsonfile:
                json.dump(
                    data, jsonfile, indent=indent, ensure_ascii=False, default=str
                )

            logger.info(f"Successfully converted data to JSON: {output_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to convert to JSON: {str(e)}")
            return False

    @staticmethod
    def convert_to_xml(
        data: Dict[str, Any], output_path: str, root_name: str = "gaia_result"
    ) -> bool:
        """Convert dictionary to XML format"""
        try:

            def dict_to_xml(parent, data_dict):
                for key, value in data_dict.items():
                    # Sanitize key name for XML
                    clean_key = re.sub(r"[^a-zA-Z0-9_]", "_", str(key))

                    if isinstance(value, dict):
                        child = ET.SubElement(parent, clean_key)
                        dict_to_xml(child, value)
                    elif isinstance(value, list):
                        for item in value:
                            child = ET.SubElement(parent, clean_key)
                            if isinstance(item, dict):
                                dict_to_xml(child, item)
                            else:
                                child.text = str(item)
                    else:
                        child = ET.SubElement(parent, clean_key)
                        child.text = str(value) if value is not None else ""

            root = ET.Element(root_name)
            dict_to_xml(root, data)

            tree = ET.ElementTree(root)
            tree.write(output_path, encoding="utf-8", xml_declaration=True)

            logger.info(f"Successfully converted data to XML: {output_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to convert to XML: {str(e)}")
            return False


# utils/validation.py - Data Validation Utilities
import re
from typing import Any, Dict, List, Optional, Tuple, Union
from urllib.parse import urlparse
from pathlib import Path
import mimetypes


class ValidationError(Exception):
    """Custom exception for validation errors"""

    pass


class DataValidator:
    """Utility class for validating various types of data and inputs"""

    @staticmethod
    def validate_task_id(task_id: str) -> Tuple[bool, str]:
        """Validate GAIA task ID format"""
        if not task_id:
            return False, "Task ID cannot be empty"

        if not isinstance(task_id, str):
            return False, "Task ID must be a string"

        # Basic format validation
        if len(task_id) > 100:
            return False, "Task ID too long (max 100 characters)"

        # Check for valid characters
        if not re.match(r"^[a-zA-Z0-9_-]+$", task_id):
            return (
                False,
                "Task ID contains invalid characters (use only letters, numbers, _, -)",
            )

        return True, "Valid task ID"

    @staticmethod
    def validate_question(question: str) -> Tuple[bool, str]:
        """Validate question format and content"""
        if not question:
            return False, "Question cannot be empty"

        if not isinstance(question, str):
            return False, "Question must be a string"

        if len(question.strip()) < 5:
            return False, "Question too short (minimum 5 characters)"

        if len(question) > 5000:
            return False, "Question too long (maximum 5000 characters)"

        return True, "Valid question"

    @staticmethod
    def validate_media_file(file_path: str) -> Tuple[bool, str]:
        """Validate media file existence and format"""
        if not file_path:
            return False, "File path cannot be empty"

        path = Path(file_path)

        if not path.exists():
            return False, f"File does not exist: {file_path}"

        if not path.is_file():
            return False, f"Path is not a file: {file_path}"

        # Check file size (limit to 100MB)
        max_size = 100 * 1024 * 1024  # 100MB
        if path.stat().st_size > max_size:
            return (
                False,
                f"File too large: {path.stat().st_size / (1024*1024):.1f}MB (max 100MB)",
            )

        # Validate MIME type
        mime_type, _ = mimetypes.guess_type(file_path)

        supported_types = [
            "image/",
            "audio/",
            "video/",
            "application/pdf",
            "text/",
            "application/json",
        ]

        if mime_type and not any(mime_type.startswith(t) for t in supported_types):
            return False, f"Unsupported file type: {mime_type}"

        return True, "Valid media file"

    @staticmethod
    def validate_url(url: str) -> Tuple[bool, str]:
        """Validate URL format"""
        if not url:
            return False, "URL cannot be empty"

        try:
            result = urlparse(url)
            if not all([result.scheme, result.netloc]):
                return False, "Invalid URL format"

            if result.scheme not in ["http", "https"]:
                return False, "URL must use HTTP or HTTPS protocol"

            return True, "Valid URL"

        except Exception as e:
            return False, f"URL validation error: {str(e)}"

    @staticmethod
    def validate_youtube_url(url: str) -> Tuple[bool, str]:
        """Validate YouTube URL specifically"""
        url_valid, url_message = DataValidator.validate_url(url)
        if not url_valid:
            return False, url_message

        youtube_patterns = [
            r"(?:https?://)?(?:www\.)?youtube\.com/watch\?v=([a-zA-Z0-9_-]+)",
            r"(?:https?://)?(?:www\.)?youtu\.be/([a-zA-Z0-9_-]+)",
            r"(?:https?://)?(?:www\.)?youtube\.com/embed/([a-zA-Z0-9_-]+)",
        ]

        if not any(re.match(pattern, url) for pattern in youtube_patterns):
            return False, "Not a valid YouTube URL"

        return True, "Valid YouTube URL"

    @staticmethod
    def validate_api_key(api_key: str, key_type: str = "generic") -> Tuple[bool, str]:
        """Validate API key format"""
        if not api_key:
            return False, f"{key_type} API key cannot be empty"

        if not isinstance(api_key, str):
            return False, f"{key_type} API key must be a string"

        # Basic format checks based on key type
        if key_type.lower() == "anthropic":
            if not api_key.startswith("sk-ant-"):
                return False, "Anthropic API key should start with 'sk-ant-'"
            if len(api_key) < 20:
                return False, "Anthropic API key too short"

        elif key_type.lower() == "openai":
            if not api_key.startswith("sk-"):
                return False, "OpenAI API key should start with 'sk-'"
            if len(api_key) < 20:
                return False, "OpenAI API key too short"

        # General validation
        if len(api_key) > 200:
            return False, f"{key_type} API key too long"

        return True, f"Valid {key_type} API key format"

    @staticmethod
    def validate_config(config: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate system configuration"""
        errors = []

        # Required fields
        required_fields = ["anthropic_api_key", "max_tool_calls", "timeout_seconds"]
        for field in required_fields:
            if field not in config:
                errors.append(f"Missing required field: {field}")

        # Validate specific fields
        if "max_tool_calls" in config:
            if (
                not isinstance(config["max_tool_calls"], int)
                or config["max_tool_calls"] <= 0
            ):
                errors.append("max_tool_calls must be a positive integer")

        if "timeout_seconds" in config:
            if (
                not isinstance(config["timeout_seconds"], (int, float))
                or config["timeout_seconds"] <= 0
            ):
                errors.append("timeout_seconds must be a positive number")

        if "log_level" in config:
            valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
            if config["log_level"] not in valid_levels:
                errors.append(f"log_level must be one of: {valid_levels}")

        # Validate API keys if present
        if "anthropic_api_key" in config:
            valid, message = DataValidator.validate_api_key(
                config["anthropic_api_key"], "anthropic"
            )
            if not valid:
                errors.append(message)

        return len(errors) == 0, errors

    @staticmethod
    def validate_tool_output(tool_output: Dict[str, Any]) -> Tuple[bool, str]:
        """Validate tool output format"""
        required_fields = ["tool_name", "raw_output", "success"]

        for field in required_fields:
            if field not in tool_output:
                return False, f"Missing required field: {field}"

        if not isinstance(tool_output["success"], bool):
            return False, "success field must be boolean"

        if not isinstance(tool_output["tool_name"], str):
            return False, "tool_name must be string"

        return True, "Valid tool output format"

    @staticmethod
    def validate_gaia_response(response: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate GAIA benchmark response format"""
        errors = []

        required_fields = ["task_id", "model_answer", "reasoning_trace", "tools_used"]
        for field in required_fields:
            if field not in response:
                errors.append(f"Missing required field: {field}")

        # Validate field types
        if "task_id" in response:
            valid, message = DataValidator.validate_task_id(response["task_id"])
            if not valid:
                errors.append(f"Invalid task_id: {message}")

        if "tools_used" in response:
            if not isinstance(response["tools_used"], list):
                errors.append("tools_used must be a list")
            else:
                for tool in response["tools_used"]:
                    if not isinstance(tool, str):
                        errors.append("All tools in tools_used must be strings")
                        break

        if "success" in response:
            if not isinstance(response["success"], bool):
                errors.append("success must be boolean")

        return len(errors) == 0, errors


# utils/performance.py - Performance Optimization Utilities
import asyncio
import time
import functools
import threading
from typing import Callable, Any, Dict, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import psutil
import gc


class PerformanceOptimizer:
    """Utility class for performance optimization"""

    def __init__(self):
        self.thread_pool = ThreadPoolExecutor(max_workers=4)
        self.performance_cache = {}
        self.cache_lock = threading.Lock()

    @staticmethod
    def time_function(func: Callable) -> Callable:
        """Decorator to time function execution"""

        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()
            result = await func(*args, **kwargs)
            execution_time = time.time() - start_time

            # Add timing info to result if it's a dict
            if isinstance(result, dict):
                result["execution_time"] = execution_time

            return result

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time

            if isinstance(result, dict):
                result["execution_time"] = execution_time

            return result

        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper

    @staticmethod
    def memory_monitor(func: Callable) -> Callable:
        """Decorator to monitor memory usage"""

        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            process = psutil.Process()
            start_memory = process.memory_info().rss / 1024 / 1024  # MB

            result = await func(*args, **kwargs)

            end_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_delta = end_memory - start_memory

            if isinstance(result, dict):
                result["memory_usage"] = {
                    "start_mb": start_memory,
                    "end_mb": end_memory,
                    "delta_mb": memory_delta,
                }

            return result

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            process = psutil.Process()
            start_memory = process.memory_info().rss / 1024 / 1024  # MB

            result = func(*args, **kwargs)

            end_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_delta = end_memory - start_memory

            if isinstance(result, dict):
                result["memory_usage"] = {
                    "start_mb": start_memory,
                    "end_mb": end_memory,
                    "delta_mb": memory_delta,
                }

            return result

        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper

    def cache_result(self, key: str, ttl: int = 300) -> Callable:
        """Decorator to cache function results with TTL"""

        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                cache_key = f"{key}:{hash(str(args) + str(kwargs))}"

                with self.cache_lock:
                    if cache_key in self.performance_cache:
                        cached_data, timestamp = self.performance_cache[cache_key]
                        if time.time() - timestamp < ttl:
                            return cached_data

                result = await func(*args, **kwargs)

                with self.cache_lock:
                    self.performance_cache[cache_key] = (result, time.time())

                return result

            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                cache_key = f"{key}:{hash(str(args) + str(kwargs))}"

                with self.cache_lock:
                    if cache_key in self.performance_cache:
                        cached_data, timestamp = self.performance_cache[cache_key]
                        if time.time() - timestamp < ttl:
                            return cached_data

                result = func(*args, **kwargs)

                with self.cache_lock:
                    self.performance_cache[cache_key] = (result, time.time())

                return result

            return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper

        return decorator

    @staticmethod
    def optimize_memory():
        """Force garbage collection and memory optimization"""
        gc.collect()

        # Get memory info
        process = psutil.Process()
        memory_info = process.memory_info()

        return {
            "memory_mb": memory_info.rss / 1024 / 1024,
            "gc_collected": gc.get_count(),
        }

    @staticmethod
    def get_system_resources() -> Dict[str, Any]:
        """Get current system resource usage"""
        return {
            "cpu_percent": psutil.cpu_percent(interval=1),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_percent": psutil.disk_usage("/").percent,
            "available_memory_gb": psutil.virtual_memory().available
            / 1024
            / 1024
            / 1024,
            "cpu_count": psutil.cpu_count(),
            "timestamp": time.time(),
        }

    async def parallel_execute(
        self, tasks: List[Callable], max_workers: int = None
    ) -> List[Any]:
        """Execute multiple tasks in parallel"""
        if max_workers is None:
            max_workers = min(len(tasks), 4)

        semaphore = asyncio.Semaphore(max_workers)

        async def execute_with_semaphore(task):
            async with semaphore:
                if asyncio.iscoroutinefunction(task):
                    return await task()
                else:
                    loop = asyncio.get_event_loop()
                    return await loop.run_in_executor(self.thread_pool, task)

        results = await asyncio.gather(
            *[execute_with_semaphore(task) for task in tasks]
        )
        return results

    def cleanup(self):
        """Cleanup resources"""
        self.thread_pool.shutdown(wait=True)
        with self.cache_lock:
            self.performance_cache.clear()


# Usage examples
def example_usage():
    """Example usage of utility functions"""

    # Data formatting
    formatter = DataFormatter()

    # Format GAIA response
    response = formatter.format_gaia_response(
        task_id="example_001",
        model_answer="The answer is 42",
        reasoning_trace="Mathematical analysis -> Universal truth",
        tools_used=["calculator", "philosophy_engine"],
        claude_reasoning_steps=["Step 1", "Step 2", "Final answer"],
    )
    print("Formatted GAIA response:", json.dumps(response, indent=2))

    # Data validation
    validator = DataValidator()

    # Validate task ID
    valid, message = validator.validate_task_id("valid_task_123")
    print(f"Task ID validation: {valid} - {message}")

    # Validate question
    valid, message = validator.validate_question("What is the meaning of life?")
    print(f"Question validation: {valid} - {message}")

    # Performance optimization
    optimizer = PerformanceOptimizer()

    # Get system resources
    resources = optimizer.get_system_resources()
    print("System resources:", json.dumps(resources, indent=2))

    # Cleanup
    optimizer.cleanup()


if __name__ == "__main__":
    example_usage()
