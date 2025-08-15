# app.py - Main GAIA Agent Application
import asyncio
import logging
import time
from typing import Dict, List, Any, Optional, TypedDict, Tuple
from dataclasses import dataclass
from enum import Enum
import json
import os
from pathlib import Path
from langgraph.graph import StateGraph, END
from anthropic import AsyncAnthropic
import aiofiles


# Configuration
@dataclass
class SystemConfig:
    anthropic_api_key: str = os.getenv("ANTHROPIC_API_KEY")
    tavily_api_key: str = os.getenv("TAVILY_API_KEY")
    youtube_api_key: str = os.getenv("YOUTUBE_API_KEY")
    max_tool_calls: int = 3
    timeout_seconds: int = 240
    log_level: str = "INFO"
    output_dir: str = "outputs"

    def validate(self) -> bool:
        required_keys = [self.anthropic_api_key]
        return all(key is not None for key in required_keys)


# State Management
class ProcessingRoute(Enum):
    MEDIA_ONLY = "media_only"
    RESEARCH_NEEDED = "research_needed"
    DIRECT_REASONING = "direct_reasoning"
    MULTIMODAL = "multimodal"


class AgentState(TypedDict):
    task_id: str
    question: str
    media_files: List[str]
    processing_route: str
    tool_outputs: List[Dict[str, Any]]
    claude_reasoning: str
    final_answer: str
    reasoning_trace: List[str]
    tools_used: List[str]
    error: Optional[str]
    execution_time: float
    start_time: float


class BasicAgent:
    """Enhanced GAIA Benchmark Agent with LangGraph and Claude 4 Sonnet"""

    def __init__(self, config: SystemConfig):
        self.config = config
        self.logger = self._setup_logging()
        self.claude_client = AsyncAnthropic(api_key=config.anthropic_api_key)
        self.tools = {}
        self.performance_metrics = {
            "total_queries": 0,
            "successful_queries": 0,
            "tool_usage_stats": {},
            "average_execution_time": 0.0,
        }

        # Initialize tools
        self._initialize_tools()

        # Build LangGraph workflow
        self.graph = self._build_workflow()

        # Ensure output directory exists
        Path(config.output_dir).mkdir(exist_ok=True)

        self.logger.info("GAIA Agent initialized successfully")

    def _setup_logging(self) -> logging.Logger:
        """Configure comprehensive logging for debugging and monitoring"""
        logger = logging.getLogger("gaia_agent")
        logger.setLevel(getattr(logging, self.config.log_level))

        # Create formatters
        detailed_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s"
        )

        # File handler for detailed logs
        file_handler = logging.FileHandler("gaia_agent.log")
        file_handler.setFormatter(detailed_formatter)
        file_handler.setLevel(logging.DEBUG)

        # Console handler for important messages
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
        console_handler.setLevel(logging.INFO)

        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

        return logger

    def _initialize_tools(self):
        """Initialize all universal tools"""
        from tools.image_analyzer import UniversalImageAnalyzer
        from tools.audio_processor import UniversalAudioProcessor
        from tools.video_analyzer import UniversalVideoAnalyzer
        from tools.web_researcher import WebResearchEngine
        from tools.youtube_processor import YouTubeContentProcessor

        self.tools = {
            "image_analyzer": UniversalImageAnalyzer(),
            "audio_processor": UniversalAudioProcessor(),
            "video_analyzer": UniversalVideoAnalyzer(),
            "youtube_processor": YouTubeContentProcessor(self.config.youtube_api_key),
            "web_researcher": WebResearchEngine(self.config.tavily_api_key),
        }

        self.logger.info(f"Initialized {len(self.tools)} universal tools")

    def _build_workflow(self) -> StateGraph:
        """Build the LangGraph workflow for systematic processing"""
        workflow = StateGraph(AgentState)

        # Add processing nodes
        workflow.add_node("classify_input", self.classify_input_node)
        workflow.add_node("process_media", self.process_media_node)
        workflow.add_node("web_research", self.web_research_node)
        workflow.add_node("claude_reasoning", self.claude_reasoning_node)
        workflow.add_node("synthesize_answer", self.synthesize_answer_node)

        # Define workflow edges
        workflow.set_entry_point("classify_input")

        # Conditional routing based on input classification
        workflow.add_conditional_edges(
            "classify_input",
            self._route_processing,
            {
                ProcessingRoute.MEDIA_ONLY.value: "process_media",
                ProcessingRoute.RESEARCH_NEEDED.value: "web_research",
                ProcessingRoute.DIRECT_REASONING.value: "claude_reasoning",
                ProcessingRoute.MULTIMODAL.value: "process_media",
            },
        )

        # Connect media processing to reasoning
        workflow.add_edge("process_media", "claude_reasoning")
        workflow.add_edge("web_research", "claude_reasoning")
        workflow.add_edge("claude_reasoning", "synthesize_answer")
        workflow.add_edge("synthesize_answer", END)

        return workflow.compile()

    async def classify_input_node(self, state: AgentState) -> AgentState:
        """Classify input to determine processing route"""
        start_time = time.time()
        self.logger.info(f"Classifying input for task {state['task_id']}")

        has_uploaded_media = bool(state.get("media_files"))
        question = state["question"].lower()

        # Check for media URLs in question text
        has_media_urls = self._has_media_urls_in_question(state["question"])
        has_any_media = has_uploaded_media or has_media_urls

        # Enhanced classification logic
        if has_any_media and any(
            keyword in question
            for keyword in ["search", "find", "lookup", "current", "recent"]
        ):
            route = ProcessingRoute.MULTIMODAL.value
        elif has_any_media:
            route = ProcessingRoute.MEDIA_ONLY.value
        elif any(
            keyword in question
            for keyword in [
                "current",
                "latest",
                "recent",
                "2024",
                "2025",
                "today",
                "now",
            ]
        ):
            route = ProcessingRoute.RESEARCH_NEEDED.value
        else:
            route = ProcessingRoute.DIRECT_REASONING.value

        self.logger.info(
            f"Classified as: {route} (uploaded_media: {has_uploaded_media}, media_urls: {has_media_urls})"
        )

        return {
            **state,
            "processing_route": route,
            "reasoning_trace": [f"Input classified as: {route}"],
            "start_time": start_time,
        }

    def _has_media_urls_in_question(self, question: str) -> bool:
        """Check if question contains media URLs"""
        import re

        # YouTube URL patterns
        youtube_patterns = [
            r"youtube\.com/watch\?v=",
            r"youtu\.be/",
            r"youtube\.com/embed/",
            r"youtube\.com/v/",
        ]

        # Direct media URL patterns
        media_patterns = [
            r"https?://[^\s]+\.(?:jpg|jpeg|png|gif|bmp|tiff|webp)",  # Images
            r"https?://[^\s]+\.(?:mp4|avi|mov|mkv|webm|flv)",  # Videos
            r"https?://[^\s]+\.(?:mp3|wav|m4a|flac|ogg|aac)",  # Audio
        ]

        all_patterns = youtube_patterns + media_patterns

        for pattern in all_patterns:
            if re.search(pattern, question, re.IGNORECASE):
                return True

        return False

    def _route_processing(self, state: AgentState) -> str:
        """Router function for conditional edges"""
        return state["processing_route"]

    async def process_media_node(self, state: AgentState) -> AgentState:
        """Process all media files and URLs using appropriate universal tools"""

        # Extract media from both uploaded files and URLs in question text
        uploaded_media = state.get("media_files", [])
        question = state.get("question", "")
        extracted_urls = self._extract_media_urls_from_question(question)

        all_media = list(uploaded_media) + extracted_urls

        self.logger.info(
            f"Processing {len(uploaded_media)} uploaded files and {len(extracted_urls)} extracted URLs"
        )

        tool_outputs = []
        tools_used = []
        reasoning_trace = state.get("reasoning_trace", [])

        for media_item in all_media:
            try:
                # Determine tool based on media type
                tool_name, tool = self._determine_media_tool(media_item)

                if tool is None:
                    self.logger.warning(f"No suitable tool for media: {media_item}")
                    reasoning_trace.append(f"Skipped unsupported media: {media_item}")
                    continue

                tools_used.append(tool_name)

                # Execute tool
                result = await self._safe_tool_execution(tool, media_item)
                tool_outputs.append(result)

                reasoning_trace.append(f"Processed {media_item} with {tool.name}")

            except Exception as e:
                self.logger.error(f"Error processing {media_item}: {str(e)}")
                reasoning_trace.append(f"Error processing {media_item}: {str(e)}")

        return {
            **state,
            "tool_outputs": state.get("tool_outputs", []) + tool_outputs,
            "tools_used": list(set(state.get("tools_used", []) + tools_used)),
            "reasoning_trace": reasoning_trace,
        }

    def _extract_media_urls_from_question(self, question: str) -> List[str]:
        """Extract media URLs from question text"""
        import re

        urls = []

        # YouTube URL patterns
        youtube_patterns = [
            r"https?://(?:www\.)?youtube\.com/watch\?v=([a-zA-Z0-9_-]{11})",
            r"https?://(?:www\.)?youtu\.be/([a-zA-Z0-9_-]{11})",
            r"https?://(?:www\.)?youtube\.com/embed/([a-zA-Z0-9_-]{11})",
            r"https?://(?:www\.)?youtube\.com/v/([a-zA-Z0-9_-]{11})",
        ]

        for pattern in youtube_patterns:
            matches = re.finditer(pattern, question)
            for match in matches:
                video_id = match.group(1)
                youtube_url = f"https://www.youtube.com/watch?v={video_id}"
                if youtube_url not in urls:
                    urls.append(youtube_url)
                    self.logger.info(f"Extracted YouTube URL: {youtube_url}")

        # Direct media file URLs (images, videos, audio)
        media_url_patterns = [
            r"https?://[^\s]+\.(?:jpg|jpeg|png|gif|bmp|tiff|webp)",  # Images
            r"https?://[^\s]+\.(?:mp4|avi|mov|mkv|webm|flv)",  # Videos
            r"https?://[^\s]+\.(?:mp3|wav|m4a|flac|ogg|aac)",  # Audio
        ]

        for pattern in media_url_patterns:
            matches = re.finditer(pattern, question, re.IGNORECASE)
            for match in matches:
                media_url = match.group(0)
                if media_url not in urls:
                    urls.append(media_url)
                    self.logger.info(f"Extracted media URL: {media_url}")

        return urls

    def _determine_media_tool(self, media_item: str) -> Tuple[str, Any]:
        """Determine appropriate tool for media item"""

        # Check if it's a YouTube URL
        if any(domain in media_item.lower() for domain in ["youtube.com", "youtu.be"]):
            if "youtube_processor" in self.tools:
                return "youtube_processor", self.tools["youtube_processor"]
            else:
                self.logger.warning(
                    "YouTube URL detected but youtube_processor not available"
                )
                return None, None

        # Check if it's a direct video URL
        video_extensions = [".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv"]
        if any(ext in media_item.lower() for ext in video_extensions):
            if "video_analyzer" in self.tools:
                return "video_analyzer", self.tools["video_analyzer"]

        # Check file extension for local files
        if Path(media_item).exists():
            file_ext = Path(media_item).suffix.lower()

            if file_ext in [".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".webp"]:
                if "image_analyzer" in self.tools:
                    return "image_analyzer", self.tools["image_analyzer"]

            elif file_ext in [".mp3", ".wav", ".m4a", ".flac", ".ogg", ".aac"]:
                if "audio_processor" in self.tools:
                    return "audio_processor", self.tools["audio_processor"]

            elif file_ext in [".mp4", ".avi", ".mov", ".mkv"]:
                if "video_analyzer" in self.tools:
                    return "video_analyzer", self.tools["video_analyzer"]

        # Check if it's an image URL
        if any(
            ext in media_item.lower()
            for ext in [".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".webp"]
        ):
            if "image_analyzer" in self.tools:
                return "image_analyzer", self.tools["image_analyzer"]

        # Check if it's an audio URL
        if any(
            ext in media_item.lower()
            for ext in [".mp3", ".wav", ".m4a", ".flac", ".ogg", ".aac"]
        ):
            if "audio_processor" in self.tools:
                return "audio_processor", self.tools["audio_processor"]

        return None, None

    async def web_research_node(self, state: AgentState) -> AgentState:
        """Conduct web research for current information"""
        self.logger.info("Conducting web research")

        reasoning_trace = state.get("reasoning_trace", [])

        try:
            tool = self.tools["web_researcher"]
            result = await self._safe_tool_execution(tool, state["question"])

            reasoning_trace.append("Conducted web research for current information")

            return {
                **state,
                "tool_outputs": state.get("tool_outputs", []) + [result],
                "tools_used": state.get("tools_used", []) + ["web_researcher"],
                "reasoning_trace": reasoning_trace,
            }

        except Exception as e:
            self.logger.error(f"Web research failed: {str(e)}")
            reasoning_trace.append(f"Web research failed: {str(e)}")

            return {
                **state,
                "reasoning_trace": reasoning_trace,
                "error": f"Web research failed: {str(e)}",
            }

    async def claude_reasoning_node(self, state: AgentState) -> AgentState:
        """Central intelligence node - Claude performs all domain reasoning"""
        self.logger.info("Starting Claude reasoning process")

        try:
            # Build comprehensive context for Claude
            context_prompt = self._build_claude_context(
                question=state["question"],
                tool_outputs=state.get("tool_outputs", []),
                task_id=state["task_id"],
            )

            # Get Claude's reasoning
            response = await self.claude_client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=4000,
                temperature=0.1,  # Low temperature for factual accuracy
                messages=[{"role": "user", "content": context_prompt}],
            )

            reasoning_text = response.content[0].text
            answer = self._extract_answer_from_reasoning(reasoning_text)

            reasoning_trace = state.get("reasoning_trace", [])
            reasoning_trace.append("Claude completed domain-specific reasoning")

            self.logger.info("Claude reasoning completed successfully")

            return {
                **state,
                "claude_reasoning": reasoning_text,
                "final_answer": answer,
                "reasoning_trace": reasoning_trace,
            }

        except Exception as e:
            self.logger.error(f"Claude reasoning failed: {str(e)}")
            return {
                **state,
                "error": f"Claude reasoning failed: {str(e)}",
                "reasoning_trace": state.get("reasoning_trace", [])
                + [f"Claude reasoning failed: {str(e)}"],
            }

    async def synthesize_answer_node(self, state: AgentState) -> AgentState:
        """Synthesize final answer and prepare output"""
        execution_time = time.time() - state.get("start_time", 0)

        # Update performance metrics
        self.performance_metrics["total_queries"] += 1
        if not state.get("error"):
            self.performance_metrics["successful_queries"] += 1

        # Update tool usage stats
        for tool in state.get("tools_used", []):
            self.performance_metrics["tool_usage_stats"][tool] = (
                self.performance_metrics["tool_usage_stats"].get(tool, 0) + 1
            )

        # Update average execution time
        self.performance_metrics["average_execution_time"] = (
            self.performance_metrics["average_execution_time"]
            * (self.performance_metrics["total_queries"] - 1)
            + execution_time
        ) / self.performance_metrics["total_queries"]

        self.logger.info(f"Task {state['task_id']} completed in {execution_time:.2f}s")

        return {
            **state,
            "execution_time": execution_time,
            "reasoning_trace": state.get("reasoning_trace", [])
            + [f"Completed in {execution_time:.2f}s"],
        }

    def _build_claude_context(
        self, question: str, tool_outputs: List[Dict], task_id: str
    ) -> str:
        """Build comprehensive context prompt for Claude reasoning"""

        prompt = f"""You are a highly capable AI agent working on GAIA benchmark task {task_id}.

QUESTION: {question}

AVAILABLE DATA FROM TOOLS:
"""

        if not tool_outputs:
            prompt += (
                "No tool outputs available - use your knowledge to answer directly.\n"
            )
        else:
            for i, output in enumerate(tool_outputs, 1):
                if output.get("success", True):
                    prompt += f"""
Tool {i}: {output.get('tool_name', 'Unknown')}
Raw Output: {output.get('raw_output', 'No output')}
Metadata: {output.get('metadata', {})}
"""
                else:
                    prompt += f"""
Tool {i}: {output.get('tool_name', 'Unknown')} - FAILED
Error: {output.get('error', 'Unknown error')}
"""

        prompt += """
INSTRUCTIONS:
1. Analyze all available data in the context of the question
2. Apply logical reasoning, inference, and domain expertise
3. Cross-reference information for consistency
4. Provide a precise, factual answer that matches expected format
5. If information is insufficient, state limitations clearly
6. Show your step-by-step reasoning process

IMPORTANT: The final answer must be precise and factual. GAIA benchmark requires exact matching.
YOUR FINAL ANSWER should be a number OR as few words as possible OR a comma separated list of
numbers and/or strings.
If you are asked for a number, don’t use comma to write your number neither use units such as $ or percent
sign unless specified otherwise.
If you are asked for a string, don’t use articles, neither abbreviations (e.g. for cities), and write the digits in
plain text unless specified otherwise.
If you are asked for a comma separated list, apply the above rules depending of whether the element to be put
in the list is a number or a string.

Format your response as:
REASONING: [Your detailed step-by-step analysis]
ANSWER: [Final precise answer only]
"""

        return prompt

    def _extract_answer_from_reasoning(self, reasoning_text: str) -> str:
        """Extract the final answer from Claude's reasoning"""
        lines = reasoning_text.split("\n")

        # Look for ANSWER: line
        for line in lines:
            if line.strip().startswith("ANSWER:"):
                return line.split("ANSWER:", 1)[1].strip()

        # Fallback: use last non-empty line
        for line in reversed(lines):
            if line.strip():
                return line.strip()

        return "Unable to determine answer"

    async def _safe_tool_execution(self, tool, input_data: Any) -> Dict[str, Any]:
        """Execute tool with comprehensive error handling and monitoring"""
        start_time = time.time()

        try:
            result = await asyncio.wait_for(
                tool.execute(input_data), timeout=self.config.timeout_seconds
            )

            execution_time = time.time() - start_time
            self.logger.debug(f"Tool {tool.name} completed in {execution_time:.2f}s")

            return {**result, "execution_time": execution_time, "success": True}

        except asyncio.TimeoutError:
            self.logger.error(
                f"Tool {tool.name} timed out after {self.config.timeout_seconds}s"
            )
            return {
                "tool_name": tool.name,
                "error": "timeout",
                "success": False,
                "execution_time": self.config.timeout_seconds,
            }
        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"Tool {tool.name} failed: {str(e)}")
            return {
                "tool_name": tool.name,
                "error": str(e),
                "success": False,
                "execution_time": execution_time,
            }

    async def process_question(
        self, task_id: str, question: str, media_files: List[str] = None
    ) -> Dict[str, Any]:
        """Main entry point for processing GAIA questions"""
        self.logger.info(f"Processing question {task_id}: {question[:100]}...")

        # Initialize state
        initial_state = AgentState(
            task_id=task_id,
            question=question,
            media_files=media_files or [],
            processing_route="",
            tool_outputs=[],
            claude_reasoning="",
            final_answer="",
            reasoning_trace=[],
            tools_used=[],
            error=None,
            execution_time=0.0,
            start_time=time.time(),
        )

        # Execute workflow
        try:
            final_state = await self.graph.ainvoke(initial_state)

            # Format output
            result = {
                "task_id": task_id,
                "model_answer": final_state.get("final_answer", "No answer generated"),
                "reasoning_trace": " -> ".join(final_state.get("reasoning_trace", [])),
                "tools_used": final_state.get("tools_used", []),
                "claude_reasoning_steps": final_state.get("claude_reasoning", "").split(
                    "\n"
                ),
                "execution_time": final_state.get("execution_time", 0.0),
                "success": not bool(final_state.get("error")),
            }

            if final_state.get("error"):
                result["error"] = final_state["error"]

            # Save result
            await self._save_result(result)

            return result

        except Exception as e:
            self.logger.error(f"Workflow execution failed: {str(e)}")
            error_result = {
                "task_id": task_id,
                "model_answer": "Error occurred during processing",
                "error": str(e),
                "success": False,
            }
            await self._save_result(error_result)
            return error_result

    async def _save_result(self, result: Dict[str, Any]):
        """Save result to file for analysis"""
        output_file = Path(self.config.output_dir) / f"{result['task_id']}_result.json"

        async with aiofiles.open(output_file, "w") as f:
            await f.write(json.dumps(result, indent=2))

        self.logger.debug(f"Result saved to {output_file}")

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics"""
        return {
            **self.performance_metrics,
            "success_rate": (
                self.performance_metrics["successful_queries"]
                / max(self.performance_metrics["total_queries"], 1)
            )
            * 100,
        }


# Example usage
async def main():
    """Example usage of the GAIA Agent"""
    config = SystemConfig()

    if not config.validate():
        print("Error: Missing required API keys")
        return

    agent = BasicAgent(config)

    # Example question processing
    result = await agent.process_question(
        task_id="example_005",
        question="Which animal is in the picture?",
        media_files=["cachorro.jpg"],
    )

    print(f"Answer: {result['model_answer']}")
    print(f"Success: {result['success']}")
    print(f"Execution time: {result['execution_time']:.2f}s")

    # Print performance metrics
    metrics = agent.get_performance_metrics()
    print(f"Performance: {metrics}")


if __name__ == "__main__":
    asyncio.run(main())
