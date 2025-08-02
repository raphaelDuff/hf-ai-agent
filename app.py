import os
import json
import asyncio
from typing import Dict, Any, List, Optional

from langgraph.graph import StateGraph, END
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel, Field

from tools import (
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
    get_all_tools,
)
from utils import parse_gaia_question


class AgentState(BaseModel):
    """Pydantic model for GAIA agent state management"""

    messages: List[Any] = Field(
        default_factory=list, description="List of conversation messages"
    )
    task_id: str = Field(..., description="Unique identifier for the task")
    original_question: str = Field(
        ..., description="The original question to be answered"
    )
    plan: Optional[str] = Field(None, description="Agent's execution plan")
    tool_results: List[str] = Field(
        default_factory=list, description="Results from tool executions"
    )
    reasoning_trace: List[str] = Field(
        default_factory=list, description="Step-by-step reasoning trace"
    )
    needs_tools: bool = Field(False, description="Whether the agent needs to use tools")
    final_answer: Optional[str] = Field(
        None, description="The final answer to the question"
    )

    class Config:
        """Pydantic configuration"""

        arbitrary_types_allowed = True  # Allow LangChain message types
        extra = "forbid"  # Prevent extra fields


class GAIAAgent:
    """LangGraph agent optimized for GAIA benchmark tasks"""

    def __init__(self, api_key: str, model: str = "claude-3-5-sonnet-20241022"):
        self.llm = ChatAnthropic(
            anthropic_api_key=api_key,
            model=model,
            temperature=0.1,  # Lower temperature for more consistent reasoning
            max_tokens=4096,
        )
        self.tools = get_all_tools()
        self.tool_map = {tool.name: tool for tool in self.tools}
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow"""
        workflow = StateGraph(AgentState)

        # Add nodes
        workflow.add_node("planner", self._planning_node)
        workflow.add_node("tool_executor", self._tool_execution_node)
        workflow.add_node("reasoner", self._reasoning_node)
        workflow.add_node("formatter", self._answer_formatter_node)

        # Add edges
        workflow.set_entry_point("planner")

        # Conditional routing from planner
        workflow.add_conditional_edges(
            "planner",
            self._should_use_tools,
            {"use_tools": "tool_executor", "skip_tools": "reasoner"},
        )

        # From tool executor to reasoner
        workflow.add_edge("tool_executor", "reasoner")

        # Conditional routing from reasoner
        workflow.add_conditional_edges(
            "reasoner",
            self._needs_more_tools,
            {"more_tools": "tool_executor", "finalize": "formatter"},
        )

        # From formatter to end
        workflow.add_edge("formatter", END)

        return workflow.compile()

    async def _planning_node(self, state: AgentState) -> AgentState:
        """Analyze the task and create an execution plan"""

        # Get GAIA question analysis
        question_analysis = parse_gaia_question(state.original_question)

        # Build tool descriptions
        tool_descriptions = []
        for tool in self.tools:
            tool_descriptions.append(f"- {tool.name}: {tool.description}")

        system_prompt = """You are a planning agent for solving complex real-world tasks optimized for the GAIA benchmark.

Question Analysis and available tools will be provided. Your job is to:
1. Analyze what type of problem this is
2. Determine which tools are most likely needed
3. Create a step-by-step approach to solve it
4. Specify what format the final answer should be in (number, string, or list)

Provide a clear, actionable plan that will lead to a precise answer."""

        user_prompt = f"""Task: {state.original_question}

Question Analysis:
- Recommended tools: {question_analysis['recommended_tools']}
- Expected answer type: {question_analysis['expected_answer_type']}
- Complexity: {question_analysis['complexity']}

Available tools:
{chr(10).join(tool_descriptions)}

Create a detailed execution plan for this task."""

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]
        response = await self.llm.ainvoke(messages)

        # Determine if tools are needed based on the plan and analysis
        plan_content = response.content.lower()
        needs_tools = len(question_analysis["recommended_tools"]) > 0 or any(
            keyword in plan_content
            for keyword in [
                "search",
                "calculate",
                "compute",
                "find",
                "look up",
                "research",
                "file",
                "data",
                "information",
                "internet",
                "web",
                "read",
                "image",
                "analyze",
            ]
        )

        state.plan = response.content
        state.needs_tools = needs_tools
        state.reasoning_trace.append(f"Planning: {response.content}")

        return state

    async def _tool_execution_node(self, state: AgentState) -> AgentState:
        """Execute necessary tools based on the plan"""

        # Build tool information for the prompt
        tool_info = []
        for tool in self.tools:
            tool_info.append(f"- {tool.name}: {tool.description}")

        system_prompt = """You are a tool execution agent. Based on the plan and previous results, determine what specific tool should be used next and with what parameters.

Respond in this exact format:
TOOL: tool_name
ARGS: parameter_value

Or respond with "NO_TOOL" if no tool is needed.

Examples:
TOOL: calculator
ARGS: 15 + 25 * 2

TOOL: web_search  
ARGS: population of Tokyo 2024

TOOL: data_processor
ARGS: 1,2,3,4,5
OPERATION: sum"""

        user_prompt = f"""Based on this plan: {state.plan}

Original question: {state.original_question}

Previous tool results: {state.tool_results}

Available tools:
{chr(10).join(tool_info)}

What specific tool should be used next and with what parameters?"""

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]
        response = await self.llm.ainvoke(messages)

        tool_response = response.content.strip()

        if "NO_TOOL" not in tool_response.upper():
            try:
                # Parse tool call
                lines = tool_response.split("\n")
                tool_name = None
                args = None
                operation = None

                for line in lines:
                    if line.startswith("TOOL:"):
                        tool_name = line.replace("TOOL:", "").strip()
                    elif line.startswith("ARGS:"):
                        args = line.replace("ARGS:", "").strip()
                    elif line.startswith("OPERATION:"):
                        operation = line.replace("OPERATION:", "").strip()

                if tool_name and tool_name in self.tool_map:
                    tool = self.tool_map[tool_name]

                    # Execute tool using correct sync/async invocation methods
                    try:
                        if tool_name == "web_search":
                            # ASYNC tool - use ainvoke
                            result = await tool.ainvoke({"query": args})
                        elif tool_name == "calculator":
                            # SYNC tool - use invoke
                            result = tool.invoke({"expression": args})
                        elif tool_name == "file_reader":
                            # SYNC tool - use invoke
                            result = tool.invoke({"filepath": args})
                        elif tool_name == "data_processor":
                            # SYNC tool - use invoke
                            if operation:
                                result = tool.invoke(
                                    {"data": args, "operation": operation}
                                )
                            else:
                                # Default operation if not specified
                                result = tool.invoke(
                                    {"data": args, "operation": "count"}
                                )
                        elif tool_name == "image_reader":
                            # SYNC tool - use invoke
                            result = tool.invoke({"filepath": args})
                        elif tool_name == "image_analyzer":
                            # ASYNC tool - use ainvoke
                            if operation:
                                result = await tool.ainvoke(
                                    {"filepath": args, "question": operation}
                                )
                            else:
                                # Default question if not specified
                                result = await tool.ainvoke(
                                    {
                                        "filepath": args,
                                        "question": "Describe what you see in this image",
                                    }
                                )
                        elif tool_name == "current_time":
                            # SYNC tool - use invoke
                            result = tool.invoke({})
                        elif tool_name == "extract_numbers":
                            # SYNC tool - use invoke
                            result = tool.invoke({"text": args})
                        elif tool_name == "extract_urls":
                            # SYNC tool - use invoke
                            result = tool.invoke({"text": args})
                        elif tool_name == "text_analyzer":
                            # SYNC tool - use invoke
                            analysis_type = operation if operation else "summary"
                            result = tool.invoke(
                                {"text": args, "analysis_type": analysis_type}
                            )
                        else:
                            result = "Tool execution method not implemented"

                        state.tool_results.append(f"{tool_name}({args}): {result}")

                    except Exception as tool_error:
                        error_msg = (
                            f"Tool {tool_name} execution error: {str(tool_error)}"
                        )
                        state.tool_results.append(error_msg)
                        print(f"Debug - {error_msg}")  # Debug output
                else:
                    state.tool_results.append(f"Unknown tool: {tool_name}")

            except Exception as e:
                state.tool_results.append(f"Tool execution error: {str(e)}")

        state.reasoning_trace.append(f"Tool execution: {tool_response}")
        return state

    async def _reasoning_node(self, state: AgentState) -> AgentState:
        """Process information and reason about the answer"""

        system_prompt = """You are a reasoning agent. Based on the plan and tool results, reason through the problem step by step.

Consider:
1. What information you have
2. What information might be missing  
3. How to arrive at the correct answer
4. Whether the answer should be a number, string, or list

End your response with either:
- "NEED_MORE_TOOLS: [explanation]" if you need additional information
- "READY_FOR_ANSWER: [your reasoning and conclusion]" """

        user_prompt = f"""You are solving this task: {state.original_question}

Plan: {state.plan}

Tool results: {state.tool_results}

Based on all available information, reason through the problem step by step."""

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]
        response = await self.llm.ainvoke(messages)

        reasoning_content = response.content
        state.reasoning_trace.append(f"Reasoning: {reasoning_content}")

        # Determine if more tools are needed
        if "NEED_MORE_TOOLS" in reasoning_content:
            state.needs_tools = True
        else:
            state.needs_tools = False
            # Extract the reasoning for final answer formatting
            if "READY_FOR_ANSWER:" in reasoning_content:
                state.final_answer = reasoning_content.split("READY_FOR_ANSWER:")[
                    1
                ].strip()

        return state

    async def _answer_formatter_node(self, state: AgentState) -> AgentState:
        """Format the final answer according to GAIA requirements"""

        system_prompt = """You are an answer formatter for the GAIA benchmark. You must provide ONLY the final answer in the exact format required:

- If it's a number: provide just the number (no commas, no units like $ or %)
- If it's a string: no articles, no abbreviations, digits in plain text
- If it's a list: comma separated, following above rules for each element

Examples:
- For "What is 2+2?": answer "4"
- For "What city?": answer "Paris" (not "the city of Paris" or "Paris, France")
- For "List three colors": answer "red, blue, green"

Provide ONLY the answer, nothing else."""

        user_prompt = f"""Original question: {state.original_question}

Reasoning and conclusion: {state.final_answer or state.reasoning_trace[-1]}

Provide the final answer in GAIA format:"""

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]
        response = await self.llm.ainvoke(messages)

        # Clean the answer
        final_answer = response.content.strip()

        # Remove any remaining formatting artifacts
        if "FINAL ANSWER:" in final_answer:
            final_answer = final_answer.split("FINAL ANSWER:")[-1].strip()

        state.final_answer = final_answer
        return state

    def _should_use_tools(self, state: AgentState) -> str:
        """Determine if tools should be used"""
        return "use_tools" if state.needs_tools else "skip_tools"

    def _needs_more_tools(self, state: AgentState) -> str:
        """Determine if more tools are needed"""
        return "more_tools" if state.needs_tools else "finalize"

    async def process_task(self, task_id: str, question: str) -> Dict[str, Any]:
        """Process a single GAIA task"""

        initial_state = AgentState(task_id=task_id, original_question=question)

        try:
            # Run the graph
            final_state = await self.graph.ainvoke(initial_state)

            return {
                "task_id": task_id,
                "model_answer": final_state.final_answer,
                "reasoning_trace": " -> ".join(final_state.reasoning_trace),
            }

        except Exception as e:
            return {
                "task_id": task_id,
                "model_answer": "Error processing task",
                "reasoning_trace": f"Error: {str(e)}",
            }


async def main():
    """Main execution function"""

    # Initialize agent
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY environment variable not set")

    agent = GAIAAgent(api_key)

    # Example tasks (replace with actual GAIA tasks)
    tasks = [
        {"task_id": "task_1", "question": "What is 15% of 240 plus 33?"},
        {
            "task_id": "task_2",
            "question": "Find the current population of Tokyo, Japan",
        },
        {
            "task_id": "task_3",
            "question": "Extract all numbers from this text: 'I bought 25 apples for $12.50 and 8 oranges for $3.25'",
        },
        {"task_id": "task_4", "question": "Who is the current CEO of Microsoft?"},
        {
            "task_id": "task_5",
            "question": "Calculate the square root of 144 and multiply by 7",
        },
    ]

    results = []

    for task in tasks:
        print(f"Processing {task['task_id']}...")
        result = await agent.process_task(task["task_id"], task["question"])
        results.append(result)
        print(f"Answer: {result['model_answer']}")
        print("---")

    # Save results in JSONL format
    with open("gaia_results.jsonl", "w") as f:
        for result in results:
            f.write(json.dumps(result) + "\n")

    print(f"Processed {len(results)} tasks. Results saved to gaia_results.jsonl")


if __name__ == "__main__":
    asyncio.run(main())
