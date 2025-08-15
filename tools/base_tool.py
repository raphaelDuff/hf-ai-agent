# tools/base_tool.py - Base class for all Universal Tools
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import asyncio
import time
import logging
from dataclasses import dataclass
from enum import Enum


class ToolStatus(Enum):
    """Tool execution status"""

    SUCCESS = "success"
    FAILURE = "failure"
    TIMEOUT = "timeout"
    PARTIAL = "partial"


@dataclass
class ToolMetrics:
    """Metrics tracking for tool performance"""

    total_executions: int = 0
    successful_executions: int = 0
    failed_executions: int = 0
    total_execution_time: float = 0.0
    average_execution_time: float = 0.0
    last_execution_time: float = 0.0

    def update(self, execution_time: float, success: bool):
        """Update metrics after tool execution"""
        self.total_executions += 1
        self.last_execution_time = execution_time
        self.total_execution_time += execution_time

        if success:
            self.successful_executions += 1
        else:
            self.failed_executions += 1

        self.average_execution_time = self.total_execution_time / self.total_executions

    @property
    def success_rate(self) -> float:
        """Calculate success rate as percentage"""
        if self.total_executions == 0:
            return 0.0
        return (self.successful_executions / self.total_executions) * 100

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary"""
        return {
            "total_executions": self.total_executions,
            "successful_executions": self.successful_executions,
            "failed_executions": self.failed_executions,
            "success_rate": round(self.success_rate, 2),
            "average_execution_time": round(self.average_execution_time, 3),
            "last_execution_time": round(self.last_execution_time, 3),
            "total_execution_time": round(self.total_execution_time, 2),
        }


class UniversalTool(ABC):
    """
    Base class for all Universal Tools in the GAIA Agent system.

    Design Principles:
    1. Single Responsibility: Each tool has one clear purpose
    2. Domain Agnostic: No domain-specific logic embedded in tools
    3. Consistent Interface: All tools follow the same input/output pattern
    4. Observable: All tools provide metrics and logging
    5. Resilient: Built-in error handling and timeout management

    Anti-Patterns:
    - NO chess-specific, medical-specific, or other domain tools
    - NO complex business logic or domain reasoning
    - NO multiple unrelated capabilities in one tool

    Usage Pattern:
    - Tool extracts raw data from input
    - Claude performs all domain-specific reasoning
    - Tool provides standardized output format
    """

    def __init__(self, name: str, timeout_seconds: int = 30):
        self.name = name
        self.timeout_seconds = timeout_seconds
        self.metrics = ToolMetrics()
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

        # Tool capabilities - subclasses should override
        self.capabilities = []

        # Tool configuration
        self.config = {}

        self.logger.info(f"Initialized {self.name}")

    @abstractmethod
    async def execute(self, input_data: Any) -> Dict[str, Any]:
        """
        Execute the tool's primary function.

        Args:
            input_data: Input data specific to the tool

        Returns:
            Standardized output dictionary with:
            - tool_name: Name of the tool
            - raw_output: Raw data/analysis from the tool
            - metadata: Additional information about execution
            - success: Boolean indicating if execution succeeded
            - execution_time: Time taken for execution
        """
        pass

    async def safe_execute(self, input_data: Any) -> Dict[str, Any]:
        """
        Execute tool with comprehensive error handling, timeout, and metrics.

        This is the main entry point that should be used by the agent system.
        """
        start_time = time.time()

        try:
            self.logger.debug(f"Starting execution of {self.name}")

            # Execute with timeout
            result = await asyncio.wait_for(
                self.execute(input_data), timeout=self.timeout_seconds
            )

            execution_time = time.time() - start_time

            # Ensure result has required fields
            result = self._ensure_standard_format(result, execution_time, True)

            # Update metrics
            self.metrics.update(execution_time, True)

            self.logger.info(
                f"{self.name} completed successfully in {execution_time:.2f}s"
            )
            return result

        except asyncio.TimeoutError:
            execution_time = time.time() - start_time
            self.logger.error(f"{self.name} timed out after {self.timeout_seconds}s")

            result = self._create_error_result(
                "timeout",
                f"Tool execution timed out after {self.timeout_seconds} seconds",
                execution_time,
            )

            self.metrics.update(execution_time, False)
            return result

        except Exception as e:
            execution_time = time.time() - start_time
            error_msg = str(e)
            self.logger.error(f"{self.name} failed: {error_msg}")

            result = self._create_error_result(
                "execution_error", error_msg, execution_time
            )
            self.metrics.update(execution_time, False)
            return result

    def _ensure_standard_format(
        self, result: Dict[str, Any], execution_time: float, success: bool
    ) -> Dict[str, Any]:
        """Ensure result follows standard format"""
        standard_result = {
            "tool_name": self.name,
            "raw_output": result.get("raw_output", ""),
            "metadata": result.get("metadata", {}),
            "success": success,
            "execution_time": execution_time,
            "timestamp": time.time(),
        }

        # Add any additional fields from the original result
        for key, value in result.items():
            if key not in standard_result:
                standard_result[key] = value

        # Ensure metadata includes tool info
        standard_result["metadata"].update(
            {
                "tool_capabilities": self.capabilities,
                "tool_version": getattr(self, "version", "1.0"),
                "execution_id": f"{self.name}_{int(time.time())}",
            }
        )

        return standard_result

    def _create_error_result(
        self, error_type: str, error_message: str, execution_time: float
    ) -> Dict[str, Any]:
        """Create standardized error result"""
        return {
            "tool_name": self.name,
            "raw_output": f"Error ({error_type}): {error_message}",
            "metadata": {
                "error_type": error_type,
                "error_message": error_message,
                "tool_capabilities": self.capabilities,
                "execution_id": f"{self.name}_error_{int(time.time())}",
            },
            "success": False,
            "execution_time": execution_time,
            "timestamp": time.time(),
            "error": error_message,
        }

    def _standardize_output(
        self, raw_output: Any, metadata: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Helper method for subclasses to create standardized output.

        This method should be called by subclasses in their execute() method
        to ensure consistent output format.
        """
        return {
            "tool_name": self.name,
            "raw_output": raw_output,
            "metadata": metadata or {},
            "success": True,
        }

    def get_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for this tool"""
        return {
            "tool_name": self.name,
            "metrics": self.metrics.to_dict(),
            "capabilities": self.capabilities,
            "timeout_seconds": self.timeout_seconds,
        }

    def reset_metrics(self):
        """Reset performance metrics"""
        self.metrics = ToolMetrics()
        self.logger.info(f"Reset metrics for {self.name}")

    def configure(self, config: Dict[str, Any]):
        """Configure tool with runtime parameters"""
        self.config.update(config)
        self.logger.debug(f"Updated configuration for {self.name}: {config}")

    def validate_input(self, input_data: Any) -> bool:
        """
        Validate input data format.

        Subclasses should override this method to implement
        input validation specific to their requirements.
        """
        return input_data is not None

    def get_health_status(self) -> Dict[str, Any]:
        """Get tool health status and diagnostics"""
        return {
            "tool_name": self.name,
            "status": "healthy" if self.metrics.success_rate >= 80 else "degraded",
            "success_rate": self.metrics.success_rate,
            "recent_execution_time": self.metrics.last_execution_time,
            "capabilities": self.capabilities,
            "configuration": self.config,
        }

    def __str__(self) -> str:
        return f"{self.name} (executions: {self.metrics.total_executions}, success_rate: {self.metrics.success_rate:.1f}%)"

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}', timeout={self.timeout_seconds}s)"


class ToolRegistry:
    """
    Registry for managing Universal Tools.

    Provides centralized tool management, health monitoring,
    and metrics aggregation across all tools.
    """

    def __init__(self):
        self.tools: Dict[str, UniversalTool] = {}
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    def register_tool(self, tool: UniversalTool):
        """Register a tool in the registry"""
        self.tools[tool.name] = tool
        self.logger.info(f"Registered tool: {tool.name}")

    def get_tool(self, name: str) -> Optional[UniversalTool]:
        """Get a tool by name"""
        return self.tools.get(name)

    def get_all_tools(self) -> Dict[str, UniversalTool]:
        """Get all registered tools"""
        return self.tools.copy()

    def get_tool_metrics(self) -> Dict[str, Any]:
        """Get metrics for all tools"""
        return {name: tool.get_metrics() for name, tool in self.tools.items()}

    def get_system_health(self) -> Dict[str, Any]:
        """Get overall system health status"""
        total_executions = sum(
            tool.metrics.total_executions for tool in self.tools.values()
        )
        total_successful = sum(
            tool.metrics.successful_executions for tool in self.tools.values()
        )

        overall_success_rate = (
            (total_successful / total_executions * 100) if total_executions > 0 else 0
        )

        tool_health = {
            name: tool.get_health_status() for name, tool in self.tools.items()
        }

        return {
            "total_tools": len(self.tools),
            "total_executions": total_executions,
            "overall_success_rate": round(overall_success_rate, 2),
            "tool_health": tool_health,
            "timestamp": time.time(),
        }

    def reset_all_metrics(self):
        """Reset metrics for all tools"""
        for tool in self.tools.values():
            tool.reset_metrics()
        self.logger.info("Reset metrics for all tools")


# Example implementation of a simple universal tool
class ExampleUniversalTool(UniversalTool):
    """Example implementation showing the Universal Tool pattern"""

    def __init__(self):
        super().__init__("Example Universal Tool")
        self.capabilities = ["example_processing", "data_extraction"]

    async def execute(self, input_data: Any) -> Dict[str, Any]:
        """Example implementation"""

        # Validate input
        if not self.validate_input(input_data):
            raise ValueError("Invalid input data")

        # Simulate processing
        await asyncio.sleep(0.1)

        # Generate raw output (no domain reasoning)
        raw_output = f"Processed: {str(input_data)}"

        # Create metadata
        metadata = {
            "input_type": type(input_data).__name__,
            "input_size": len(str(input_data)),
            "processing_method": "example_processing",
        }

        # Return standardized output
        return self._standardize_output(raw_output, metadata)

    def validate_input(self, input_data: Any) -> bool:
        """Validate that input is not None and convertible to string"""
        try:
            str(input_data)
            return True
        except:
            return False


# Usage example
async def test_base_tool():
    """Test the base tool functionality"""
    tool = ExampleUniversalTool()

    # Test successful execution
    result = await tool.safe_execute("test input")
    print(f"Success: {result['success']}")
    print(f"Output: {result['raw_output']}")
    print(f"Execution time: {result['execution_time']:.3f}s")

    # Test metrics
    metrics = tool.get_metrics()
    print(f"Tool metrics: {metrics}")

    # Test health status
    health = tool.get_health_status()
    print(f"Health status: {health}")


if __name__ == "__main__":
    asyncio.run(test_base_tool())
