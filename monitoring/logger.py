# monitoring/logger.py - Advanced Monitoring and Logging System
import logging
import json
import time
import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, asdict
from pathlib import Path
import threading
from collections import defaultdict, deque
import statistics

import aiofiles
import psutil  # For system metrics


@dataclass
class ExecutionMetric:
    """Individual execution metric"""

    timestamp: float
    task_id: str
    component: str  # 'tool', 'claude', 'workflow'
    operation: str
    duration: float
    success: bool
    input_size: int = 0
    output_size: int = 0
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class SystemHealth:
    """System health snapshot"""

    timestamp: float
    cpu_usage: float
    memory_usage: float
    disk_usage: float
    active_tasks: int
    total_executions: int
    success_rate: float
    average_response_time: float
    error_rate: float


class MetricsCollector:
    """Advanced metrics collection and analysis"""

    def __init__(self, max_history: int = 10000):
        self.metrics: deque[ExecutionMetric] = deque(maxlen=max_history)
        self.system_health_history: deque[SystemHealth] = deque(maxlen=1000)
        self.active_tasks: Dict[str, float] = {}  # task_id -> start_time
        self.lock = threading.Lock()

        # Performance thresholds
        self.thresholds = {
            "max_response_time": 30.0,
            "min_success_rate": 0.8,
            "max_error_rate": 0.2,
            "max_cpu_usage": 0.8,
            "max_memory_usage": 0.8,
        }

        # Alert callbacks
        self.alert_callbacks: List[Callable] = []

    def record_execution(self, metric: ExecutionMetric):
        """Record an execution metric"""
        with self.lock:
            self.metrics.append(metric)

            # Check for performance alerts
            self._check_performance_alerts(metric)

    def start_task(self, task_id: str):
        """Mark task as started"""
        with self.lock:
            self.active_tasks[task_id] = time.time()

    def complete_task(self, task_id: str):
        """Mark task as completed"""
        with self.lock:
            if task_id in self.active_tasks:
                del self.active_tasks[task_id]

    def record_system_health(self):
        """Record current system health snapshot"""
        try:
            # Get system metrics
            cpu_usage = psutil.cpu_percent()
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage("/")

            # Calculate performance metrics from recent executions
            recent_metrics = self._get_recent_metrics(minutes=5)

            total_executions = len(recent_metrics)
            success_rate = sum(1 for m in recent_metrics if m.success) / max(
                total_executions, 1
            )
            avg_response_time = (
                statistics.mean([m.duration for m in recent_metrics])
                if recent_metrics
                else 0
            )
            error_rate = 1 - success_rate

            health = SystemHealth(
                timestamp=time.time(),
                cpu_usage=cpu_usage / 100.0,
                memory_usage=memory.percent / 100.0,
                disk_usage=disk.percent / 100.0,
                active_tasks=len(self.active_tasks),
                total_executions=total_executions,
                success_rate=success_rate,
                average_response_time=avg_response_time,
                error_rate=error_rate,
            )

            with self.lock:
                self.system_health_history.append(health)

            # Check system health alerts
            self._check_system_health_alerts(health)

        except Exception as e:
            logging.error(f"Failed to collect system health metrics: {e}")

    def _get_recent_metrics(self, minutes: int = 5) -> List[ExecutionMetric]:
        """Get metrics from the last N minutes"""
        cutoff_time = time.time() - (minutes * 60)
        with self.lock:
            return [m for m in self.metrics if m.timestamp >= cutoff_time]

    def _check_performance_alerts(self, metric: ExecutionMetric):
        """Check if metric triggers performance alerts"""
        alerts = []

        if metric.duration > self.thresholds["max_response_time"]:
            alerts.append(
                {
                    "type": "slow_execution",
                    "metric": metric,
                    "threshold": self.thresholds["max_response_time"],
                    "actual": metric.duration,
                }
            )

        if not metric.success:
            alerts.append(
                {
                    "type": "execution_failure",
                    "metric": metric,
                    "error": metric.error_message,
                }
            )

        for alert in alerts:
            self._trigger_alert(alert)

    def _check_system_health_alerts(self, health: SystemHealth):
        """Check system health for alerts"""
        alerts = []

        if health.cpu_usage > self.thresholds["max_cpu_usage"]:
            alerts.append(
                {
                    "type": "high_cpu_usage",
                    "actual": health.cpu_usage,
                    "threshold": self.thresholds["max_cpu_usage"],
                }
            )

        if health.memory_usage > self.thresholds["max_memory_usage"]:
            alerts.append(
                {
                    "type": "high_memory_usage",
                    "actual": health.memory_usage,
                    "threshold": self.thresholds["max_memory_usage"],
                }
            )

        if health.success_rate < self.thresholds["min_success_rate"]:
            alerts.append(
                {
                    "type": "low_success_rate",
                    "actual": health.success_rate,
                    "threshold": self.thresholds["min_success_rate"],
                }
            )

        for alert in alerts:
            self._trigger_alert(alert)

    def _trigger_alert(self, alert: Dict[str, Any]):
        """Trigger alert callbacks"""
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                logging.error(f"Alert callback failed: {e}")

    def add_alert_callback(self, callback: Callable):
        """Add alert callback function"""
        self.alert_callbacks.append(callback)

    def get_performance_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get performance summary for the last N hours"""
        cutoff_time = time.time() - (hours * 3600)

        with self.lock:
            recent_metrics = [m for m in self.metrics if m.timestamp >= cutoff_time]
            recent_health = [
                h for h in self.system_health_history if h.timestamp >= cutoff_time
            ]

        if not recent_metrics:
            return {"error": "No metrics available for the specified period"}

        # Calculate performance statistics
        total_executions = len(recent_metrics)
        successful_executions = sum(1 for m in recent_metrics if m.success)
        success_rate = successful_executions / total_executions

        durations = [m.duration for m in recent_metrics]
        avg_duration = statistics.mean(durations)
        median_duration = statistics.median(durations)
        p95_duration = (
            statistics.quantiles(durations, n=20)[18]
            if len(durations) > 20
            else max(durations)
        )

        # Component breakdown
        component_stats = defaultdict(
            lambda: {"count": 0, "success": 0, "total_time": 0}
        )
        for metric in recent_metrics:
            stats = component_stats[metric.component]
            stats["count"] += 1
            if metric.success:
                stats["success"] += 1
            stats["total_time"] += metric.duration

        # Error analysis
        error_types = defaultdict(int)
        for metric in recent_metrics:
            if not metric.success and metric.error_message:
                error_types[metric.error_message] += 1

        # System health summary
        health_summary = {}
        if recent_health:
            health_summary = {
                "avg_cpu_usage": statistics.mean([h.cpu_usage for h in recent_health]),
                "avg_memory_usage": statistics.mean(
                    [h.memory_usage for h in recent_health]
                ),
                "max_active_tasks": max([h.active_tasks for h in recent_health]),
                "avg_active_tasks": statistics.mean(
                    [h.active_tasks for h in recent_health]
                ),
            }

        return {
            "period_hours": hours,
            "total_executions": total_executions,
            "success_rate": success_rate,
            "performance": {
                "average_duration": avg_duration,
                "median_duration": median_duration,
                "p95_duration": p95_duration,
                "fastest_execution": min(durations),
                "slowest_execution": max(durations),
            },
            "component_breakdown": {
                component: {
                    "executions": stats["count"],
                    "success_rate": stats["success"] / stats["count"],
                    "average_duration": stats["total_time"] / stats["count"],
                }
                for component, stats in component_stats.items()
            },
            "error_analysis": dict(error_types),
            "system_health": health_summary,
            "current_active_tasks": len(self.active_tasks),
        }


class AdvancedLogger:
    """Advanced logging system with structured output and performance tracking"""

    def __init__(self, log_dir: str = "logs", enable_metrics: bool = True):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)

        self.metrics_collector = MetricsCollector() if enable_metrics else None

        # Setup structured logging
        self.logger = self._setup_structured_logger()

        # Start background health monitoring
        if enable_metrics:
            self._start_health_monitoring()

    def _setup_structured_logger(self) -> logging.Logger:
        """Setup structured logging with multiple outputs"""
        logger = logging.getLogger("gaia_agent_advanced")
        logger.setLevel(logging.DEBUG)

        # Clear existing handlers
        logger.handlers.clear()

        # Detailed file handler
        detailed_handler = logging.FileHandler(self.log_dir / "detailed.log")
        detailed_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s"
        )
        detailed_handler.setFormatter(detailed_formatter)
        detailed_handler.setLevel(logging.DEBUG)

        # Error file handler
        error_handler = logging.FileHandler(self.log_dir / "errors.log")
        error_formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(message)s - %(pathname)s:%(lineno)d"
        )
        error_handler.setFormatter(error_formatter)
        error_handler.setLevel(logging.ERROR)

        # Performance file handler (JSON format)
        performance_handler = logging.FileHandler(self.log_dir / "performance.jsonl")
        performance_handler.setLevel(logging.INFO)

        # Console handler for important messages
        console_handler = logging.StreamHandler()
        console_formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(message)s"
        )
        console_handler.setFormatter(console_formatter)
        console_handler.setLevel(logging.INFO)

        logger.addHandler(detailed_handler)
        logger.addHandler(error_handler)
        logger.addHandler(performance_handler)
        logger.addHandler(console_handler)

        return logger

    def _start_health_monitoring(self):
        """Start background health monitoring"""

        def health_monitor():
            while True:
                try:
                    if self.metrics_collector:
                        self.metrics_collector.record_system_health()
                    time.sleep(60)  # Record health every minute
                except Exception as e:
                    self.logger.error(f"Health monitoring error: {e}")
                    time.sleep(60)

        monitor_thread = threading.Thread(target=health_monitor, daemon=True)
        monitor_thread.start()

    async def log_execution(
        self,
        task_id: str,
        component: str,
        operation: str,
        duration: float,
        success: bool,
        input_data: Any = None,
        output_data: Any = None,
        error_message: str = None,
        metadata: Dict[str, Any] = None,
    ):
        """Log an execution with structured data"""

        # Calculate sizes
        input_size = len(str(input_data)) if input_data else 0
        output_size = len(str(output_data)) if output_data else 0

        # Create metric
        metric = ExecutionMetric(
            timestamp=time.time(),
            task_id=task_id,
            component=component,
            operation=operation,
            duration=duration,
            success=success,
            input_size=input_size,
            output_size=output_size,
            error_message=error_message,
            metadata=metadata or {},
        )

        # Record metric
        if self.metrics_collector:
            self.metrics_collector.record_execution(metric)

        # Log structured data
        log_data = {
            "timestamp": datetime.fromtimestamp(metric.timestamp).isoformat(),
            "task_id": task_id,
            "component": component,
            "operation": operation,
            "duration": duration,
            "success": success,
            "input_size": input_size,
            "output_size": output_size,
            "throughput": output_size / duration if duration > 0 else 0,
            "metadata": metadata or {},
        }

        if error_message:
            log_data["error"] = error_message

        # Write to performance log
        await self._write_performance_log(log_data)

        # Standard logging
        level = logging.INFO if success else logging.ERROR
        message = f"{component}.{operation} - {duration:.3f}s - {'SUCCESS' if success else 'FAILED'}"
        if error_message:
            message += f" - {error_message}"

        self.logger.log(
            level, message, extra={"task_id": task_id, "structured_data": log_data}
        )

    async def _write_performance_log(self, log_data: Dict[str, Any]):
        """Write performance data to JSON lines file"""
        try:
            performance_file = self.log_dir / "performance.jsonl"
            async with aiofiles.open(performance_file, "a") as f:
                await f.write(json.dumps(log_data) + "\n")
        except Exception as e:
            self.logger.error(f"Failed to write performance log: {e}")

    def log_claude_reasoning(
        self,
        task_id: str,
        question: str,
        tool_outputs: List[Dict],
        reasoning: str,
        answer: str,
        confidence: float = None,
    ):
        """Log Claude reasoning process for analysis"""

        reasoning_data = {
            "timestamp": datetime.now().isoformat(),
            "task_id": task_id,
            "question": question,
            "num_tool_outputs": len(tool_outputs),
            "tool_types": [
                output.get("tool_name", "unknown") for output in tool_outputs
            ],
            "reasoning_length": len(reasoning),
            "answer_length": len(answer),
            "answer": answer,
        }

        if confidence is not None:
            reasoning_data["confidence"] = confidence

        # Analyze reasoning quality
        reasoning_analysis = self._analyze_reasoning_quality(
            reasoning, question, answer
        )
        reasoning_data["quality_metrics"] = reasoning_analysis

        self.logger.info(
            "Claude reasoning completed",
            extra={"task_id": task_id, "reasoning_data": reasoning_data},
        )

    def _analyze_reasoning_quality(
        self, reasoning: str, question: str, answer: str
    ) -> Dict[str, Any]:
        """Analyze quality of Claude's reasoning"""

        # Basic quality metrics
        reasoning_lower = reasoning.lower()
        question_lower = question.lower()

        metrics = {
            "reasoning_length": len(reasoning),
            "has_step_by_step": any(
                phrase in reasoning_lower
                for phrase in [
                    "step 1",
                    "first",
                    "then",
                    "next",
                    "finally",
                    "therefore",
                ]
            ),
            "references_question": any(
                word in reasoning_lower for word in question_lower.split()[:5]
            ),
            "has_explicit_answer": "answer:" in reasoning_lower,
            "reasoning_to_answer_ratio": len(reasoning) / max(len(answer), 1),
            "mentions_tools": any(
                phrase in reasoning_lower
                for phrase in [
                    "image",
                    "analysis",
                    "data",
                    "output",
                    "tool",
                    "information",
                ]
            ),
        }

        # Calculate confidence score based on reasoning patterns
        confidence_indicators = [
            "clearly",
            "definitely",
            "certainly",
            "obvious",
            "evident",
            "shows",
            "indicates",
            "demonstrates",
            "confirms",
        ]
        uncertainty_indicators = [
            "possibly",
            "might",
            "could",
            "unclear",
            "uncertain",
            "difficult to determine",
            "cannot be sure",
        ]

        confidence_count = sum(
            1 for indicator in confidence_indicators if indicator in reasoning_lower
        )
        uncertainty_count = sum(
            1 for indicator in uncertainty_indicators if indicator in reasoning_lower
        )

        metrics["confidence_indicators"] = confidence_count
        metrics["uncertainty_indicators"] = uncertainty_count
        metrics["confidence_score"] = max(
            0, min(1, (confidence_count - uncertainty_count + 2) / 4)
        )

        return metrics

    def get_performance_dashboard(self) -> Dict[str, Any]:
        """Get comprehensive performance dashboard data"""
        if not self.metrics_collector:
            return {"error": "Metrics collection not enabled"}

        # Get performance summaries for different time periods
        dashboard = {
            "last_hour": self.metrics_collector.get_performance_summary(hours=1),
            "last_24_hours": self.metrics_collector.get_performance_summary(hours=24),
            "last_week": self.metrics_collector.get_performance_summary(hours=168),
            "system_status": self._get_current_system_status(),
            "recent_errors": self._get_recent_errors(),
            "performance_trends": self._get_performance_trends(),
        }

        return dashboard

    def _get_current_system_status(self) -> Dict[str, Any]:
        """Get current system status"""
        if (
            not self.metrics_collector
            or not self.metrics_collector.system_health_history
        ):
            return {"status": "unknown"}

        latest_health = self.metrics_collector.system_health_history[-1]

        # Determine overall status
        status = "healthy"
        if (
            latest_health.cpu_usage > 0.8
            or latest_health.memory_usage > 0.8
            or latest_health.success_rate < 0.8
        ):
            status = "degraded"

        if (
            latest_health.cpu_usage > 0.95
            or latest_health.memory_usage > 0.95
            or latest_health.success_rate < 0.5
        ):
            status = "critical"

        return {
            "status": status,
            "cpu_usage": latest_health.cpu_usage,
            "memory_usage": latest_health.memory_usage,
            "active_tasks": latest_health.active_tasks,
            "success_rate": latest_health.success_rate,
            "last_updated": datetime.fromtimestamp(latest_health.timestamp).isoformat(),
        }

    def _get_recent_errors(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent errors"""
        if not self.metrics_collector:
            return []

        recent_errors = []
        with self.metrics_collector.lock:
            for metric in reversed(self.metrics_collector.metrics):
                if not metric.success and len(recent_errors) < limit:
                    recent_errors.append(
                        {
                            "timestamp": datetime.fromtimestamp(
                                metric.timestamp
                            ).isoformat(),
                            "task_id": metric.task_id,
                            "component": metric.component,
                            "operation": metric.operation,
                            "error": metric.error_message,
                            "duration": metric.duration,
                        }
                    )

        return recent_errors

    def _get_performance_trends(self) -> Dict[str, Any]:
        """Calculate performance trends"""
        if not self.metrics_collector:
            return {}

        # Get metrics from last 24 hours in hourly buckets
        now = time.time()
        hourly_stats = {}

        for hour in range(24):
            hour_start = now - (hour + 1) * 3600
            hour_end = now - hour * 3600

            with self.metrics_collector.lock:
                hour_metrics = [
                    m
                    for m in self.metrics_collector.metrics
                    if hour_start <= m.timestamp < hour_end
                ]

            if hour_metrics:
                hourly_stats[hour] = {
                    "executions": len(hour_metrics),
                    "success_rate": sum(1 for m in hour_metrics if m.success)
                    / len(hour_metrics),
                    "avg_duration": statistics.mean([m.duration for m in hour_metrics]),
                }

        return {
            "hourly_execution_count": [
                hourly_stats.get(h, {}).get("executions", 0) for h in range(23, -1, -1)
            ],
            "hourly_success_rate": [
                hourly_stats.get(h, {}).get("success_rate", 0)
                for h in range(23, -1, -1)
            ],
            "hourly_avg_duration": [
                hourly_stats.get(h, {}).get("avg_duration", 0)
                for h in range(23, -1, -1)
            ],
        }

    async def export_performance_report(self, hours: int = 24) -> str:
        """Export detailed performance report"""
        if not self.metrics_collector:
            return "Metrics collection not enabled"

        report_data = {
            "report_generated": datetime.now().isoformat(),
            "period_hours": hours,
            "performance_summary": self.metrics_collector.get_performance_summary(
                hours
            ),
            "system_status": self._get_current_system_status(),
            "recent_errors": self._get_recent_errors(limit=50),
            "performance_trends": self._get_performance_trends(),
        }

        # Write report to file
        report_file = self.log_dir / f"performance_report_{int(time.time())}.json"
        async with aiofiles.open(report_file, "w") as f:
            await f.write(json.dumps(report_data, indent=2))

        return str(report_file)


# Alert system
def setup_alert_system(logger: AdvancedLogger):
    """Setup alert system with various notification channels"""

    def console_alert(alert: Dict[str, Any]):
        """Simple console alert"""
        print(f"🚨 ALERT: {alert['type']} - {alert}")

    def log_alert(alert: Dict[str, Any]):
        """Log alert to file"""
        logger.logger.warning(
            f"Performance Alert: {alert['type']}", extra={"alert_data": alert}
        )

    # Add alert callbacks
    if logger.metrics_collector:
        logger.metrics_collector.add_alert_callback(console_alert)
        logger.metrics_collector.add_alert_callback(log_alert)


# Usage example
async def example_usage():
    """Example usage of the monitoring system"""

    # Setup advanced logging
    logger = AdvancedLogger(log_dir="example_logs")
    setup_alert_system(logger)

    # Simulate some executions
    for i in range(10):
        start_time = time.time()

        # Simulate work
        await asyncio.sleep(0.1)

        # Log execution
        await logger.log_execution(
            task_id=f"test_{i}",
            component="test_tool",
            operation="execute",
            duration=time.time() - start_time,
            success=i % 8 != 0,  # 87.5% success rate
            input_data=f"input_{i}",
            output_data=f"output_{i}",
            error_message="Test error" if i % 8 == 0 else None,
            metadata={"test_iteration": i},
        )

    # Get performance dashboard
    dashboard = logger.get_performance_dashboard()
    print("Performance Dashboard:")
    print(json.dumps(dashboard, indent=2))

    # Export report
    report_file = await logger.export_performance_report()
    print(f"Performance report exported to: {report_file}")


if __name__ == "__main__":
    asyncio.run(example_usage())
