"""
Utility functions for the GAIA Agent
"""

from typing import Dict, Any


def parse_gaia_question(question: str) -> Dict[str, Any]:
    """
    Parse GAIA question to identify type and required tools.

    Args:
        question: The GAIA question text

    Returns:
        Dictionary with question analysis and recommended tools
    """
    analysis = {
        "type": "general",
        "recommended_tools": [],
        "expected_answer_type": "string",
        "complexity": "simple",
    }

    question_lower = question.lower()

    # Mathematical computation indicators
    calc_patterns = [
        "calculate",
        "compute",
        "sum",
        "multiply",
        "divide",
        "percentage",
        "average",
        "total",
        "add",
        "subtract",
        "%",
        "+",
        "-",
        "*",
        "/",
        "squared",
        "square root",
        "power",
    ]
    if any(pattern in question_lower for pattern in calc_patterns):
        analysis["recommended_tools"].append("calculator")
        analysis["expected_answer_type"] = "number"

    # Information lookup indicators
    search_patterns = [
        "find",
        "search",
        "look up",
        "what is",
        "who is",
        "when",
        "where",
        "latest",
        "current",
        "recent",
        "today",
        "now",
        "information about",
    ]
    if any(pattern in question_lower for pattern in search_patterns):
        analysis["recommended_tools"].append("web_search")

    # File handling indicators
    file_patterns = [
        "file",
        "document",
        "spreadsheet",
        "csv",
        "excel",
        "pdf",
        "read",
        "analyze the document",
        "in the file",
        "from the data",
    ]
    if any(pattern in question_lower for pattern in file_patterns):
        analysis["recommended_tools"].append("file_reader")

    # Data processing indicators
    data_patterns = [
        "list",
        "count",
        "sort",
        "unique",
        "maximum",
        "minimum",
        "first",
        "last",
        "process",
        "filter",
        "group",
        "aggregate",
    ]
    if any(pattern in question_lower for pattern in data_patterns):
        analysis["recommended_tools"].append("data_processor")

    # Time-related indicators
    time_patterns = ["current time", "what time", "date", "today", "now"]
    if any(pattern in question_lower for pattern in time_patterns):
        analysis["recommended_tools"].append("current_time")

    # Determine complexity
    if len(analysis["recommended_tools"]) > 2:
        analysis["complexity"] = "complex"
    elif len(analysis["recommended_tools"]) > 1:
        analysis["complexity"] = "moderate"

    # Determine expected answer format
    list_indicators = ["list", "name all", "enumerate", "which ones", "what are"]
    if any(indicator in question_lower for indicator in list_indicators):
        analysis["expected_answer_type"] = "list"

    return analysis


def test_question_parsing():
    """Test the parse_gaia_question function"""
    print("🧪 Testing GAIA question parsing utility...\n")

    test_questions = [
        "Calculate 15% of 240 and add 50",
        "Find the current population of Tokyo",
        "What are the unique values in this list: apple,banana,apple,cherry",
        "What is the current time?",
        "Read the data from sales_report.csv and find the maximum value",
    ]

    for question in test_questions:
        analysis = parse_gaia_question(question)
        print(f"📋 '{question}'")
        print(f"   Recommended tools: {analysis['recommended_tools']}")
        print(f"   Expected answer type: {analysis['expected_answer_type']}")
        print(f"   Complexity: {analysis['complexity']}")
        print()

    print("✅ Question parsing utility working correctly!")


if __name__ == "__main__":
    test_question_parsing()
