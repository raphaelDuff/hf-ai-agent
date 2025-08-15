# tools/image_analyzer.py - Universal Image Analyzer
import asyncio
import base64
import logging
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import json
import time

try:
    import cv2
    import numpy as np
    from PIL import Image, ImageEnhance
    import pytesseract
    import easyocr
except ImportError:
    print("Warning: Some image processing libraries not installed. Install with:")
    print("pip install opencv-python pillow pytesseract easyocr")

from .base_tool import UniversalTool


class UniversalImageAnalyzer(UniversalTool):
    """
    Universal Image Analyzer - Processes any visual content without domain assumptions.

    This tool provides raw visual data extraction that Claude can interpret contextually:
    - Visual description of objects, scenes, text, diagrams
    - Spatial relationship analysis
    - Text extraction (OCR) from images
    - Chart/graph data extraction
    - Color and composition analysis

    Anti-pattern: NO chess-specific, medical-specific, or other domain tools
    Usage: Describes what it sees, lets Claude reason about domain-specific implications
    """

    def __init__(self):
        super().__init__("Universal Image Analyzer")
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

        # Initialize OCR engines
        self.ocr_engines = {}
        self._initialize_ocr_engines()

        # Analysis capabilities
        self.capabilities = [
            "visual_description",
            "text_extraction",
            "spatial_analysis",
            "color_analysis",
            "object_detection",
            "composition_analysis",
        ]

        self.logger.info("Universal Image Analyzer initialized")

    def _initialize_ocr_engines(self):
        """Initialize available OCR engines"""
        try:
            # Test Tesseract
            pytesseract.get_tesseract_version()
            self.ocr_engines["tesseract"] = True
            self.logger.info("Tesseract OCR engine available")
        except:
            self.ocr_engines["tesseract"] = False
            self.logger.warning("Tesseract OCR not available")

        try:
            # Test EasyOCR
            self.easyocr_reader = easyocr.Reader(["en"])
            self.ocr_engines["easyocr"] = True
            self.logger.info("EasyOCR engine available")
        except:
            self.ocr_engines["easyocr"] = False
            self.logger.warning("EasyOCR not available")

    async def execute(self, image_path: str) -> Dict[str, Any]:
        """
        Execute comprehensive image analysis

        Args:
            image_path: Path to image file

        Returns:
            Standardized output with visual analysis data
        """
        start_time = time.time()

        try:
            # Load and validate image
            image = await self._load_image(image_path)
            if image is None:
                return self._error_output("Failed to load image")

            # Perform comprehensive analysis
            analysis_results = {}

            # 1. Basic image properties
            analysis_results["properties"] = await self._analyze_properties(image)

            # 2. Visual description
            analysis_results["visual_description"] = (
                await self._describe_visual_content(image)
            )

            # 3. Text extraction (OCR)
            analysis_results["text_content"] = await self._extract_text(image)

            # 4. Spatial analysis
            analysis_results["spatial_analysis"] = (
                await self._analyze_spatial_relationships(image)
            )

            # 5. Color analysis
            analysis_results["color_analysis"] = await self._analyze_colors(image)

            # 6. Object/region detection
            analysis_results["objects_regions"] = await self._detect_objects_regions(
                image
            )

            # 7. Composition analysis
            analysis_results["composition"] = await self._analyze_composition(image)

            # Compile comprehensive output
            raw_output = self._compile_analysis_output(analysis_results)

            metadata = {
                "image_path": image_path,
                "analysis_time": time.time() - start_time,
                "capabilities_used": list(analysis_results.keys()),
                "ocr_engines_available": self.ocr_engines,
            }

            return self._standardize_output(raw_output, metadata)

        except Exception as e:
            self.logger.error(f"Image analysis failed: {str(e)}")
            return self._error_output(f"Analysis failed: {str(e)}")

    async def _load_image(self, image_path: str) -> Optional[np.ndarray]:
        """Load and validate image file"""
        try:
            if not Path(image_path).exists():
                self.logger.error(f"Image file not found: {image_path}")
                return None

            # Load with OpenCV (BGR format)
            image = cv2.imread(image_path)

            if image is None:
                # Try with PIL as fallback
                pil_image = Image.open(image_path)
                image = np.array(pil_image.convert("RGB"))
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            if image is None:
                self.logger.error(f"Failed to load image: {image_path}")
                return None

            self.logger.debug(f"Loaded image: {image.shape}")
            return image

        except Exception as e:
            self.logger.error(f"Error loading image {image_path}: {str(e)}")
            return None

    async def _analyze_properties(self, image: np.ndarray) -> Dict[str, Any]:
        """Analyze basic image properties"""
        height, width = image.shape[:2]
        channels = image.shape[2] if len(image.shape) > 2 else 1

        return {
            "dimensions": {"width": width, "height": height},
            "channels": channels,
            "total_pixels": width * height,
            "aspect_ratio": round(width / height, 2),
            "orientation": (
                "landscape"
                if width > height
                else "portrait" if height > width else "square"
            ),
        }

    async def _describe_visual_content(self, image: np.ndarray) -> Dict[str, Any]:
        """Generate comprehensive visual description"""

        # Basic scene analysis
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Edge detection for complexity
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size

        # Brightness analysis
        brightness = np.mean(gray)

        # Contrast analysis
        contrast = np.std(gray)

        # Detect major regions/shapes
        contours, _ = cv2.findContours(
            edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        major_shapes = self._classify_contours(contours)

        description = {
            "overall_brightness": (
                "bright"
                if brightness > 170
                else "dark" if brightness < 85 else "moderate"
            ),
            "contrast_level": (
                "high" if contrast > 50 else "low" if contrast < 20 else "moderate"
            ),
            "edge_density": edge_density,
            "complexity": (
                "high"
                if edge_density > 0.1
                else "low" if edge_density < 0.03 else "moderate"
            ),
            "major_shapes_detected": len(contours),
            "shape_classifications": major_shapes,
            "dominant_patterns": self._detect_patterns(gray),
        }

        return description

    def _classify_contours(self, contours: List) -> Dict[str, int]:
        """Classify detected contours by shape"""
        shape_counts = {"rectangular": 0, "circular": 0, "triangular": 0, "complex": 0}

        for contour in contours:
            if cv2.contourArea(contour) < 100:  # Skip very small contours
                continue

            # Approximate contour
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)

            vertices = len(approx)

            if vertices == 3:
                shape_counts["triangular"] += 1
            elif vertices == 4:
                shape_counts["rectangular"] += 1
            elif vertices > 8:
                # Check if roughly circular
                area = cv2.contourArea(contour)
                perimeter = cv2.arcLength(contour, True)
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter * perimeter)
                    if circularity > 0.7:
                        shape_counts["circular"] += 1
                    else:
                        shape_counts["complex"] += 1
                else:
                    shape_counts["complex"] += 1
            else:
                shape_counts["complex"] += 1

        return shape_counts

    def _detect_patterns(self, gray_image: np.ndarray) -> List[str]:
        """Detect common visual patterns"""
        patterns = []

        # Line detection
        lines = cv2.HoughLines(
            cv2.Canny(gray_image, 50, 150), 1, np.pi / 180, threshold=100
        )
        if lines is not None and len(lines) > 10:
            patterns.append("grid_like_structure")

        # Check for text-like regions (high horizontal edge density)
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
        horizontal_lines = cv2.morphologyEx(
            cv2.Canny(gray_image, 50, 150), cv2.MORPH_CLOSE, horizontal_kernel
        )
        if np.sum(horizontal_lines > 0) / horizontal_lines.size > 0.02:
            patterns.append("text_like_regions")

        # Check for chart-like patterns (axes, data points)
        # Simple heuristic: look for strong vertical and horizontal lines
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 25))
        vertical_lines = cv2.morphologyEx(
            cv2.Canny(gray_image, 50, 150), cv2.MORPH_CLOSE, vertical_kernel
        )

        if np.sum(horizontal_lines > 0) > 1000 and np.sum(vertical_lines > 0) > 1000:
            patterns.append("chart_like_structure")

        return patterns

    async def _extract_text(self, image: np.ndarray) -> Dict[str, Any]:
        """Extract text using available OCR engines"""
        text_results = {
            "engines_used": [],
            "extracted_text": "",
            "confidence_scores": [],
        }

        # Convert BGR to RGB for OCR engines
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_image)

        # Try Tesseract OCR
        if self.ocr_engines.get("tesseract", False):
            try:
                tesseract_text = pytesseract.image_to_string(pil_image)
                if tesseract_text.strip():
                    text_results["engines_used"].append("tesseract")
                    text_results[
                        "extracted_text"
                    ] += f"[Tesseract] {tesseract_text.strip()}\n"

                    # Get confidence data
                    data = pytesseract.image_to_data(
                        pil_image, output_type=pytesseract.Output.DICT
                    )
                    confidences = [int(conf) for conf in data["conf"] if int(conf) > 0]
                    if confidences:
                        text_results["confidence_scores"].append(
                            {
                                "engine": "tesseract",
                                "average_confidence": np.mean(confidences),
                                "min_confidence": min(confidences),
                                "max_confidence": max(confidences),
                            }
                        )
            except Exception as e:
                self.logger.warning(f"Tesseract OCR failed: {str(e)}")

        # Try EasyOCR
        if self.ocr_engines.get("easyocr", False):
            try:
                easyocr_results = self.easyocr_reader.readtext(np.array(pil_image))
                if easyocr_results:
                    text_results["engines_used"].append("easyocr")
                    easyocr_text = " ".join([result[1] for result in easyocr_results])
                    text_results["extracted_text"] += f"[EasyOCR] {easyocr_text}\n"

                    confidences = [result[2] for result in easyocr_results]
                    text_results["confidence_scores"].append(
                        {
                            "engine": "easyocr",
                            "average_confidence": np.mean(confidences),
                            "min_confidence": min(confidences),
                            "max_confidence": max(confidences),
                        }
                    )
            except Exception as e:
                self.logger.warning(f"EasyOCR failed: {str(e)}")

        # Clean up text
        text_results["extracted_text"] = text_results["extracted_text"].strip()
        text_results["has_text"] = bool(text_results["extracted_text"])

        return text_results

    async def _analyze_spatial_relationships(self, image: np.ndarray) -> Dict[str, Any]:
        """Analyze spatial relationships and layout"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape

        # Divide image into quadrants for regional analysis
        quadrants = {
            "top_left": gray[0 : height // 2, 0 : width // 2],
            "top_right": gray[0 : height // 2, width // 2 : width],
            "bottom_left": gray[height // 2 : height, 0 : width // 2],
            "bottom_right": gray[height // 2 : height, width // 2 : width],
        }

        quadrant_analysis = {}
        for quad_name, quad_image in quadrants.items():
            # Analyze each quadrant
            edges = cv2.Canny(quad_image, 50, 150)
            edge_density = np.sum(edges > 0) / edges.size
            brightness = np.mean(quad_image)

            quadrant_analysis[quad_name] = {
                "edge_density": edge_density,
                "brightness": brightness,
                "activity_level": "high" if edge_density > 0.05 else "low",
            }

        # Center region analysis
        center_x, center_y = width // 2, height // 2
        center_region = gray[
            center_y - 50 : center_y + 50, center_x - 50 : center_x + 50
        ]
        center_activity = (
            np.sum(cv2.Canny(center_region, 50, 150) > 0) / center_region.size
            if center_region.size > 0
            else 0
        )

        return {
            "quadrant_analysis": quadrant_analysis,
            "center_focus": center_activity > 0.05,
            "dominant_region": max(
                quadrant_analysis.keys(),
                key=lambda k: quadrant_analysis[k]["edge_density"],
            ),
            "layout_symmetry": self._calculate_symmetry(gray),
        }

    def _calculate_symmetry(self, gray_image: np.ndarray) -> Dict[str, float]:
        """Calculate horizontal and vertical symmetry"""
        height, width = gray_image.shape

        # Horizontal symmetry (top vs bottom)
        top_half = gray_image[0 : height // 2, :]
        bottom_half = np.flipud(gray_image[height // 2 : height, :])

        # Resize to match if needed
        min_height = min(top_half.shape[0], bottom_half.shape[0])
        top_half = top_half[:min_height, :]
        bottom_half = bottom_half[:min_height, :]

        horizontal_diff = np.mean(
            np.abs(top_half.astype(float) - bottom_half.astype(float))
        )
        horizontal_symmetry = max(0, 1 - horizontal_diff / 255)

        # Vertical symmetry (left vs right)
        left_half = gray_image[:, 0 : width // 2]
        right_half = np.fliplr(gray_image[:, width // 2 : width])

        min_width = min(left_half.shape[1], right_half.shape[1])
        left_half = left_half[:, :min_width]
        right_half = right_half[:, :min_width]

        vertical_diff = np.mean(
            np.abs(left_half.astype(float) - right_half.astype(float))
        )
        vertical_symmetry = max(0, 1 - vertical_diff / 255)

        return {
            "horizontal_symmetry": round(horizontal_symmetry, 3),
            "vertical_symmetry": round(vertical_symmetry, 3),
        }

    async def _analyze_colors(self, image: np.ndarray) -> Dict[str, Any]:
        """Analyze color composition and dominant colors"""
        # Convert to different color spaces for analysis
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Color statistics
        color_stats = {}
        for i, channel in enumerate(["Red", "Green", "Blue"]):
            channel_data = rgb_image[:, :, i]
            color_stats[channel.lower()] = {
                "mean": float(np.mean(channel_data)),
                "std": float(np.std(channel_data)),
                "min": int(np.min(channel_data)),
                "max": int(np.max(channel_data)),
            }

        # Dominant colors using k-means clustering
        dominant_colors = self._find_dominant_colors(rgb_image, k=5)

        # Color temperature estimation
        avg_r, avg_g, avg_b = [color_stats[c]["mean"] for c in ["red", "green", "blue"]]
        color_temperature = (
            "warm" if avg_r > avg_b else "cool" if avg_b > avg_r else "neutral"
        )

        # Saturation analysis
        saturation = hsv_image[:, :, 1]
        avg_saturation = np.mean(saturation)
        saturation_level = (
            "high"
            if avg_saturation > 150
            else "low" if avg_saturation < 50 else "moderate"
        )

        return {
            "color_statistics": color_stats,
            "dominant_colors": dominant_colors,
            "color_temperature": color_temperature,
            "saturation_level": saturation_level,
            "average_saturation": float(avg_saturation),
            "color_diversity": float(
                np.std([color_stats[c]["mean"] for c in ["red", "green", "blue"]])
            ),
        }

    def _find_dominant_colors(self, rgb_image: np.ndarray, k: int = 5) -> List[Dict]:
        """Find dominant colors using k-means clustering"""
        try:
            from sklearn.cluster import KMeans

            # Reshape image to list of pixels
            pixels = rgb_image.reshape(-1, 3)

            # Subsample for performance
            if len(pixels) > 10000:
                indices = np.random.choice(len(pixels), 10000, replace=False)
                pixels = pixels[indices]

            # Perform k-means clustering
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(pixels)

            # Get dominant colors and their percentages
            colors = kmeans.cluster_centers_.astype(int)
            labels = kmeans.labels_

            dominant_colors = []
            for i, color in enumerate(colors):
                percentage = np.sum(labels == i) / len(labels) * 100
                dominant_colors.append(
                    {
                        "rgb": color.tolist(),
                        "hex": f"#{color[0]:02x}{color[1]:02x}{color[2]:02x}",
                        "percentage": round(percentage, 2),
                    }
                )

            # Sort by percentage
            dominant_colors.sort(key=lambda x: x["percentage"], reverse=True)
            return dominant_colors

        except ImportError:
            # Fallback without sklearn
            return [{"rgb": [128, 128, 128], "hex": "#808080", "percentage": 100.0}]
        except Exception as e:
            self.logger.warning(f"Dominant color analysis failed: {str(e)}")
            return []

    async def _detect_objects_regions(self, image: np.ndarray) -> Dict[str, Any]:
        """Detect objects and regions of interest"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Edge-based region detection
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(
            edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # Filter significant contours
        significant_contours = [c for c in contours if cv2.contourArea(c) > 500]

        regions = []
        for i, contour in enumerate(significant_contours[:10]):  # Limit to top 10
            x, y, w, h = cv2.boundingRect(contour)
            area = cv2.contourArea(contour)

            regions.append(
                {
                    "region_id": i,
                    "bounding_box": {
                        "x": int(x),
                        "y": int(y),
                        "width": int(w),
                        "height": int(h),
                    },
                    "area": float(area),
                    "relative_size": area / (image.shape[0] * image.shape[1]),
                    "aspect_ratio": round(w / h, 2) if h > 0 else 0,
                }
            )

        # Corner detection for feature points
        corners = cv2.goodFeaturesToTrack(
            gray, maxCorners=100, qualityLevel=0.01, minDistance=10
        )
        corner_count = len(corners) if corners is not None else 0

        return {
            "significant_regions": regions,
            "total_regions_detected": len(significant_contours),
            "corner_features": corner_count,
            "feature_density": (
                corner_count / (image.shape[0] * image.shape[1])
                if image.size > 0
                else 0
            ),
        }

    async def _analyze_composition(self, image: np.ndarray) -> Dict[str, Any]:
        """Analyze image composition and layout"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape

        # Rule of thirds analysis
        third_h, third_w = height // 3, width // 3
        thirds_grid = {
            "top_left": gray[0:third_h, 0:third_w],
            "top_center": gray[0:third_h, third_w : 2 * third_w],
            "top_right": gray[0:third_h, 2 * third_w : width],
            "middle_left": gray[third_h : 2 * third_h, 0:third_w],
            "center": gray[third_h : 2 * third_h, third_w : 2 * third_w],
            "middle_right": gray[third_h : 2 * third_h, 2 * third_w : width],
            "bottom_left": gray[2 * third_h : height, 0:third_w],
            "bottom_center": gray[2 * third_h : height, third_w : 2 * third_w],
            "bottom_right": gray[2 * third_h : height, 2 * third_w : width],
        }

        # Analyze activity in each third
        thirds_activity = {}
        for region_name, region in thirds_grid.items():
            if region.size > 0:
                edges = cv2.Canny(region, 50, 150)
                activity = np.sum(edges > 0) / edges.size
                thirds_activity[region_name] = activity

        # Find most active region
        most_active_region = (
            max(thirds_activity.keys(), key=lambda k: thirds_activity[k])
            if thirds_activity
            else "center"
        )

        # Overall composition metrics
        total_edges = cv2.Canny(gray, 50, 150)
        edge_distribution = np.std(
            [thirds_activity.get(region, 0) for region in thirds_grid.keys()]
        )

        return {
            "rule_of_thirds_analysis": thirds_activity,
            "focal_point": most_active_region,
            "composition_balance": (
                "balanced" if edge_distribution < 0.02 else "unbalanced"
            ),
            "edge_distribution_variance": float(edge_distribution),
            "overall_complexity": float(np.sum(total_edges > 0) / total_edges.size),
        }

    def _compile_analysis_output(self, analysis_results: Dict[str, Any]) -> str:
        """Compile all analysis results into a comprehensive description"""

        output_lines = ["=== UNIVERSAL IMAGE ANALYSIS ===\n"]

        # Image properties
        props = analysis_results.get("properties", {})
        output_lines.append(f"IMAGE PROPERTIES:")
        output_lines.append(
            f"- Dimensions: {props.get('dimensions', {}).get('width', 'Unknown')} x {props.get('dimensions', {}).get('height', 'Unknown')}"
        )
        output_lines.append(f"- Orientation: {props.get('orientation', 'Unknown')}")
        output_lines.append(f"- Aspect ratio: {props.get('aspect_ratio', 'Unknown')}")

        # Visual description
        visual = analysis_results.get("visual_description", {})
        output_lines.append(f"\nVISUAL CHARACTERISTICS:")
        output_lines.append(
            f"- Overall brightness: {visual.get('overall_brightness', 'Unknown')}"
        )
        output_lines.append(
            f"- Contrast level: {visual.get('contrast_level', 'Unknown')}"
        )
        output_lines.append(
            f"- Visual complexity: {visual.get('complexity', 'Unknown')}"
        )
        output_lines.append(
            f"- Major shapes detected: {visual.get('major_shapes_detected', 0)}"
        )
        if visual.get("shape_classifications"):
            for shape, count in visual["shape_classifications"].items():
                if count > 0:
                    output_lines.append(f"  - {shape}: {count}")
        if visual.get("dominant_patterns"):
            output_lines.append(
                f"- Detected patterns: {', '.join(visual['dominant_patterns'])}"
            )

        # Text content
        text = analysis_results.get("text_content", {})
        output_lines.append(f"\nTEXT CONTENT:")
        if text.get("has_text"):
            output_lines.append(f"- Text detected: YES")
            output_lines.append(
                f"- OCR engines used: {', '.join(text.get('engines_used', []))}"
            )
            output_lines.append(
                f"- Extracted text: {text.get('extracted_text', 'None')}"
            )
        else:
            output_lines.append(f"- Text detected: NO")

        # Spatial analysis
        spatial = analysis_results.get("spatial_analysis", {})
        output_lines.append(f"\nSPATIAL LAYOUT:")
        output_lines.append(
            f"- Dominant region: {spatial.get('dominant_region', 'Unknown')}"
        )
        output_lines.append(
            f"- Center focus: {'Yes' if spatial.get('center_focus') else 'No'}"
        )
        symmetry = spatial.get("layout_symmetry", {})
        output_lines.append(
            f"- Horizontal symmetry: {symmetry.get('horizontal_symmetry', 'Unknown')}"
        )
        output_lines.append(
            f"- Vertical symmetry: {symmetry.get('vertical_symmetry', 'Unknown')}"
        )

        # Color analysis
        colors = analysis_results.get("color_analysis", {})
        output_lines.append(f"\nCOLOR COMPOSITION:")
        output_lines.append(
            f"- Color temperature: {colors.get('color_temperature', 'Unknown')}"
        )
        output_lines.append(
            f"- Saturation level: {colors.get('saturation_level', 'Unknown')}"
        )
        dominant = colors.get("dominant_colors", [])
        if dominant:
            output_lines.append(f"- Top 3 dominant colors:")
            for color in dominant[:3]:
                output_lines.append(
                    f"  - {color.get('hex', 'Unknown')} ({color.get('percentage', 0):.1f}%)"
                )

        # Objects and regions
        objects = analysis_results.get("objects_regions", {})
        output_lines.append(f"\nOBJECTS AND REGIONS:")
        output_lines.append(
            f"- Significant regions detected: {objects.get('total_regions_detected', 0)}"
        )
        output_lines.append(f"- Feature points: {objects.get('corner_features', 0)}")

        # Composition
        composition = analysis_results.get("composition", {})
        output_lines.append(f"\nCOMPOSITION:")
        output_lines.append(
            f"- Focal point: {composition.get('focal_point', 'Unknown')}"
        )
        output_lines.append(
            f"- Balance: {composition.get('composition_balance', 'Unknown')}"
        )
        output_lines.append(
            f"- Overall complexity: {composition.get('overall_complexity', 0):.3f}"
        )

        return "\n".join(output_lines)

    def _error_output(self, error_message: str) -> Dict[str, Any]:
        """Generate standardized error output"""
        return {
            "tool_name": self.name,
            "raw_output": f"Error: {error_message}",
            "success": False,
            "error": error_message,
            "metadata": {"error_occurred": True},
        }


# Example usage and testing
async def test_image_analyzer():
    """Test the Universal Image Analyzer"""
    analyzer = UniversalImageAnalyzer()

    # Test with a sample image (you would provide actual image path)
    test_image_path = "test_image.jpg"

    if Path(test_image_path).exists():
        result = await analyzer.execute(test_image_path)

        print("Analysis Result:")
        print(f"Success: {result.get('success', False)}")
        print(f"Tool: {result.get('tool_name')}")
        print(
            f"Analysis time: {result.get('metadata', {}).get('analysis_time', 0):.2f}s"
        )
        print("\nRaw Output:")
        print(result.get("raw_output", "No output"))
    else:
        print(f"Test image not found: {test_image_path}")


if __name__ == "__main__":
    asyncio.run(test_image_analyzer())
