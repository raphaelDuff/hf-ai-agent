# tools/video_analyzer.py - Universal Video Analyzer
import asyncio
import logging
import tempfile
import time
import json
import subprocess
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import math

try:
    import cv2
    import numpy as np
    from PIL import Image

    CV2_AVAILABLE = True
except ImportError:
    print("Warning: OpenCV not installed. Install with: pip install opencv-python")
    CV2_AVAILABLE = False

try:
    import librosa
    import soundfile as sf

    AUDIO_LIBS_AVAILABLE = True
except ImportError:
    AUDIO_LIBS_AVAILABLE = False

from .base_tool import UniversalTool


class UniversalVideoAnalyzer(UniversalTool):
    """
    Universal Video Analyzer - Extracts information from video content without domain assumptions.

    This tool provides raw video data extraction that Claude can interpret contextually:
    - Key frame extraction at specified intervals
    - Video-to-text transcription (audio track)
    - Scene change detection
    - Object tracking and counting across frames
    - Video metadata extraction (duration, resolution, etc.)
    - Motion analysis and temporal patterns

    Anti-pattern: NO sports analysis, movie analysis, or content-specific tools
    Usage: Provides raw visual and audio data for Claude to interpret contextually
    """

    def __init__(self):
        super().__init__("Universal Video Analyzer")
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

        self.capabilities = [
            "frame_extraction",
            "audio_extraction",
            "scene_detection",
            "motion_analysis",
            "metadata_extraction",
            "temporal_analysis",
            "object_tracking",
        ]

        # Configuration
        self.config = {
            "max_duration_minutes": 10,  # Maximum video duration to process
            "frame_extraction_interval": 1.0,  # Extract frame every N seconds
            "max_frames": 100,  # Maximum frames to extract
            "scene_change_threshold": 0.3,  # Threshold for scene change detection
            "motion_threshold": 30,  # Motion detection threshold
            "max_objects_to_track": 10,  # Maximum objects to track
            "audio_sample_rate": 16000,  # Audio extraction sample rate
        }

        # Check FFmpeg availability
        self._check_ffmpeg_availability()

        self.logger.info("Universal Video Analyzer initialized")

    def _check_ffmpeg_availability(self):
        """Check if FFmpeg is available for video processing"""
        try:
            subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
            self.ffmpeg_available = True
            self.logger.info("FFmpeg available for video processing")
        except (subprocess.CalledProcessError, FileNotFoundError):
            self.ffmpeg_available = False
            self.logger.warning(
                "FFmpeg not available - some video processing features will be limited"
            )

    async def execute(self, video_path: str) -> Dict[str, Any]:
        """
        Execute comprehensive video analysis

        Args:
            video_path: Path to video file

        Returns:
            Standardized output with video analysis data
        """
        start_time = time.time()

        try:
            # Validate video file
            if not Path(video_path).exists():
                return self._error_output(f"Video file not found: {video_path}")

            if not CV2_AVAILABLE:
                return self._error_output("OpenCV not available for video processing")

            # Load video and get basic properties
            video_properties = await self._analyze_video_properties(video_path)
            if not video_properties["success"]:
                return self._error_output(
                    f"Failed to load video: {video_properties['error']}"
                )

            # Check duration limit
            duration = video_properties["duration_seconds"]
            max_duration = self.config["max_duration_minutes"] * 60
            if duration > max_duration:
                return self._error_output(
                    f"Video too long: {duration:.1f}s (max: {max_duration}s)"
                )

            # Perform comprehensive analysis
            analysis_results = {}

            # 1. Video properties and metadata
            analysis_results["properties"] = video_properties

            # 2. Frame extraction and analysis
            analysis_results["frame_analysis"] = await self._extract_and_analyze_frames(
                video_path
            )

            # 3. Audio extraction and analysis
            analysis_results["audio_analysis"] = await self._extract_and_analyze_audio(
                video_path
            )

            # 4. Scene change detection
            analysis_results["scene_analysis"] = await self._detect_scene_changes(
                video_path
            )

            # 5. Motion analysis
            analysis_results["motion_analysis"] = await self._analyze_motion_patterns(
                video_path
            )

            # 6. Object tracking (if applicable)
            analysis_results["object_tracking"] = (
                await self._track_objects_across_frames(video_path)
            )

            # 7. Temporal pattern analysis
            analysis_results["temporal_patterns"] = (
                await self._analyze_temporal_patterns(analysis_results)
            )

            # Compile comprehensive output
            raw_output = self._compile_analysis_output(analysis_results)

            metadata = {
                "video_path": video_path,
                "analysis_time": time.time() - start_time,
                "duration_seconds": duration,
                "total_frames": video_properties.get("total_frames", 0),
                "capabilities_used": list(analysis_results.keys()),
                "ffmpeg_available": self.ffmpeg_available,
            }

            return self._standardize_output(raw_output, metadata)

        except Exception as e:
            self.logger.error(f"Video analysis failed: {str(e)}")
            return self._error_output(f"Analysis failed: {str(e)}")

    async def _analyze_video_properties(self, video_path: str) -> Dict[str, Any]:
        """Analyze basic video properties and metadata"""
        try:
            cap = cv2.VideoCapture(video_path)

            if not cap.isOpened():
                return {"success": False, "error": "Cannot open video file"}

            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            duration = frame_count / fps if fps > 0 else 0

            # Get codec information
            fourcc = cap.get(cv2.CAP_PROP_FOURCC)
            codec = "".join([chr((int(fourcc) >> 8 * i) & 0xFF) for i in range(4)])

            cap.release()

            return {
                "success": True,
                "duration_seconds": duration,
                "fps": fps,
                "total_frames": frame_count,
                "width": width,
                "height": height,
                "resolution": f"{width}x{height}",
                "codec": codec.strip(),
                "aspect_ratio": round(width / height, 2) if height > 0 else 0,
                "estimated_bitrate": self._estimate_bitrate(video_path, duration),
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    def _estimate_bitrate(self, video_path: str, duration: float) -> Optional[float]:
        """Estimate video bitrate from file size and duration"""
        try:
            file_size = Path(video_path).stat().st_size
            if duration > 0:
                bitrate_bps = (file_size * 8) / duration
                return round(bitrate_bps / 1000, 2)  # Convert to kbps
        except:
            return None

    async def _extract_and_analyze_frames(self, video_path: str) -> Dict[str, Any]:
        """Extract key frames and analyze visual content"""
        try:
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            # Calculate frame extraction interval
            interval_frames = int(fps * self.config["frame_extraction_interval"])
            max_frames = min(
                self.config["max_frames"], frame_count // max(interval_frames, 1)
            )

            extracted_frames = []
            frame_analyses = []

            for i in range(0, frame_count, interval_frames):
                if len(extracted_frames) >= max_frames:
                    break

                cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                ret, frame = cap.read()

                if not ret:
                    continue

                timestamp = i / fps

                # Analyze frame
                frame_analysis = self._analyze_single_frame(frame, timestamp, i)
                frame_analyses.append(frame_analysis)

                # Store frame info (not the actual frame data to save memory)
                extracted_frames.append(
                    {
                        "frame_number": i,
                        "timestamp": timestamp,
                        "analysis": frame_analysis,
                    }
                )

            cap.release()

            # Aggregate frame analyses
            overall_analysis = self._aggregate_frame_analyses(frame_analyses)

            return {
                "total_frames_extracted": len(extracted_frames),
                "extraction_interval_seconds": self.config["frame_extraction_interval"],
                "frames": extracted_frames,
                "overall_visual_analysis": overall_analysis,
            }

        except Exception as e:
            self.logger.error(f"Frame extraction failed: {str(e)}")
            return {"error": str(e)}

    def _analyze_single_frame(
        self, frame: np.ndarray, timestamp: float, frame_number: int
    ) -> Dict[str, Any]:
        """Analyze a single video frame"""
        try:
            height, width = frame.shape[:2]

            # Convert to different color spaces for analysis
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            # Basic visual characteristics
            brightness = np.mean(gray)
            contrast = np.std(gray)

            # Edge detection for complexity
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / edges.size

            # Color analysis
            color_analysis = self._analyze_frame_colors(frame, hsv)

            # Object/shape detection
            contours, _ = cv2.findContours(
                edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            significant_objects = len([c for c in contours if cv2.contourArea(c) > 500])

            # Motion estimation (compared to previous frame if available)
            motion_score = self._estimate_frame_motion(frame, gray)

            return {
                "timestamp": timestamp,
                "frame_number": frame_number,
                "brightness": float(brightness),
                "contrast": float(contrast),
                "edge_density": float(edge_density),
                "visual_complexity": (
                    "high"
                    if edge_density > 0.1
                    else "low" if edge_density < 0.03 else "moderate"
                ),
                "significant_objects": significant_objects,
                "color_analysis": color_analysis,
                "motion_score": motion_score,
                "dominant_regions": self._identify_dominant_regions(gray),
            }

        except Exception as e:
            return {"error": str(e)}

    def _analyze_frame_colors(
        self, frame: np.ndarray, hsv: np.ndarray
    ) -> Dict[str, Any]:
        """Analyze color composition of a frame"""
        try:
            # Dominant colors (simplified)
            colors_bgr = frame.reshape(-1, 3)

            # Sample colors for performance
            if len(colors_bgr) > 10000:
                indices = np.random.choice(len(colors_bgr), 10000, replace=False)
                colors_bgr = colors_bgr[indices]

            # Basic color statistics
            mean_color = np.mean(colors_bgr, axis=0)

            # Saturation analysis
            saturation = hsv[:, :, 1]
            avg_saturation = np.mean(saturation)

            # Color temperature estimation
            b, g, r = mean_color
            color_temperature = "warm" if r > b else "cool" if b > r else "neutral"

            return {
                "mean_color_bgr": mean_color.tolist(),
                "color_temperature": color_temperature,
                "average_saturation": float(avg_saturation),
                "saturation_level": (
                    "high"
                    if avg_saturation > 150
                    else "low" if avg_saturation < 50 else "moderate"
                ),
            }

        except Exception as e:
            return {"error": str(e)}

    def _estimate_frame_motion(self, frame: np.ndarray, gray: np.ndarray) -> float:
        """Estimate motion in frame (placeholder for optical flow)"""
        # Simplified motion estimation based on edge variance
        # In a full implementation, this would use optical flow
        edges = cv2.Canny(gray, 50, 150)
        motion_score = np.var(edges.astype(float))
        return float(motion_score / 1000)  # Normalize

    def _identify_dominant_regions(self, gray: np.ndarray) -> Dict[str, Any]:
        """Identify dominant regions in the frame"""
        height, width = gray.shape

        # Divide into quadrants
        quadrants = {
            "top_left": gray[0 : height // 2, 0 : width // 2],
            "top_right": gray[0 : height // 2, width // 2 : width],
            "bottom_left": gray[height // 2 : height, 0 : width // 2],
            "bottom_right": gray[height // 2 : height, width // 2 : width],
        }

        activity_scores = {}
        for quad_name, quad_region in quadrants.items():
            edges = cv2.Canny(quad_region, 50, 150)
            activity_scores[quad_name] = float(np.sum(edges > 0) / edges.size)

        most_active = max(activity_scores.keys(), key=lambda k: activity_scores[k])

        return {"quadrant_activity": activity_scores, "most_active_region": most_active}

    def _aggregate_frame_analyses(
        self, frame_analyses: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Aggregate analyses across all frames"""
        if not frame_analyses:
            return {"error": "No frames to analyze"}

        valid_analyses = [f for f in frame_analyses if "error" not in f]

        if not valid_analyses:
            return {"error": "No valid frame analyses"}

        # Calculate averages and trends
        brightness_values = [f["brightness"] for f in valid_analyses]
        contrast_values = [f["contrast"] for f in valid_analyses]
        complexity_values = [f["edge_density"] for f in valid_analyses]
        motion_values = [f["motion_score"] for f in valid_analyses]

        # Trend analysis
        brightness_trend = self._calculate_trend(brightness_values)
        motion_trend = self._calculate_trend(motion_values)

        return {
            "total_frames_analyzed": len(valid_analyses),
            "average_brightness": float(np.mean(brightness_values)),
            "brightness_trend": brightness_trend,
            "average_contrast": float(np.mean(contrast_values)),
            "average_complexity": float(np.mean(complexity_values)),
            "average_motion": float(np.mean(motion_values)),
            "motion_trend": motion_trend,
            "visual_consistency": "high" if np.std(brightness_values) < 20 else "low",
            "complexity_classification": self._classify_overall_complexity(
                complexity_values
            ),
        }

    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend direction in a series of values"""
        if len(values) < 3:
            return "insufficient_data"

        # Simple linear trend
        x = np.arange(len(values))
        slope = np.polyfit(x, values, 1)[0]

        if abs(slope) < 0.1:
            return "stable"
        elif slope > 0:
            return "increasing"
        else:
            return "decreasing"

    def _classify_overall_complexity(self, complexity_values: List[float]) -> str:
        """Classify overall visual complexity"""
        avg_complexity = np.mean(complexity_values)

        if avg_complexity > 0.1:
            return "highly_complex"
        elif avg_complexity > 0.05:
            return "moderately_complex"
        else:
            return "simple"

    async def _extract_and_analyze_audio(self, video_path: str) -> Dict[str, Any]:
        """Extract and analyze audio track from video"""
        try:
            if not self.ffmpeg_available:
                return {"error": "FFmpeg not available for audio extraction"}

            # Extract audio using FFmpeg
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
                temp_audio_path = temp_audio.name

            ffmpeg_cmd = [
                "ffmpeg",
                "-i",
                video_path,
                "-vn",  # No video
                "-acodec",
                "pcm_s16le",  # Audio codec
                "-ar",
                str(self.config["audio_sample_rate"]),  # Sample rate
                "-ac",
                "1",  # Mono
                "-y",
                temp_audio_path,
            ]

            process = await asyncio.create_subprocess_exec(
                *ffmpeg_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            stdout, stderr = await process.communicate()

            if process.returncode != 0:
                Path(temp_audio_path).unlink(missing_ok=True)
                return {"error": f"Audio extraction failed: {stderr.decode()}"}

            # Analyze extracted audio
            audio_analysis = await self._analyze_extracted_audio(temp_audio_path)

            # Clean up temporary file
            Path(temp_audio_path).unlink(missing_ok=True)

            return audio_analysis

        except Exception as e:
            self.logger.error(f"Audio extraction failed: {str(e)}")
            return {"error": str(e)}

    async def _analyze_extracted_audio(self, audio_path: str) -> Dict[str, Any]:
        """Analyze extracted audio file"""
        try:
            if not AUDIO_LIBS_AVAILABLE:
                return {
                    "audio_present": True,
                    "detailed_analysis": "Audio libraries not available",
                }

            # Load audio
            audio_data, sample_rate = librosa.load(
                audio_path, sr=self.config["audio_sample_rate"]
            )

            if len(audio_data) == 0:
                return {"audio_present": False}

            # Basic audio analysis
            duration = len(audio_data) / sample_rate
            rms_energy = np.sqrt(np.mean(audio_data**2))

            # Silence detection
            silence_threshold = np.percentile(np.abs(audio_data), 20)
            silence_ratio = np.sum(np.abs(audio_data) < silence_threshold) / len(
                audio_data
            )

            # Speech/music classification (very basic)
            zero_crossing_rate = librosa.feature.zero_crossing_rate(audio_data)[0]
            avg_zcr = np.mean(zero_crossing_rate)

            content_type = "speech" if avg_zcr > 0.1 else "music_or_other"

            return {
                "audio_present": True,
                "duration_seconds": float(duration),
                "average_energy": float(rms_energy),
                "silence_ratio": float(silence_ratio),
                "estimated_content_type": content_type,
                "has_significant_audio": rms_energy > 0.01,
                "audio_quality": "good" if rms_energy > 0.1 else "low",
            }

        except Exception as e:
            return {"error": str(e)}

    async def _detect_scene_changes(self, video_path: str) -> Dict[str, Any]:
        """Detect scene changes in video"""
        try:
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)

            scene_changes = []
            prev_hist = None
            frame_number = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Calculate color histogram for scene change detection
                hist = cv2.calcHist(
                    [frame], [0, 1, 2], None, [50, 50, 50], [0, 256, 0, 256, 0, 256]
                )

                if prev_hist is not None:
                    # Compare histograms
                    correlation = cv2.compareHist(hist, prev_hist, cv2.HISTCMP_CORREL)

                    if correlation < (1 - self.config["scene_change_threshold"]):
                        timestamp = frame_number / fps
                        scene_changes.append(
                            {
                                "frame_number": frame_number,
                                "timestamp": timestamp,
                                "correlation_score": float(correlation),
                            }
                        )

                prev_hist = hist
                frame_number += 1

                # Limit processing for performance
                if frame_number % 30 == 0:  # Check every 30 frames
                    await asyncio.sleep(0)  # Yield control

            cap.release()

            return {
                "scene_changes_detected": len(scene_changes),
                "scene_changes": scene_changes,
                "average_scene_duration": self._calculate_average_scene_duration(
                    scene_changes, frame_number / fps
                ),
                "scene_stability": (
                    "high"
                    if len(scene_changes) < 5
                    else "moderate" if len(scene_changes) < 15 else "low"
                ),
            }

        except Exception as e:
            return {"error": str(e)}

    def _calculate_average_scene_duration(
        self, scene_changes: List[Dict], total_duration: float
    ) -> float:
        """Calculate average duration between scene changes"""
        if len(scene_changes) <= 1:
            return total_duration

        durations = []
        prev_timestamp = 0

        for change in scene_changes:
            durations.append(change["timestamp"] - prev_timestamp)
            prev_timestamp = change["timestamp"]

        # Add final scene duration
        durations.append(total_duration - prev_timestamp)

        return float(np.mean(durations))

    async def _analyze_motion_patterns(self, video_path: str) -> Dict[str, Any]:
        """Analyze motion patterns in video"""
        try:
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)

            motion_scores = []
            prev_gray = None
            frame_number = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                if prev_gray is not None:
                    # Calculate optical flow magnitude
                    flow = cv2.calcOpticalFlowPyrLK(
                        prev_gray,
                        gray,
                        np.array(
                            [[100, 100]], dtype=np.float32
                        ),  # Simple tracking point
                        None,
                    )

                    # Calculate frame difference as motion proxy
                    frame_diff = cv2.absdiff(prev_gray, gray)
                    motion_score = np.mean(frame_diff)

                    timestamp = frame_number / fps
                    motion_scores.append(
                        {"timestamp": timestamp, "motion_score": float(motion_score)}
                    )

                prev_gray = gray
                frame_number += 1

                # Limit processing
                if frame_number % 10 == 0:  # Process every 10th frame
                    await asyncio.sleep(0)

            cap.release()

            if not motion_scores:
                return {"error": "No motion data calculated"}

            # Analyze motion patterns
            scores = [m["motion_score"] for m in motion_scores]

            return {
                "total_motion_measurements": len(motion_scores),
                "average_motion": float(np.mean(scores)),
                "motion_variance": float(np.var(scores)),
                "peak_motion": float(np.max(scores)),
                "motion_classification": self._classify_motion_level(scores),
                "motion_timeline": motion_scores[::10],  # Sample for timeline
            }

        except Exception as e:
            return {"error": str(e)}

    def _classify_motion_level(self, motion_scores: List[float]) -> str:
        """Classify overall motion level"""
        avg_motion = np.mean(motion_scores)

        if avg_motion > 50:
            return "high_motion"
        elif avg_motion > 20:
            return "moderate_motion"
        else:
            return "low_motion"

    async def _track_objects_across_frames(self, video_path: str) -> Dict[str, Any]:
        """Basic object tracking across frames"""
        try:
            # Simplified object tracking using corner detection
            cap = cv2.VideoCapture(video_path)

            # Get first frame for initial object detection
            ret, first_frame = cap.read()
            if not ret:
                return {"error": "Cannot read first frame"}

            gray_first = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)

            # Detect initial features/objects
            corners = cv2.goodFeaturesToTrack(
                gray_first,
                maxCorners=self.config["max_objects_to_track"],
                qualityLevel=0.01,
                minDistance=30,
            )

            if corners is None:
                return {"trackable_objects": 0, "tracking_data": []}

            tracking_data = []
            frame_number = 0
            fps = cap.get(cv2.CAP_PROP_FPS)

            # Reset video to beginning
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

            while frame_number < 100:  # Limit frames for performance
                ret, frame = cap.read()
                if not ret:
                    break

                timestamp = frame_number / fps
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                # Count significant features in current frame
                current_corners = cv2.goodFeaturesToTrack(
                    gray, maxCorners=50, qualityLevel=0.01, minDistance=30
                )

                feature_count = (
                    len(current_corners) if current_corners is not None else 0
                )

                tracking_data.append(
                    {
                        "frame_number": frame_number,
                        "timestamp": timestamp,
                        "feature_count": feature_count,
                    }
                )

                frame_number += 1

                if frame_number % 20 == 0:  # Process every 20th frame
                    await asyncio.sleep(0)

            cap.release()

            # Analyze tracking consistency
            feature_counts = [t["feature_count"] for t in tracking_data]
            tracking_consistency = 1 - (
                np.std(feature_counts) / max(np.mean(feature_counts), 1)
            )

            return {
                "initial_trackable_objects": len(corners),
                "tracking_frames": len(tracking_data),
                "tracking_consistency": float(tracking_consistency),
                "average_features_per_frame": float(np.mean(feature_counts)),
                "tracking_stability": (
                    "high"
                    if tracking_consistency > 0.8
                    else "moderate" if tracking_consistency > 0.5 else "low"
                ),
            }

        except Exception as e:
            return {"error": str(e)}

    async def _analyze_temporal_patterns(
        self, analysis_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze temporal patterns across the video"""
        try:
            # Extract temporal data from various analyses
            temporal_data = {}

            # Frame analysis temporal patterns
            frame_analysis = analysis_results.get("frame_analysis", {})
            if "frames" in frame_analysis:
                timestamps = [f["timestamp"] for f in frame_analysis["frames"]]
                temporal_data["frame_timestamps"] = timestamps
                temporal_data["frame_interval_consistency"] = (
                    self._check_interval_consistency(timestamps)
                )

            # Scene change temporal patterns
            scene_analysis = analysis_results.get("scene_analysis", {})
            if "scene_changes" in scene_analysis:
                scene_changes = scene_analysis["scene_changes"]
                if scene_changes:
                    scene_timestamps = [s["timestamp"] for s in scene_changes]
                    temporal_data["scene_change_pattern"] = (
                        self._analyze_scene_change_pattern(scene_timestamps)
                    )

            # Motion temporal patterns
            motion_analysis = analysis_results.get("motion_analysis", {})
            if "motion_timeline" in motion_analysis:
                motion_timeline = motion_analysis["motion_timeline"]
                temporal_data["motion_pattern"] = self._analyze_motion_temporal_pattern(
                    motion_timeline
                )

            # Overall video structure
            video_props = analysis_results.get("properties", {})
            duration = video_props.get("duration_seconds", 0)

            temporal_data["video_structure"] = self._classify_video_structure(
                duration, temporal_data
            )

            return temporal_data

        except Exception as e:
            return {"error": str(e)}

    def _check_interval_consistency(self, timestamps: List[float]) -> str:
        """Check if frame extraction intervals are consistent"""
        if len(timestamps) < 2:
            return "insufficient_data"

        intervals = [
            timestamps[i + 1] - timestamps[i] for i in range(len(timestamps) - 1)
        ]
        interval_std = np.std(intervals)

        return "consistent" if interval_std < 0.1 else "inconsistent"

    def _analyze_scene_change_pattern(
        self, scene_timestamps: List[float]
    ) -> Dict[str, Any]:
        """Analyze pattern in scene changes"""
        if len(scene_timestamps) < 2:
            return {"pattern": "no_changes"}

        intervals = [
            scene_timestamps[i + 1] - scene_timestamps[i]
            for i in range(len(scene_timestamps) - 1)
        ]

        return {
            "pattern": (
                "regular"
                if np.std(intervals) < np.mean(intervals) * 0.5
                else "irregular"
            ),
            "average_scene_length": float(np.mean(intervals)),
            "scene_variance": float(np.var(intervals)),
        }

    def _analyze_motion_temporal_pattern(
        self, motion_timeline: List[Dict]
    ) -> Dict[str, Any]:
        """Analyze temporal pattern in motion"""
        if not motion_timeline:
            return {"pattern": "no_data"}

        motion_scores = [m["motion_score"] for m in motion_timeline]

        # Detect motion peaks
        motion_threshold = np.mean(motion_scores) + np.std(motion_scores)
        peaks = [i for i, score in enumerate(motion_scores) if score > motion_threshold]

        return {
            "motion_peaks": len(peaks),
            "motion_distribution": (
                "concentrated"
                if len(peaks) < len(motion_scores) * 0.3
                else "distributed"
            ),
            "motion_intensity_variance": float(np.var(motion_scores)),
        }

    def _classify_video_structure(
        self, duration: float, temporal_data: Dict[str, Any]
    ) -> str:
        """Classify overall video structure"""
        if duration < 30:
            return "short_clip"
        elif duration < 300:  # 5 minutes
            base_structure = "medium_clip"
        else:
            base_structure = "long_video"

        # Modify based on scene changes
        scene_pattern = temporal_data.get("scene_change_pattern", {})
        if scene_pattern.get("pattern") == "regular":
            return f"{base_structure}_structured"
        else:
            return f"{base_structure}_unstructured"

    def _compile_analysis_output(self, analysis_results: Dict[str, Any]) -> str:
        """Compile all analysis results into a comprehensive description"""

        output_lines = ["=== UNIVERSAL VIDEO ANALYSIS ===\n"]

        # Video properties
        props = analysis_results.get("properties", {})
        output_lines.append("VIDEO PROPERTIES:")
        output_lines.append(
            f"- Duration: {props.get('duration_seconds', 'Unknown'):.1f} seconds"
        )
        output_lines.append(f"- Resolution: {props.get('resolution', 'Unknown')}")
        output_lines.append(f"- Frame rate: {props.get('fps', 'Unknown'):.1f} FPS")
        output_lines.append(f"- Total frames: {props.get('total_frames', 'Unknown')}")
        output_lines.append(f"- Codec: {props.get('codec', 'Unknown')}")

        # Frame analysis
        frame_analysis = analysis_results.get("frame_analysis", {})
        if "error" not in frame_analysis:
            overall_visual = frame_analysis.get("overall_visual_analysis", {})
            output_lines.append("\nVISUAL CONTENT ANALYSIS:")
            output_lines.append(
                f"- Frames analyzed: {frame_analysis.get('total_frames_extracted', 0)}"
            )
            output_lines.append(
                f"- Average brightness: {overall_visual.get('average_brightness', 0):.1f}"
            )
            output_lines.append(
                f"- Visual complexity: {overall_visual.get('complexity_classification', 'Unknown')}"
            )
            output_lines.append(
                f"- Visual consistency: {overall_visual.get('visual_consistency', 'Unknown')}"
            )
            output_lines.append(
                f"- Motion trend: {overall_visual.get('motion_trend', 'Unknown')}"
            )
        else:
            output_lines.append(
                f"\nVISUAL CONTENT ANALYSIS: Failed - {frame_analysis['error']}"
            )

        # Audio analysis
        audio_analysis = analysis_results.get("audio_analysis", {})
        output_lines.append("\nAUDIO ANALYSIS:")
        if "error" not in audio_analysis:
            if audio_analysis.get("audio_present"):
                output_lines.append(f"- Audio present: YES")
                output_lines.append(
                    f"- Duration: {audio_analysis.get('duration_seconds', 0):.1f} seconds"
                )
                output_lines.append(
                    f"- Content type: {audio_analysis.get('estimated_content_type', 'Unknown')}"
                )
                output_lines.append(
                    f"- Audio quality: {audio_analysis.get('audio_quality', 'Unknown')}"
                )
                output_lines.append(
                    f"- Silence ratio: {audio_analysis.get('silence_ratio', 0):.2f}"
                )
            else:
                output_lines.append("- Audio present: NO")
        else:
            output_lines.append(f"- Audio analysis failed: {audio_analysis['error']}")

        # Scene analysis
        scene_analysis = analysis_results.get("scene_analysis", {})
        output_lines.append("\nSCENE ANALYSIS:")
        if "error" not in scene_analysis:
            output_lines.append(
                f"- Scene changes detected: {scene_analysis.get('scene_changes_detected', 0)}"
            )
            output_lines.append(
                f"- Scene stability: {scene_analysis.get('scene_stability', 'Unknown')}"
            )
            output_lines.append(
                f"- Average scene duration: {scene_analysis.get('average_scene_duration', 0):.1f}s"
            )
        else:
            output_lines.append(f"- Scene analysis failed: {scene_analysis['error']}")

        # Motion analysis
        motion_analysis = analysis_results.get("motion_analysis", {})
        output_lines.append("\nMOTION ANALYSIS:")
        if "error" not in motion_analysis:
            output_lines.append(
                f"- Motion classification: {motion_analysis.get('motion_classification', 'Unknown')}"
            )
            output_lines.append(
                f"- Average motion score: {motion_analysis.get('average_motion', 0):.1f}"
            )
            output_lines.append(
                f"- Peak motion: {motion_analysis.get('peak_motion', 0):.1f}"
            )
        else:
            output_lines.append(f"- Motion analysis failed: {motion_analysis['error']}")

        # Object tracking
        tracking = analysis_results.get("object_tracking", {})
        output_lines.append("\nOBJECT TRACKING:")
        if "error" not in tracking:
            output_lines.append(
                f"- Initial trackable objects: {tracking.get('initial_trackable_objects', 0)}"
            )
            output_lines.append(
                f"- Tracking stability: {tracking.get('tracking_stability', 'Unknown')}"
            )
            output_lines.append(
                f"- Average features per frame: {tracking.get('average_features_per_frame', 0):.1f}"
            )
        else:
            output_lines.append(f"- Object tracking failed: {tracking['error']}")

        # Temporal patterns
        temporal = analysis_results.get("temporal_patterns", {})
        output_lines.append("\nTEMPORAL PATTERNS:")
        if "error" not in temporal:
            output_lines.append(
                f"- Video structure: {temporal.get('video_structure', 'Unknown')}"
            )
            scene_pattern = temporal.get("scene_change_pattern", {})
            if scene_pattern:
                output_lines.append(
                    f"- Scene change pattern: {scene_pattern.get('pattern', 'Unknown')}"
                )
            motion_pattern = temporal.get("motion_pattern", {})
            if motion_pattern:
                output_lines.append(
                    f"- Motion distribution: {motion_pattern.get('motion_distribution', 'Unknown')}"
                )
        else:
            output_lines.append(f"- Temporal analysis failed: {temporal['error']}")

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
async def test_video_analyzer():
    """Test the Universal Video Analyzer"""
    analyzer = UniversalVideoAnalyzer()

    # Test with a sample video file (you would provide actual video path)
    test_video_path = "test_video.mp4"

    if Path(test_video_path).exists():
        result = await analyzer.execute(test_video_path)

        print("Analysis Result:")
        print(f"Success: {result.get('success', False)}")
        print(f"Tool: {result.get('tool_name')}")
        print(
            f"Analysis time: {result.get('metadata', {}).get('analysis_time', 0):.2f}s"
        )
        print("\nRaw Output:")
        print(result.get("raw_output", "No output"))
    else:
        print(f"Test video not found: {test_video_path}")


if __name__ == "__main__":
    asyncio.run(test_video_analyzer())
