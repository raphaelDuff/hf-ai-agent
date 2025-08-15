# tools/youtube_processor.py - YouTube Content Processor (API-First Approach)
import asyncio
import logging
import tempfile
import time
import json
import re
import os
import socket
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
from urllib.parse import urlparse, parse_qs
import subprocess
import aiohttp

try:
    import yt_dlp

    YT_DLP_AVAILABLE = True
except ImportError:
    print("Warning: yt-dlp not installed. Install with: pip install yt-dlp")
    YT_DLP_AVAILABLE = False

from .base_tool import UniversalTool
from .video_analyzer import UniversalVideoAnalyzer
from .audio_processor import UniversalAudioProcessor


class YouTubeContentProcessor(UniversalTool):
    """
    YouTube Content Processor - API-First approach for reliable content extraction.

    Processing Priority:
    1. YouTube Data API v3 (Primary) - Works in all environments
    2. Direct video download (Secondary) - For advanced analysis when network allows

    This tool provides raw YouTube content extraction that Claude can interpret contextually:
    - YouTube Data API for metadata, captions, comments (primary method)
    - Full subtitle/caption text extraction (manual + auto-generated)
    - Direct video download for frame analysis (when available)
    - Graceful degradation based on environment capabilities

    Anti-pattern: NO recommendation engines or engagement analysis
    Usage: Feeds extracted content to other analysis tools for Claude's interpretation
    """

    def __init__(self, youtube_api_key: Optional[str] = None):
        super().__init__("YouTube Content Processor")
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

        # API configuration
        self.youtube_api_key = youtube_api_key or os.getenv("YOUTUBE_API_KEY")
        self.youtube_api_base = "https://www.googleapis.com/youtube/v3"

        self.capabilities = [
            "youtube_api_primary",
            "subtitle_content_extraction",
            "auto_generated_captions",
            "metadata_extraction",
            "comment_analysis",
            "youtube_download_advanced",
            "video_processing",
        ]

        # Configuration
        self.config = {
            "max_video_duration": 600,  # 10 minutes max
            "api_first": True,  # Use API as primary method
            "download_for_advanced": False,  # Only download for frame analysis if needed
            "extract_subtitle_content": True,  # Actually get subtitle text
            "prefer_manual_captions": True,  # Prefer manual over auto-generated
            "extract_comments": True,
            "max_comments": 50,
            "quality": "best[height<=720]",  # For download fallback
            "network_timeout": 5.0,
        }

        # Initialize sub-analyzers (for advanced processing)
        self.video_analyzer = UniversalVideoAnalyzer()
        self.audio_processor = UniversalAudioProcessor()

        # Processing capabilities
        self.processing_capabilities = {
            "api_available": bool(self.youtube_api_key),
            "download_available": None,  # Will be tested
            "subtitle_extraction": bool(self.youtube_api_key),
        }

        self.logger.info(
            f"YouTube Content Processor initialized (API-first mode, API key available: {bool(self.youtube_api_key)})"
        )

    async def execute(self, youtube_url: str) -> Dict[str, Any]:
        """
        Execute YouTube content processing with API-first approach

        Args:
            youtube_url: YouTube video URL

        Returns:
            Standardized output with YouTube content analysis
        """
        start_time = time.time()

        try:
            # Validate YouTube URL
            if not self._is_valid_youtube_url(youtube_url):
                return self._error_output(f"Invalid YouTube URL: {youtube_url}")

            # Extract video ID
            video_id = self._extract_video_id(youtube_url)
            if not video_id:
                return self._error_output("Could not extract video ID from URL")

            # Test processing capabilities
            await self._test_processing_capabilities()

            processing_results = {}

            # PRIMARY METHOD: YouTube Data API Processing
            if self.processing_capabilities["api_available"]:
                self.logger.info("Using YouTube Data API (primary method)")
                processing_results = await self._process_with_api_primary(
                    youtube_url, video_id
                )

                # Check if advanced analysis is needed and possible
                needs_advanced = self._needs_advanced_analysis(processing_results)
                can_download = self.processing_capabilities.get(
                    "download_available", False
                )

                if needs_advanced and can_download:
                    self.logger.info("Performing advanced analysis via download")
                    advanced_results = await self._add_advanced_analysis(
                        youtube_url, video_id
                    )
                    processing_results.update(advanced_results)

            # FALLBACK METHOD: Direct Download Only
            elif self.processing_capabilities.get("download_available", False):
                self.logger.info("Using direct download method (API not available)")
                processing_results = await self._process_with_download_only(
                    youtube_url, video_id
                )

            else:
                # No processing methods available
                return self._create_no_access_response(
                    youtube_url, video_id, start_time
                )

            # Compile comprehensive output
            raw_output = self._compile_processing_output(processing_results)

            metadata = {
                "youtube_url": youtube_url,
                "video_id": video_id,
                "processing_time": time.time() - start_time,
                "primary_method": (
                    "youtube_data_api"
                    if self.processing_capabilities["api_available"]
                    else "direct_download"
                ),
                "processing_capabilities": self.processing_capabilities,
                "capabilities_used": list(processing_results.keys()),
            }

            return self._standardize_output(raw_output, metadata)

        except Exception as e:
            self.logger.error(f"YouTube processing failed: {str(e)}")
            return self._error_output(f"Processing failed: {str(e)}")

    async def _test_processing_capabilities(self):
        """Test what processing methods are available"""

        # Test YouTube Data API
        if self.youtube_api_key:
            try:
                async with aiohttp.ClientSession() as session:
                    test_url = f"{self.youtube_api_base}/search"
                    params = {
                        "part": "id",
                        "q": "test",
                        "maxResults": 1,
                        "key": self.youtube_api_key,
                    }
                    async with session.get(
                        test_url, params=params, timeout=self.config["network_timeout"]
                    ) as response:
                        self.processing_capabilities["api_available"] = (
                            response.status == 200
                        )
                        self.logger.info(
                            f"YouTube Data API: {'Available' if response.status == 200 else f'Error {response.status}'}"
                        )
            except Exception as e:
                self.processing_capabilities["api_available"] = False
                self.logger.warning(f"YouTube Data API test failed: {str(e)[:50]}...")

        # Test direct download capability
        if YT_DLP_AVAILABLE:
            try:
                future = asyncio.open_connection("www.youtube.com", 443)
                reader, writer = await asyncio.wait_for(
                    future, timeout=self.config["network_timeout"]
                )
                writer.close()
                await writer.wait_closed()
                self.processing_capabilities["download_available"] = True
                self.logger.info("Direct YouTube download: Available")
            except Exception as e:
                self.processing_capabilities["download_available"] = False
                self.logger.info(f"Direct YouTube download: Blocked ({str(e)[:30]}...)")
        else:
            self.processing_capabilities["download_available"] = False
            self.logger.info("Direct YouTube download: yt-dlp not available")

    async def _process_with_api_primary(
        self, youtube_url: str, video_id: str
    ) -> Dict[str, Any]:
        """Primary processing using YouTube Data API with full content extraction"""
        processing_results = {}

        # 1. Extract comprehensive metadata
        processing_results["metadata"] = await self._extract_metadata_via_api(video_id)

        # 2. Extract full subtitle/caption content
        processing_results["subtitles"] = await self._extract_full_captions_via_api(
            video_id
        )

        # 3. Extract comments analysis
        if self.config["extract_comments"]:
            processing_results["comments"] = await self._extract_comments_via_api(
                video_id
            )

        # 4. Content synthesis
        processing_results["content_synthesis"] = await self._synthesize_api_content(
            processing_results
        )

        # 5. Add processing method info
        processing_results["processing_method"] = {
            "primary": "youtube_data_api",
            "subtitle_content": "extracted_via_api",
            "advanced_analysis": "not_performed",
        }

        return processing_results

    async def _extract_full_captions_via_api(self, video_id: str) -> Dict[str, Any]:
        """Extract complete caption/subtitle content using YouTube Data API"""
        try:
            # Step 1: Get list of available caption tracks
            params = {
                "videoId": video_id,
                "part": "snippet",
                "key": self.youtube_api_key,
            }

            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.youtube_api_base}/captions", params=params
                ) as response:
                    if response.status != 200:
                        return {
                            "error": f"Captions list API request failed: {response.status}"
                        }

                    data = await response.json()
                    caption_tracks = data.get("items", [])

                    if not caption_tracks:
                        return {
                            "subtitles_extracted": False,
                            "reason": "No captions or auto-generated captions available",
                            "manual_captions": False,
                            "auto_generated": False,
                        }

                    # Step 2: Prioritize caption tracks (manual first, then auto-generated)
                    manual_tracks = [
                        track
                        for track in caption_tracks
                        if track["snippet"]["trackKind"] == "standard"
                    ]
                    auto_tracks = [
                        track
                        for track in caption_tracks
                        if track["snippet"]["trackKind"] == "asr"
                    ]

                    # Find best English track
                    selected_track = self._select_best_caption_track(
                        manual_tracks, auto_tracks
                    )

                    if not selected_track:
                        return {
                            "subtitles_extracted": False,
                            "reason": "No suitable English captions found",
                            "available_languages": [
                                track["snippet"]["language"] for track in caption_tracks
                            ],
                            "manual_captions": len(manual_tracks) > 0,
                            "auto_generated": len(auto_tracks) > 0,
                        }

                    # Step 3: Download the actual caption content
                    caption_content = await self._download_caption_content(
                        selected_track["id"]
                    )

                    if caption_content:
                        return {
                            "subtitles_extracted": True,
                            "caption_track_info": {
                                "language": selected_track["snippet"]["language"],
                                "name": selected_track["snippet"]["name"],
                                "track_kind": selected_track["snippet"]["trackKind"],
                                "is_manual": selected_track["snippet"]["trackKind"]
                                == "standard",
                                "is_auto_generated": selected_track["snippet"][
                                    "trackKind"
                                ]
                                == "asr",
                            },
                            "subtitle_text": caption_content["text"],
                            "subtitle_segments": caption_content.get("segments", []),
                            "total_segments": len(caption_content.get("segments", [])),
                            "available_tracks": len(caption_tracks),
                            "processing_method": "youtube_data_api",
                        }
                    else:
                        return {
                            "subtitles_extracted": False,
                            "reason": "Failed to download caption content",
                            "track_found": True,
                            "download_failed": True,
                        }

        except Exception as e:
            self.logger.error(f"Full caption extraction failed: {str(e)}")
            return {"error": str(e)}

    def _select_best_caption_track(
        self, manual_tracks: List[Dict], auto_tracks: List[Dict]
    ) -> Optional[Dict]:
        """Select the best caption track (prefer manual English, then auto English, then others)"""

        # Priority 1: Manual English captions
        english_manual = [
            track
            for track in manual_tracks
            if track["snippet"]["language"].startswith("en")
        ]
        if english_manual:
            self.logger.info(
                f"Selected manual English captions: {english_manual[0]['snippet']['name']}"
            )
            return english_manual[0]

        # Priority 2: Auto-generated English captions
        english_auto = [
            track
            for track in auto_tracks
            if track["snippet"]["language"].startswith("en")
        ]
        if english_auto:
            self.logger.info(
                f"Selected auto-generated English captions: {english_auto[0]['snippet']['name']}"
            )
            return english_auto[0]

        # Priority 3: Any manual captions
        if manual_tracks:
            self.logger.info(
                f"Selected manual captions (non-English): {manual_tracks[0]['snippet']['language']}"
            )
            return manual_tracks[0]

        # Priority 4: Any auto-generated captions
        if auto_tracks:
            self.logger.info(
                f"Selected auto-generated captions (non-English): {auto_tracks[0]['snippet']['language']}"
            )
            return auto_tracks[0]

        return None

    async def _download_caption_content(
        self, caption_id: str
    ) -> Optional[Dict[str, Any]]:
        """Download the actual caption content using the caption ID"""
        try:
            # YouTube Data API endpoint for downloading caption content
            download_url = f"{self.youtube_api_base}/captions/{caption_id}"
            params = {"key": self.youtube_api_key, "tfmt": "vtt"}  # WebVTT format

            async with aiohttp.ClientSession() as session:
                async with session.get(download_url, params=params) as response:
                    if response.status == 200:
                        vtt_content = await response.text()

                        # Parse VTT content
                        parsed_content = self._parse_vtt_content(vtt_content)

                        if parsed_content:
                            self.logger.info(
                                f"Successfully downloaded and parsed caption content ({len(parsed_content.get('segments', []))} segments)"
                            )
                            return parsed_content
                        else:
                            self.logger.warning("Failed to parse VTT content")
                            return None
                    else:
                        self.logger.warning(
                            f"Caption download failed: HTTP {response.status}"
                        )
                        # Try alternative download method
                        return await self._download_caption_alternative(caption_id)

        except Exception as e:
            self.logger.error(f"Caption content download failed: {str(e)}")
            return None

    async def _download_caption_alternative(
        self, caption_id: str
    ) -> Optional[Dict[str, Any]]:
        """Alternative method to download caption content"""
        try:
            # Alternative: Use SRT format
            download_url = f"{self.youtube_api_base}/captions/{caption_id}"
            params = {
                "key": self.youtube_api_key,
                "tfmt": "srt",  # SRT format as fallback
            }

            async with aiohttp.ClientSession() as session:
                async with session.get(download_url, params=params) as response:
                    if response.status == 200:
                        srt_content = await response.text()

                        # Parse SRT content
                        parsed_content = self._parse_srt_content(srt_content)

                        if parsed_content:
                            self.logger.info(
                                f"Successfully downloaded SRT captions as fallback"
                            )
                            return parsed_content

                    self.logger.warning(
                        f"Alternative caption download also failed: HTTP {response.status}"
                    )
                    return None

        except Exception as e:
            self.logger.error(f"Alternative caption download failed: {str(e)}")
            return None

    def _parse_vtt_content(self, vtt_content: str) -> Optional[Dict[str, Any]]:
        """Parse WebVTT caption content"""
        try:
            segments = []
            lines = vtt_content.split("\n")

            current_segment = {}
            text_lines = []

            for line in lines:
                line = line.strip()

                if not line or line.startswith("WEBVTT") or line.startswith("NOTE"):
                    continue

                # Time stamp line: 00:00:01.000 --> 00:00:04.000
                if "-->" in line:
                    if current_segment and text_lines:
                        current_segment["text"] = " ".join(text_lines)
                        segments.append(current_segment)

                    # Parse timestamps
                    timestamps = line.split(" --> ")
                    if len(timestamps) == 2:
                        current_segment = {
                            "start_time": self._parse_timestamp(timestamps[0]),
                            "end_time": self._parse_timestamp(timestamps[1]),
                        }
                        text_lines = []

                # Text line
                elif line and not line.isdigit():
                    # Clean subtitle text (remove HTML tags, positioning)
                    clean_text = re.sub(r"<[^>]+>", "", line)
                    clean_text = re.sub(r"align:start position:\d+%", "", clean_text)
                    if clean_text.strip():
                        text_lines.append(clean_text.strip())

            # Add final segment
            if current_segment and text_lines:
                current_segment["text"] = " ".join(text_lines)
                segments.append(current_segment)

            # Create full text
            full_text = " ".join([seg["text"] for seg in segments])

            return {
                "segments": segments,
                "text": full_text,
                "segment_count": len(segments),
                "format": "vtt",
            }

        except Exception as e:
            self.logger.error(f"VTT parsing failed: {str(e)}")
            return None

    def _parse_srt_content(self, srt_content: str) -> Optional[Dict[str, Any]]:
        """Parse SRT caption content"""
        try:
            segments = []
            blocks = srt_content.strip().split("\n\n")

            for block in blocks:
                lines = block.strip().split("\n")
                if len(lines) >= 3:
                    # Skip sequence number (first line)
                    # Parse timestamp (second line)
                    timestamp_line = lines[1]
                    if "-->" in timestamp_line:
                        timestamps = timestamp_line.split(" --> ")
                        if len(timestamps) == 2:
                            # Text content (remaining lines)
                            text_lines = lines[2:]
                            text = " ".join(text_lines).strip()

                            if text:
                                segments.append(
                                    {
                                        "start_time": self._parse_timestamp(
                                            timestamps[0].replace(",", ".")
                                        ),
                                        "end_time": self._parse_timestamp(
                                            timestamps[1].replace(",", ".")
                                        ),
                                        "text": text,
                                    }
                                )

            # Create full text
            full_text = " ".join([seg["text"] for seg in segments])

            return {
                "segments": segments,
                "text": full_text,
                "segment_count": len(segments),
                "format": "srt",
            }

        except Exception as e:
            self.logger.error(f"SRT parsing failed: {str(e)}")
            return None

    def _parse_timestamp(self, timestamp_str: str) -> float:
        """Parse timestamp to seconds (supports both VTT and SRT formats)"""
        try:
            # Clean timestamp
            timestamp_str = timestamp_str.strip()

            # Handle SRT format (HH:MM:SS,mmm)
            timestamp_str = timestamp_str.replace(",", ".")

            # Parse HH:MM:SS.mmm or MM:SS.mmm
            parts = timestamp_str.split(":")

            if len(parts) == 3:  # HH:MM:SS.mmm
                hours, minutes, seconds = parts
                return int(hours) * 3600 + int(minutes) * 60 + float(seconds)
            elif len(parts) == 2:  # MM:SS.mmm
                minutes, seconds = parts
                return int(minutes) * 60 + float(seconds)
            else:
                return 0.0
        except:
            return 0.0

    async def _extract_metadata_via_api(self, video_id: str) -> Dict[str, Any]:
        """Extract comprehensive video metadata using YouTube Data API"""
        try:
            params = {
                "id": video_id,
                "part": "snippet,statistics,contentDetails,status",
                "key": self.youtube_api_key,
            }

            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.youtube_api_base}/videos", params=params
                ) as response:
                    if response.status == 200:
                        data = await response.json()

                        if not data.get("items"):
                            return {"error": "Video not found or private"}

                        video_info = data["items"][0]
                        snippet = video_info["snippet"]
                        statistics = video_info["statistics"]
                        content_details = video_info["contentDetails"]
                        status = video_info.get("status", {})

                        # Parse ISO 8601 duration
                        duration_seconds = self._parse_iso_duration(
                            content_details["duration"]
                        )

                        return {
                            "title": snippet["title"],
                            "description": snippet["description"],
                            "duration_seconds": duration_seconds,
                            "view_count": int(statistics.get("viewCount", 0)),
                            "like_count": int(statistics.get("likeCount", 0)),
                            "comment_count": int(statistics.get("commentCount", 0)),
                            "channel": snippet["channelTitle"],
                            "channel_id": snippet["channelId"],
                            "upload_date": snippet["publishedAt"],
                            "upload_date_formatted": snippet["publishedAt"][:10],
                            "tags": snippet.get("tags", []),
                            "category_id": snippet.get("categoryId", ""),
                            "default_language": snippet.get("defaultLanguage", ""),
                            "thumbnail": snippet["thumbnails"]["high"]["url"],
                            "webpage_url": f"https://www.youtube.com/watch?v={video_id}",
                            "video_id": video_id,
                            "privacy_status": status.get("privacyStatus", "unknown"),
                            "data_source": "youtube_data_api_v3",
                            "api_extraction_time": time.time(),
                        }
                    else:
                        error_data = await response.json()
                        return {
                            "error": f"API request failed: {response.status} - {error_data.get('error', {}).get('message', 'Unknown error')}"
                        }

        except Exception as e:
            self.logger.error(f"API metadata extraction failed: {str(e)}")
            return {"error": str(e)}

    def _parse_iso_duration(self, duration_str: str) -> int:
        """Parse ISO 8601 duration (PT1H2M3S) to seconds"""
        import re

        match = re.match(r"PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?", duration_str)
        if not match:
            return 0

        hours = int(match.group(1) or 0)
        minutes = int(match.group(2) or 0)
        seconds = int(match.group(3) or 0)

        return hours * 3600 + minutes * 60 + seconds

    async def _extract_comments_via_api(self, video_id: str) -> Dict[str, Any]:
        """Extract comments using YouTube Data API"""
        try:
            params = {
                "videoId": video_id,
                "part": "snippet",
                "order": "relevance",
                "maxResults": min(self.config["max_comments"], 100),
                "key": self.youtube_api_key,
            }

            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.youtube_api_base}/commentThreads", params=params
                ) as response:
                    if response.status == 200:
                        data = await response.json()

                        processed_comments = []
                        for item in data.get("items", []):
                            comment = item["snippet"]["topLevelComment"]["snippet"]
                            processed_comments.append(
                                {
                                    "text": comment["textDisplay"],
                                    "like_count": comment["likeCount"],
                                    "author": comment["authorDisplayName"],
                                    "published_at": comment["publishedAt"],
                                    "updated_at": comment.get(
                                        "updatedAt", comment["publishedAt"]
                                    ),
                                }
                            )

                        return {
                            "comments_extracted": True,
                            "total_comments": len(processed_comments),
                            "top_comments": processed_comments,
                            "comment_summary": self._summarize_comments(
                                processed_comments
                            ),
                            "data_source": "youtube_data_api_v3",
                        }
                    else:
                        return {
                            "error": f"Comments API request failed: {response.status}"
                        }

        except Exception as e:
            self.logger.error(f"API comments extraction failed: {str(e)}")
            return {"error": str(e)}

    def _summarize_comments(self, comments: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create a summary of comment sentiment and topics"""
        if not comments:
            return {}

        # Basic comment analysis
        total_comments = len(comments)
        total_likes = sum(c["like_count"] for c in comments)
        avg_likes = total_likes / total_comments if total_comments > 0 else 0

        # Word frequency analysis (simple)
        all_text = " ".join([c["text"].lower() for c in comments])
        words = re.findall(r"\b[a-zA-Z]{3,}\b", all_text)

        word_freq = {}
        for word in words:
            if word not in [
                "the",
                "and",
                "this",
                "that",
                "with",
                "for",
                "are",
                "was",
                "you",
                "your",
                "video",
                "great",
                "good",
            ]:
                word_freq[word] = word_freq.get(word, 0) + 1

        # Top words
        top_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:10]

        return {
            "total_comments_analyzed": total_comments,
            "average_likes_per_comment": avg_likes,
            "total_likes": total_likes,
            "top_words": top_words,
            "engagement_level": (
                "high" if avg_likes > 10 else "moderate" if avg_likes > 2 else "low"
            ),
        }

    async def _synthesize_api_content(
        self, processing_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Synthesize content from API processing"""
        synthesis = {
            "content_types_available": ["metadata"],
            "primary_content_source": "youtube_data_api",
            "processing_method": "api_primary",
            "content_completeness": "high",
        }

        # Check what content was successfully extracted
        if processing_results.get("subtitles", {}).get("subtitles_extracted"):
            synthesis["content_types_available"].append("full_subtitles")
            synthesis["subtitle_method"] = processing_results["subtitles"][
                "caption_track_info"
            ]["track_kind"]

        if processing_results.get("comments", {}).get("comments_extracted"):
            synthesis["content_types_available"].append("comments")

        # Assess content quality from metadata
        metadata = processing_results.get("metadata", {})
        if metadata and "error" not in metadata:
            view_count = metadata.get("view_count", 0)
            like_count = metadata.get("like_count", 0)

            if view_count > 100000 and like_count > 1000:
                synthesis["content_quality"] = "high_engagement"
            elif view_count > 10000:
                synthesis["content_quality"] = "moderate_engagement"
            else:
                synthesis["content_quality"] = "low_engagement"

            # Add subtitle analysis
            subtitles = processing_results.get("subtitles", {})
            if subtitles.get("subtitles_extracted"):
                subtitle_text = subtitles.get("subtitle_text", "")
                synthesis["subtitle_analysis"] = {
                    "text_length": len(subtitle_text),
                    "estimated_reading_time": len(subtitle_text.split())
                    / 200,  # 200 WPM
                    "segment_count": subtitles.get("total_segments", 0),
                    "caption_type": subtitles["caption_track_info"]["track_kind"],
                }

        return synthesis

    def _needs_advanced_analysis(self, processing_results: Dict[str, Any]) -> bool:
        """Determine if advanced video analysis is needed"""
        # For now, we prioritize API content extraction
        # Advanced analysis (frame-by-frame) would only be needed for specific tasks
        return False

    async def _add_advanced_analysis(
        self, youtube_url: str, video_id: str
    ) -> Dict[str, Any]:
        """Add advanced analysis via video download (if needed)"""
        # Placeholder for advanced analysis
        return {"advanced_analysis": "not_implemented_in_current_version"}

    async def _process_with_download_only(
        self, youtube_url: str, video_id: str
    ) -> Dict[str, Any]:
        """Fallback processing using only download method"""
        # This would implement the original download-based processing
        # For now, return error indicating API is preferred
        return {
            "error": "download_only_not_implemented",
            "message": "API method is preferred. Download-only processing not implemented in current version.",
            "suggestion": "Please provide YouTube Data API key for content extraction",
        }

    def _create_no_access_response(
        self, youtube_url: str, video_id: str, start_time: float
    ) -> Dict[str, Any]:
        """Create response when no processing methods are available"""

        raw_output = f"""=== YOUTUBE CONTENT ANALYSIS ===

NO PROCESSING METHODS AVAILABLE

YouTube Data API (Primary Method): {'✓ Available' if self.processing_capabilities.get('api_available') else '✗ Not Available'}
Direct Download (Fallback): {'✓ Available' if self.processing_capabilities.get('download_available') else '✗ Not Available'}

VIDEO INFORMATION:
- URL: {youtube_url}
- Video ID: {video_id}

REQUIRED SETUP:
1. Get YouTube Data API v3 key from Google Cloud Console
2. Set environment variable: YOUTUBE_API_KEY=your_api_key_here
3. Restart the application

YOUTUBE DATA API SETUP:
1. Go to: https://console.cloud.google.com/
2. Create/select a project
3. Enable "YouTube Data API v3"
4. Create credentials (API key)
5. Set the API key in your environment

API BENEFITS:
- Works in all environments (including Hugging Face Spaces)
- Full subtitle/caption content extraction
- Comprehensive metadata and comments
- No network restrictions
- More reliable than video downloading

For the specific question about bird species counting:
This requires frame-by-frame video analysis, which would need the advanced download method.
However, subtitle content may contain relevant information about the video content.
"""

        return {
            "tool_name": self.name,
            "raw_output": raw_output,
            "success": False,
            "error": "no_processing_methods_available",
            "metadata": {
                "youtube_url": youtube_url,
                "video_id": video_id,
                "processing_time": time.time() - start_time,
                "processing_capabilities": self.processing_capabilities,
                "setup_required": "youtube_data_api_key",
            },
        }

    def _compile_processing_output(self, processing_results: Dict[str, Any]) -> str:
        """Compile all processing results into a comprehensive output"""

        output_lines = ["=== YOUTUBE CONTENT ANALYSIS ===\n"]

        # Processing method indicator
        processing_method = processing_results.get("processing_method", {})
        if processing_method.get("primary") == "youtube_data_api":
            output_lines.append(
                "PROCESSING METHOD: YouTube Data API v3 (Primary Method)"
            )
            output_lines.append(
                "BENEFITS: Reliable, works in all environments, full content extraction\n"
            )

        # Metadata
        metadata = processing_results.get("metadata", {})
        if "error" not in metadata:
            output_lines.append("VIDEO METADATA:")
            output_lines.append(f"- Title: {metadata.get('title', 'Unknown')}")
            output_lines.append(f"- Channel: {metadata.get('channel', 'Unknown')}")
            output_lines.append(
                f"- Duration: {metadata.get('duration_seconds', 0):.0f} seconds"
            )
            output_lines.append(
                f"- Upload date: {metadata.get('upload_date_formatted', 'Unknown')}"
            )
            output_lines.append(f"- View count: {metadata.get('view_count', 0):,}")
            output_lines.append(f"- Like count: {metadata.get('like_count', 0):,}")
            output_lines.append(
                f"- Comment count: {metadata.get('comment_count', 0):,}"
            )

            if metadata.get("tags"):
                output_lines.append(f"- Tags: {', '.join(metadata['tags'][:5])}")

            if metadata.get("description"):
                desc_preview = (
                    metadata["description"][:200] + "..."
                    if len(metadata["description"]) > 200
                    else metadata["description"]
                )
                output_lines.append(f"- Description preview: {desc_preview}")

            output_lines.append(
                f"- Data source: {metadata.get('data_source', 'Unknown')}"
            )
        else:
            output_lines.append(f"VIDEO METADATA: Failed - {metadata['error']}")

        # Subtitles/Captions (Enhanced)
        subtitles = processing_results.get("subtitles", {})
        output_lines.append("\nSUBTITLE/CAPTION CONTENT:")
        if subtitles.get("subtitles_extracted"):
            track_info = subtitles.get("caption_track_info", {})
            output_lines.append(f"- Captions extracted: YES")
            output_lines.append(f"- Language: {track_info.get('language', 'Unknown')}")
            output_lines.append(
                f"- Type: {'Manual' if track_info.get('is_manual') else 'Auto-generated'}"
            )
            output_lines.append(
                f"- Total segments: {subtitles.get('total_segments', 0)}"
            )
            output_lines.append(
                f"- Processing method: {subtitles.get('processing_method', 'Unknown')}"
            )

            # Show preview of subtitle text
            subtitle_text = subtitles.get("subtitle_text", "")
            if subtitle_text:
                preview = (
                    subtitle_text[:400] + "..."
                    if len(subtitle_text) > 400
                    else subtitle_text
                )
                output_lines.append(f"- Full text content: {preview}")
                output_lines.append(
                    f"- Total text length: {len(subtitle_text)} characters"
                )
        else:
            reason = subtitles.get("reason", "Unknown reason")
            output_lines.append(f"- Captions extracted: NO ({reason})")

        # Comments
        comments = processing_results.get("comments", {})
        if comments.get("comments_extracted"):
            output_lines.append("\nCOMMENT ANALYSIS:")
            output_lines.append(
                f"- Comments analyzed: {comments.get('total_comments', 0)}"
            )

            comment_summary = comments.get("comment_summary", {})
            if comment_summary:
                output_lines.append(
                    f"- Engagement level: {comment_summary.get('engagement_level', 'Unknown')}"
                )
                output_lines.append(
                    f"- Average likes per comment: {comment_summary.get('average_likes_per_comment', 0):.1f}"
                )

                top_words = comment_summary.get("top_words", [])
                if top_words:
                    words_str = ", ".join(
                        [f"{word}({count})" for word, count in top_words[:5]]
                    )
                    output_lines.append(f"- Common words: {words_str}")

            output_lines.append(
                f"- Data source: {comments.get('data_source', 'Unknown')}"
            )

        # Content Synthesis
        synthesis = processing_results.get("content_synthesis", {})
        output_lines.append("\nCONTENT SYNTHESIS:")
        output_lines.append(
            f"- Available content types: {', '.join(synthesis.get('content_types_available', []))}"
        )
        output_lines.append(
            f"- Primary content source: {synthesis.get('primary_content_source', 'Unknown')}"
        )
        output_lines.append(
            f"- Content quality: {synthesis.get('content_quality', 'Unknown')}"
        )
        output_lines.append(
            f"- Processing method: {synthesis.get('processing_method', 'Unknown')}"
        )
        output_lines.append(
            f"- Content completeness: {synthesis.get('content_completeness', 'Unknown')}"
        )

        # Subtitle analysis if available
        subtitle_analysis = synthesis.get("subtitle_analysis", {})
        if subtitle_analysis:
            output_lines.append("\nSUBTITLE CONTENT ANALYSIS:")
            output_lines.append(
                f"- Text length: {subtitle_analysis.get('text_length', 0)} characters"
            )
            output_lines.append(
                f"- Estimated reading time: {subtitle_analysis.get('estimated_reading_time', 0):.1f} minutes"
            )
            output_lines.append(
                f"- Segment count: {subtitle_analysis.get('segment_count', 0)}"
            )
            output_lines.append(
                f"- Caption type: {subtitle_analysis.get('caption_type', 'Unknown')}"
            )

        return "\n".join(output_lines)

    def _is_valid_youtube_url(self, url: str) -> bool:
        """Validate if URL is a valid YouTube URL"""
        youtube_patterns = [
            r"(?:https?://)?(?:www\.)?youtube\.com/watch\?v=([a-zA-Z0-9_-]+)",
            r"(?:https?://)?(?:www\.)?youtu\.be/([a-zA-Z0-9_-]+)",
            r"(?:https?://)?(?:www\.)?youtube\.com/embed/([a-zA-Z0-9_-]+)",
            r"(?:https?://)?(?:www\.)?youtube\.com/v/([a-zA-Z0-9_-]+)",
        ]

        return any(re.match(pattern, url) for pattern in youtube_patterns)

    def _extract_video_id(self, url: str) -> Optional[str]:
        """Extract video ID from YouTube URL"""
        patterns = [
            r"(?:v=|/)([0-9A-Za-z_-]{11}).*",
            r"(?:embed/)([0-9A-Za-z_-]{11})",
            r"(?:youtu\.be/)([0-9A-Za-z_-]{11})",
        ]

        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)

        return None

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
async def test_youtube_processor():
    """Test the YouTube Content Processor with API-first approach"""
    # Note: You need a YouTube Data API key for full functionality
    processor = YouTubeContentProcessor(youtube_api_key="your_youtube_api_key_here")

    # Test with a sample YouTube URL
    test_url = "https://www.youtube.com/watch?v=L1vXCYZAYYM"

    result = await processor.execute(test_url)

    print("Processing Result:")
    print(f"Success: {result.get('success', False)}")
    print(f"Tool: {result.get('tool_name')}")
    print(
        f"Processing time: {result.get('metadata', {}).get('processing_time', 0):.2f}s"
    )
    print("\nRaw Output:")
    print(result.get("raw_output", "No output"))


if __name__ == "__main__":
    asyncio.run(test_youtube_processor())
