# tools/audio_processor.py - Universal Audio Processor
import asyncio
import logging
import tempfile
import time
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import json
import wave
import subprocess

try:
    import librosa
    import numpy as np
    import scipy.signal
    import soundfile as sf

    AUDIO_LIBS_AVAILABLE = True
except ImportError:
    print("Warning: Audio processing libraries not installed. Install with:")
    print("pip install librosa soundfile scipy")
    AUDIO_LIBS_AVAILABLE = False

from .base_tool import UniversalTool


class UniversalAudioProcessor(UniversalTool):
    """
    Universal Audio Processor - Processes any audio content for information extraction.

    This tool provides raw audio data extraction that Claude can interpret contextually:
    - Audio transcription to text
    - Speaker identification and diarization
    - Audio quality metrics and technical analysis
    - Background sound identification
    - Temporal audio analysis

    Anti-pattern: NO music analysis, speech emotion, or domain-specific audio tools
    Usage: Converts audio to analyzable text/data for Claude's reasoning
    """

    def __init__(self):
        super().__init__("Universal Audio Processor")
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

        # Audio processing capabilities
        self.capabilities = [
            "transcription",
            "speaker_analysis",
            "audio_quality_analysis",
            "temporal_analysis",
            "frequency_analysis",
            "background_sound_detection",
        ]

        # Configuration
        self.config = {
            "sample_rate": 16000,  # Standard for speech processing
            "chunk_duration": 30,  # Process in 30-second chunks
            "min_speech_duration": 0.5,  # Minimum speech segment duration
            "noise_threshold": 0.01,  # Noise detection threshold
        }

        # Initialize transcription engines
        self._initialize_transcription_engines()

        self.logger.info("Universal Audio Processor initialized")

    def _initialize_transcription_engines(self):
        """Initialize available transcription engines"""
        self.transcription_engines = {}

        # Check for Whisper (OpenAI's speech recognition)
        try:
            import whisper

            self.transcription_engines["whisper"] = {
                "available": True,
                "model": None,  # Will be loaded on first use
            }
            self.logger.info("Whisper transcription engine available")
        except ImportError:
            self.transcription_engines["whisper"] = {"available": False}
            self.logger.warning(
                "Whisper not available - install with: pip install openai-whisper"
            )

        # Check for SpeechRecognition library
        try:
            import speech_recognition as sr

            self.transcription_engines["speech_recognition"] = {
                "available": True,
                "recognizer": sr.Recognizer(),
            }
            self.logger.info("SpeechRecognition engine available")
        except ImportError:
            self.transcription_engines["speech_recognition"] = {"available": False}
            self.logger.warning("SpeechRecognition not available")

        # Check for system FFmpeg for audio conversion
        try:
            subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
            self.transcription_engines["ffmpeg"] = {"available": True}
            self.logger.info("FFmpeg available for audio conversion")
        except (subprocess.CalledProcessError, FileNotFoundError):
            self.transcription_engines["ffmpeg"] = {"available": False}
            self.logger.warning(
                "FFmpeg not available - some audio formats may not be supported"
            )

    async def execute(self, audio_path: str) -> Dict[str, Any]:
        """
        Execute comprehensive audio analysis

        Args:
            audio_path: Path to audio file

        Returns:
            Standardized output with audio analysis data
        """
        start_time = time.time()

        try:
            # Load and validate audio
            audio_data, sample_rate = await self._load_audio(audio_path)
            if audio_data is None:
                return self._error_output("Failed to load audio file")

            # Perform comprehensive analysis
            analysis_results = {}

            # 1. Basic audio properties
            analysis_results["properties"] = await self._analyze_properties(
                audio_data, sample_rate
            )

            # 2. Audio transcription
            analysis_results["transcription"] = await self._transcribe_audio(
                audio_path, audio_data, sample_rate
            )

            # 3. Speaker analysis
            analysis_results["speaker_analysis"] = await self._analyze_speakers(
                audio_data, sample_rate
            )

            # 4. Audio quality metrics
            analysis_results["quality_metrics"] = await self._analyze_audio_quality(
                audio_data, sample_rate
            )

            # 5. Temporal analysis
            analysis_results["temporal_analysis"] = (
                await self._analyze_temporal_patterns(audio_data, sample_rate)
            )

            # 6. Frequency analysis
            analysis_results["frequency_analysis"] = (
                await self._analyze_frequency_content(audio_data, sample_rate)
            )

            # 7. Background sound detection
            analysis_results["background_analysis"] = (
                await self._analyze_background_sounds(audio_data, sample_rate)
            )

            # Compile comprehensive output
            raw_output = self._compile_analysis_output(analysis_results)

            metadata = {
                "audio_path": audio_path,
                "analysis_time": time.time() - start_time,
                "original_sample_rate": sample_rate,
                "duration_seconds": len(audio_data) / sample_rate,
                "transcription_engines_available": {
                    engine: info["available"]
                    for engine, info in self.transcription_engines.items()
                },
            }

            return self._standardize_output(raw_output, metadata)

        except Exception as e:
            self.logger.error(f"Audio analysis failed: {str(e)}")
            return self._error_output(f"Analysis failed: {str(e)}")

    async def _load_audio(self, audio_path: str) -> Tuple[Optional[np.ndarray], int]:
        """Load and preprocess audio file"""
        try:
            if not Path(audio_path).exists():
                self.logger.error(f"Audio file not found: {audio_path}")
                return None, 0

            if not AUDIO_LIBS_AVAILABLE:
                self.logger.error("Audio processing libraries not available")
                return None, 0

            # Load audio with librosa
            audio_data, sample_rate = librosa.load(
                audio_path,
                sr=self.config["sample_rate"],  # Resample to standard rate
                mono=True,  # Convert to mono
            )

            if len(audio_data) == 0:
                self.logger.error(f"Empty audio file: {audio_path}")
                return None, 0

            # Normalize audio
            audio_data = librosa.util.normalize(audio_data)

            self.logger.debug(
                f"Loaded audio: {len(audio_data)} samples at {sample_rate} Hz"
            )
            return audio_data, sample_rate

        except Exception as e:
            self.logger.error(f"Error loading audio {audio_path}: {str(e)}")

            # Fallback: try with FFmpeg conversion
            if self.transcription_engines["ffmpeg"]["available"]:
                return await self._load_audio_with_ffmpeg(audio_path)

            return None, 0

    async def _load_audio_with_ffmpeg(
        self, audio_path: str
    ) -> Tuple[Optional[np.ndarray], int]:
        """Load audio using FFmpeg conversion as fallback"""
        try:
            # Convert to WAV using FFmpeg
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_wav:
                temp_wav_path = temp_wav.name

            ffmpeg_cmd = [
                "ffmpeg",
                "-i",
                audio_path,
                "-ar",
                str(self.config["sample_rate"]),
                "-ac",
                "1",  # Mono
                "-y",
                temp_wav_path,
            ]

            process = await asyncio.create_subprocess_exec(
                *ffmpeg_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            await process.communicate()

            if process.returncode == 0 and AUDIO_LIBS_AVAILABLE:
                audio_data, sample_rate = librosa.load(temp_wav_path)
                Path(temp_wav_path).unlink()  # Clean up
                return librosa.util.normalize(audio_data), sample_rate
            else:
                Path(temp_wav_path).unlink()  # Clean up
                return None, 0

        except Exception as e:
            self.logger.error(f"FFmpeg audio conversion failed: {str(e)}")
            return None, 0

    async def _analyze_properties(
        self, audio_data: np.ndarray, sample_rate: int
    ) -> Dict[str, Any]:
        """Analyze basic audio properties"""
        duration = len(audio_data) / sample_rate

        # Audio level analysis
        rms_energy = np.sqrt(np.mean(audio_data**2))
        peak_amplitude = np.max(np.abs(audio_data))

        # Dynamic range
        dynamic_range = 20 * np.log10(peak_amplitude / max(rms_energy, 1e-10))

        # Zero crossing rate (indicator of speech vs music)
        zcr = (
            librosa.feature.zero_crossing_rate(audio_data)[0]
            if AUDIO_LIBS_AVAILABLE
            else [0]
        )
        avg_zcr = np.mean(zcr)

        return {
            "duration_seconds": round(duration, 2),
            "sample_rate": sample_rate,
            "total_samples": len(audio_data),
            "rms_energy": float(rms_energy),
            "peak_amplitude": float(peak_amplitude),
            "dynamic_range_db": float(dynamic_range),
            "average_zero_crossing_rate": float(avg_zcr),
            "estimated_content_type": (
                "speech" if avg_zcr > 0.1 else "music_or_environmental"
            ),
        }

    async def _transcribe_audio(
        self, audio_path: str, audio_data: np.ndarray, sample_rate: int
    ) -> Dict[str, Any]:
        """Transcribe audio using available engines"""
        transcription_results = {
            "engines_used": [],
            "transcribed_text": "",
            "confidence_scores": [],
            "word_timestamps": [],
            "detected_language": "unknown",
        }

        # Try Whisper first (most accurate)
        if self.transcription_engines["whisper"]["available"]:
            whisper_result = await self._transcribe_with_whisper(audio_path)
            if whisper_result["success"]:
                transcription_results["engines_used"].append("whisper")
                transcription_results["transcribed_text"] = whisper_result["text"]
                transcription_results["confidence_scores"].append(
                    whisper_result["confidence"]
                )
                transcription_results["detected_language"] = whisper_result.get(
                    "language", "unknown"
                )
                if whisper_result.get("segments"):
                    transcription_results["word_timestamps"] = whisper_result[
                        "segments"
                    ]

        # Fallback to SpeechRecognition
        elif self.transcription_engines["speech_recognition"]["available"]:
            sr_result = await self._transcribe_with_speech_recognition(audio_path)
            if sr_result["success"]:
                transcription_results["engines_used"].append("speech_recognition")
                transcription_results["transcribed_text"] = sr_result["text"]
                transcription_results["confidence_scores"].append(
                    sr_result.get("confidence", 0.5)
                )

        # Analyze transcription quality
        if transcription_results["transcribed_text"]:
            transcription_results["analysis"] = self._analyze_transcription_quality(
                transcription_results["transcribed_text"]
            )

        transcription_results["has_speech"] = bool(
            transcription_results["transcribed_text"].strip()
        )

        return transcription_results

    async def _transcribe_with_whisper(self, audio_path: str) -> Dict[str, Any]:
        """Transcribe using OpenAI Whisper"""
        try:
            import whisper

            # Load model if not already loaded
            if self.transcription_engines["whisper"]["model"] is None:
                self.logger.info("Loading Whisper model...")
                self.transcription_engines["whisper"]["model"] = whisper.load_model(
                    "base"
                )

            model = self.transcription_engines["whisper"]["model"]

            # Transcribe
            result = model.transcribe(audio_path, word_timestamps=True)

            # Calculate average confidence from segments
            avg_confidence = 0.8  # Whisper doesn't provide confidence, use default
            if "segments" in result:
                # Use proxy confidence based on segment consistency
                segment_lengths = [
                    len(seg.get("text", "")) for seg in result["segments"]
                ]
                avg_confidence = min(
                    0.9,
                    0.5
                    + (
                        np.std(segment_lengths) / np.mean(segment_lengths)
                        if segment_lengths
                        else 0
                    ),
                )

            return {
                "success": True,
                "text": result["text"].strip(),
                "language": result.get("language", "unknown"),
                "confidence": avg_confidence,
                "segments": result.get("segments", []),
            }

        except Exception as e:
            self.logger.warning(f"Whisper transcription failed: {str(e)}")
            return {"success": False, "error": str(e)}

    async def _transcribe_with_speech_recognition(
        self, audio_path: str
    ) -> Dict[str, Any]:
        """Transcribe using SpeechRecognition library"""
        try:
            import speech_recognition as sr

            recognizer = self.transcription_engines["speech_recognition"]["recognizer"]

            # Convert to WAV if needed
            with tempfile.NamedTemporaryFile(suffix=".wav") as temp_wav:
                if not audio_path.lower().endswith(".wav"):
                    # Convert to WAV format
                    if self.transcription_engines["ffmpeg"]["available"]:
                        ffmpeg_cmd = ["ffmpeg", "-i", audio_path, "-y", temp_wav.name]
                        process = await asyncio.create_subprocess_exec(*ffmpeg_cmd)
                        await process.communicate()
                        audio_file_path = temp_wav.name
                    else:
                        raise Exception("Cannot convert audio format without FFmpeg")
                else:
                    audio_file_path = audio_path

                # Transcribe
                with sr.AudioFile(audio_file_path) as source:
                    audio = recognizer.record(source)

                # Try Google Speech Recognition (free tier)
                text = recognizer.recognize_google(audio)

                return {
                    "success": True,
                    "text": text,
                    "confidence": 0.7,  # Default confidence for Google API
                }

        except Exception as e:
            self.logger.warning(f"SpeechRecognition failed: {str(e)}")
            return {"success": False, "error": str(e)}

    def _analyze_transcription_quality(self, text: str) -> Dict[str, Any]:
        """Analyze quality of transcribed text"""
        words = text.split()

        return {
            "word_count": len(words),
            "character_count": len(text),
            "average_word_length": (
                np.mean([len(word) for word in words]) if words else 0
            ),
            "has_punctuation": any(char in text for char in ".,!?;:"),
            "has_capitalization": any(char.isupper() for char in text),
            "estimated_speaking_rate": len(words)
            / max(1, len(text) / 100),  # Rough WPM estimate
            "language_indicators": self._detect_language_patterns(text),
        }

    def _detect_language_patterns(self, text: str) -> Dict[str, Any]:
        """Detect language patterns in text"""
        text_lower = text.lower()

        # Simple language detection patterns
        english_patterns = ["the", "and", "is", "in", "to", "of", "a", "that"]
        spanish_patterns = ["el", "la", "de", "que", "y", "en", "un", "es"]
        french_patterns = ["le", "de", "et", "à", "un", "il", "être", "et"]

        pattern_matches = {
            "english": sum(1 for pattern in english_patterns if pattern in text_lower),
            "spanish": sum(1 for pattern in spanish_patterns if pattern in text_lower),
            "french": sum(1 for pattern in french_patterns if pattern in text_lower),
        }

        return pattern_matches

    async def _analyze_speakers(
        self, audio_data: np.ndarray, sample_rate: int
    ) -> Dict[str, Any]:
        """Analyze speaker characteristics and count"""
        try:
            if not AUDIO_LIBS_AVAILABLE:
                return {"error": "Audio libraries not available"}

            # Simple speaker analysis using audio features
            # This is a basic implementation - production would use specialized models

            # Split audio into segments for analysis
            segment_duration = 5  # 5-second segments
            segment_samples = segment_duration * sample_rate
            segments = [
                audio_data[i : i + segment_samples]
                for i in range(0, len(audio_data), segment_samples)
                if len(audio_data[i : i + segment_samples]) >= segment_samples // 2
            ]

            speaker_features = []
            for segment in segments:
                if len(segment) > 0:
                    # Extract basic voice features
                    mfccs = librosa.feature.mfcc(y=segment, sr=sample_rate, n_mfcc=13)
                    spectral_centroid = librosa.feature.spectral_centroid(
                        y=segment, sr=sample_rate
                    )

                    features = {
                        "mfcc_mean": np.mean(mfccs, axis=1).tolist(),
                        "spectral_centroid": float(np.mean(spectral_centroid)),
                        "energy": float(np.mean(segment**2)),
                    }
                    speaker_features.append(features)

            # Estimate number of speakers using clustering of features
            if len(speaker_features) > 1:
                estimated_speakers = self._estimate_speaker_count(speaker_features)
            else:
                estimated_speakers = 1 if speaker_features else 0

            # Voice characteristics
            voice_characteristics = self._analyze_voice_characteristics(
                audio_data, sample_rate
            )

            return {
                "estimated_speaker_count": estimated_speakers,
                "voice_characteristics": voice_characteristics,
                "segment_count": len(segments),
                "analysis_confidence": "low",  # Simple analysis has low confidence
            }

        except Exception as e:
            self.logger.warning(f"Speaker analysis failed: {str(e)}")
            return {"error": f"Speaker analysis failed: {str(e)}"}

    def _estimate_speaker_count(self, speaker_features: List[Dict]) -> int:
        """Estimate speaker count using simple clustering"""
        try:
            # Simple speaker count estimation using spectral centroid variance
            centroids = [features["spectral_centroid"] for features in speaker_features]

            if len(centroids) < 2:
                return 1

            # Use coefficient of variation to estimate speaker diversity
            cv = np.std(centroids) / np.mean(centroids) if np.mean(centroids) > 0 else 0

            # Simple heuristic: high variation suggests multiple speakers
            if cv > 0.3:
                return min(3, int(cv * 5))  # Cap at 3 speakers for simple analysis
            else:
                return 1

        except Exception:
            return 1

    def _analyze_voice_characteristics(
        self, audio_data: np.ndarray, sample_rate: int
    ) -> Dict[str, Any]:
        """Analyze voice characteristics"""
        try:
            # Fundamental frequency (pitch) analysis
            pitches, magnitudes = librosa.core.piptrack(y=audio_data, sr=sample_rate)
            pitch_values = pitches[magnitudes > np.median(magnitudes)]

            if len(pitch_values) > 0:
                avg_pitch = np.mean(pitch_values[pitch_values > 0])
                pitch_range = np.max(pitch_values) - np.min(
                    pitch_values[pitch_values > 0]
                )
            else:
                avg_pitch = 0
                pitch_range = 0

            # Speaking rate estimation
            onset_frames = librosa.onset.onset_detect(y=audio_data, sr=sample_rate)
            speaking_rate = (
                len(onset_frames) / (len(audio_data) / sample_rate) * 60
            )  # onsets per minute

            return {
                "average_pitch_hz": float(avg_pitch),
                "pitch_range_hz": float(pitch_range),
                "estimated_speaking_rate": float(speaking_rate),
                "voice_classification": self._classify_voice_type(avg_pitch),
            }

        except Exception as e:
            self.logger.warning(f"Voice characteristic analysis failed: {str(e)}")
            return {"error": str(e)}

    def _classify_voice_type(self, avg_pitch: float) -> str:
        """Simple voice type classification based on pitch"""
        if avg_pitch == 0:
            return "undetected"
        elif avg_pitch < 150:
            return "low_pitch"
        elif avg_pitch < 250:
            return "medium_pitch"
        else:
            return "high_pitch"

    async def _analyze_audio_quality(
        self, audio_data: np.ndarray, sample_rate: int
    ) -> Dict[str, Any]:
        """Analyze audio quality metrics"""
        try:
            # Signal-to-noise ratio estimation
            signal_power = np.mean(audio_data**2)
            noise_estimate = np.percentile(
                np.abs(audio_data), 10
            )  # Estimate noise from quiet parts
            snr_db = 10 * np.log10(signal_power / max(noise_estimate**2, 1e-10))

            # Clipping detection
            clipping_threshold = 0.95
            clipped_samples = np.sum(np.abs(audio_data) > clipping_threshold)
            clipping_percentage = (clipped_samples / len(audio_data)) * 100

            # Frequency response analysis
            if AUDIO_LIBS_AVAILABLE:
                freqs, psd = scipy.signal.welch(audio_data, sample_rate)

                # Find dominant frequency
                dominant_freq_idx = np.argmax(psd)
                dominant_frequency = freqs[dominant_freq_idx]

                # Frequency distribution
                low_freq_power = np.sum(psd[freqs < 300])
                mid_freq_power = np.sum(psd[(freqs >= 300) & (freqs < 3000)])
                high_freq_power = np.sum(psd[freqs >= 3000])
                total_power = low_freq_power + mid_freq_power + high_freq_power

                freq_distribution = {
                    "low_freq_percentage": (
                        (low_freq_power / total_power) * 100 if total_power > 0 else 0
                    ),
                    "mid_freq_percentage": (
                        (mid_freq_power / total_power) * 100 if total_power > 0 else 0
                    ),
                    "high_freq_percentage": (
                        (high_freq_power / total_power) * 100 if total_power > 0 else 0
                    ),
                }
            else:
                dominant_frequency = 0
                freq_distribution = {}

            # Overall quality assessment
            quality_score = min(
                100, max(0, 100 - (clipping_percentage * 20) + min(snr_db / 2, 25))
            )

            return {
                "signal_to_noise_ratio_db": float(snr_db),
                "clipping_percentage": float(clipping_percentage),
                "dominant_frequency_hz": float(dominant_frequency),
                "frequency_distribution": freq_distribution,
                "quality_score": float(quality_score),
                "quality_assessment": self._assess_quality_level(quality_score),
            }

        except Exception as e:
            self.logger.warning(f"Audio quality analysis failed: {str(e)}")
            return {"error": str(e)}

    def _assess_quality_level(self, quality_score: float) -> str:
        """Assess quality level based on score"""
        if quality_score >= 80:
            return "excellent"
        elif quality_score >= 60:
            return "good"
        elif quality_score >= 40:
            return "fair"
        else:
            return "poor"

    async def _analyze_temporal_patterns(
        self, audio_data: np.ndarray, sample_rate: int
    ) -> Dict[str, Any]:
        """Analyze temporal patterns in audio"""
        try:
            duration = len(audio_data) / sample_rate

            # Silence detection
            silence_threshold = np.percentile(np.abs(audio_data), 20)
            silence_mask = np.abs(audio_data) < silence_threshold

            # Find silence and speech segments
            silence_segments = []
            speech_segments = []

            current_type = "silence" if silence_mask[0] else "speech"
            segment_start = 0

            for i, is_silence in enumerate(silence_mask):
                if (is_silence and current_type == "speech") or (
                    not is_silence and current_type == "silence"
                ):
                    # Segment boundary
                    segment_end = i / sample_rate
                    segment_duration = segment_end - segment_start

                    if current_type == "silence":
                        silence_segments.append(
                            {"start": segment_start, "duration": segment_duration}
                        )
                    else:
                        speech_segments.append(
                            {"start": segment_start, "duration": segment_duration}
                        )

                    current_type = "silence" if is_silence else "speech"
                    segment_start = segment_end

            # Add final segment
            final_duration = duration - segment_start
            if current_type == "silence":
                silence_segments.append(
                    {"start": segment_start, "duration": final_duration}
                )
            else:
                speech_segments.append(
                    {"start": segment_start, "duration": final_duration}
                )

            # Calculate statistics
            total_silence = sum(seg["duration"] for seg in silence_segments)
            total_speech = sum(seg["duration"] for seg in speech_segments)

            return {
                "total_duration": duration,
                "speech_duration": total_speech,
                "silence_duration": total_silence,
                "speech_percentage": (
                    (total_speech / duration) * 100 if duration > 0 else 0
                ),
                "silence_segments": len(silence_segments),
                "speech_segments": len(speech_segments),
                "average_speech_segment_duration": (
                    total_speech / len(speech_segments) if speech_segments else 0
                ),
                "average_silence_segment_duration": (
                    total_silence / len(silence_segments) if silence_segments else 0
                ),
                "speaking_pattern": self._classify_speaking_pattern(
                    speech_segments, silence_segments
                ),
            }

        except Exception as e:
            self.logger.warning(f"Temporal analysis failed: {str(e)}")
            return {"error": str(e)}

    def _classify_speaking_pattern(
        self, speech_segments: List[Dict], silence_segments: List[Dict]
    ) -> str:
        """Classify the speaking pattern"""
        if not speech_segments:
            return "no_speech"

        avg_speech_duration = np.mean([seg["duration"] for seg in speech_segments])
        avg_silence_duration = (
            np.mean([seg["duration"] for seg in silence_segments])
            if silence_segments
            else 0
        )

        if avg_speech_duration > 10 and avg_silence_duration < 2:
            return "continuous_speech"
        elif len(speech_segments) > 5 and avg_silence_duration > 1:
            return "conversational"
        elif avg_speech_duration < 5:
            return "fragmented"
        else:
            return "standard"

    async def _analyze_frequency_content(
        self, audio_data: np.ndarray, sample_rate: int
    ) -> Dict[str, Any]:
        """Analyze frequency content of audio"""
        try:
            if not AUDIO_LIBS_AVAILABLE:
                return {"error": "Audio libraries not available"}

            # Compute spectrogram
            stft = librosa.stft(audio_data)
            magnitude = np.abs(stft)

            # Frequency bins
            freqs = librosa.fft_frequencies(sr=sample_rate)

            # Spectral features
            spectral_centroid = librosa.feature.spectral_centroid(
                y=audio_data, sr=sample_rate
            )
            spectral_bandwidth = librosa.feature.spectral_bandwidth(
                y=audio_data, sr=sample_rate
            )
            spectral_rolloff = librosa.feature.spectral_rolloff(
                y=audio_data, sr=sample_rate
            )

            # MFCCs for voice characteristics
            mfccs = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=13)

            return {
                "spectral_centroid_hz": float(np.mean(spectral_centroid)),
                "spectral_bandwidth_hz": float(np.mean(spectral_bandwidth)),
                "spectral_rolloff_hz": float(np.mean(spectral_rolloff)),
                "mfcc_coefficients": np.mean(mfccs, axis=1).tolist()[
                    :5
                ],  # First 5 MFCCs
                "frequency_range": {
                    "min_hz": float(freqs[0]),
                    "max_hz": float(freqs[-1]),
                },
                "dominant_frequencies": self._find_dominant_frequencies(
                    magnitude, freqs
                ),
            }

        except Exception as e:
            self.logger.warning(f"Frequency analysis failed: {str(e)}")
            return {"error": str(e)}

    def _find_dominant_frequencies(
        self, magnitude: np.ndarray, freqs: np.ndarray, top_n: int = 5
    ) -> List[Dict]:
        """Find dominant frequencies in the spectrum"""
        try:
            # Average magnitude across time
            avg_magnitude = np.mean(magnitude, axis=1)

            # Find peaks
            peaks = scipy.signal.find_peaks(
                avg_magnitude, height=np.percentile(avg_magnitude, 75)
            )[0]

            # Sort by magnitude and take top N
            peak_magnitudes = avg_magnitude[peaks]
            sorted_indices = np.argsort(peak_magnitudes)[::-1]
            top_peaks = peaks[sorted_indices[:top_n]]

            dominant_freqs = []
            for peak_idx in top_peaks:
                dominant_freqs.append(
                    {
                        "frequency_hz": float(freqs[peak_idx]),
                        "magnitude": float(avg_magnitude[peak_idx]),
                    }
                )

            return dominant_freqs

        except Exception:
            return []

    async def _analyze_background_sounds(
        self, audio_data: np.ndarray, sample_rate: int
    ) -> Dict[str, Any]:
        """Analyze background sounds and noise"""
        try:
            # Simple background sound analysis
            # This would benefit from trained models for better accuracy

            # Detect consistent low-level sounds (potential background)
            energy_threshold = np.percentile(np.abs(audio_data), 25)
            background_mask = np.abs(audio_data) < energy_threshold

            background_percentage = np.sum(background_mask) / len(audio_data) * 100

            # Analyze frequency content of background
            if (
                AUDIO_LIBS_AVAILABLE and np.sum(background_mask) > 1000
            ):  # Enough samples
                background_audio = audio_data[background_mask]
                background_freqs, background_psd = scipy.signal.welch(
                    background_audio, sample_rate
                )

                # Classify background type based on frequency content
                low_freq_energy = np.sum(background_psd[background_freqs < 500])
                mid_freq_energy = np.sum(
                    background_psd[
                        (background_freqs >= 500) & (background_freqs < 2000)
                    ]
                )
                high_freq_energy = np.sum(background_psd[background_freqs >= 2000])

                total_bg_energy = low_freq_energy + mid_freq_energy + high_freq_energy

                if total_bg_energy > 0:
                    freq_profile = {
                        "low_freq_ratio": low_freq_energy / total_bg_energy,
                        "mid_freq_ratio": mid_freq_energy / total_bg_energy,
                        "high_freq_ratio": high_freq_energy / total_bg_energy,
                    }

                    background_type = self._classify_background_type(freq_profile)
                else:
                    freq_profile = {}
                    background_type = "unknown"
            else:
                freq_profile = {}
                background_type = "minimal"

            return {
                "background_percentage": float(background_percentage),
                "background_type": background_type,
                "frequency_profile": freq_profile,
                "noise_level": (
                    "low"
                    if background_percentage < 20
                    else "moderate" if background_percentage < 50 else "high"
                ),
            }

        except Exception as e:
            self.logger.warning(f"Background sound analysis failed: {str(e)}")
            return {"error": str(e)}

    def _classify_background_type(self, freq_profile: Dict[str, float]) -> str:
        """Classify background type based on frequency profile"""
        if freq_profile["low_freq_ratio"] > 0.6:
            return "low_frequency_hum"
        elif freq_profile["high_freq_ratio"] > 0.5:
            return "high_frequency_noise"
        elif freq_profile["mid_freq_ratio"] > 0.4:
            return "environmental_sounds"
        else:
            return "broadband_noise"

    def _compile_analysis_output(self, analysis_results: Dict[str, Any]) -> str:
        """Compile all analysis results into a comprehensive description"""

        output_lines = ["=== UNIVERSAL AUDIO ANALYSIS ===\n"]

        # Audio properties
        props = analysis_results.get("properties", {})
        output_lines.append("AUDIO PROPERTIES:")
        output_lines.append(
            f"- Duration: {props.get('duration_seconds', 'Unknown')} seconds"
        )
        output_lines.append(f"- Sample rate: {props.get('sample_rate', 'Unknown')} Hz")
        output_lines.append(
            f"- Content type: {props.get('estimated_content_type', 'Unknown')}"
        )
        output_lines.append(
            f"- Dynamic range: {props.get('dynamic_range_db', 'Unknown'):.1f} dB"
        )

        # Transcription
        transcription = analysis_results.get("transcription", {})
        output_lines.append("\nTRANSCRIPTION:")
        if transcription.get("has_speech"):
            output_lines.append(f"- Speech detected: YES")
            output_lines.append(
                f"- Engines used: {', '.join(transcription.get('engines_used', []))}"
            )
            output_lines.append(
                f"- Language: {transcription.get('detected_language', 'Unknown')}"
            )
            output_lines.append(
                f"- Text: {transcription.get('transcribed_text', 'None')}"
            )

            # Transcription analysis
            analysis = transcription.get("analysis", {})
            if analysis:
                output_lines.append(f"- Word count: {analysis.get('word_count', 0)}")
                output_lines.append(
                    f"- Speaking rate: {analysis.get('estimated_speaking_rate', 0):.1f} WPM"
                )
        else:
            output_lines.append("- Speech detected: NO")

        # Speaker analysis
        speakers = analysis_results.get("speaker_analysis", {})
        output_lines.append("\nSPEAKER ANALYSIS:")
        if "error" not in speakers:
            output_lines.append(
                f"- Estimated speakers: {speakers.get('estimated_speaker_count', 'Unknown')}"
            )
            voice_chars = speakers.get("voice_characteristics", {})
            if voice_chars:
                output_lines.append(
                    f"- Average pitch: {voice_chars.get('average_pitch_hz', 0):.1f} Hz"
                )
                output_lines.append(
                    f"- Voice type: {voice_chars.get('voice_classification', 'Unknown')}"
                )
                output_lines.append(
                    f"- Speaking rate: {voice_chars.get('estimated_speaking_rate', 0):.1f} onsets/min"
                )
        else:
            output_lines.append(f"- Analysis failed: {speakers['error']}")

        # Audio quality
        quality = analysis_results.get("quality_metrics", {})
        output_lines.append("\nAUDIO QUALITY:")
        if "error" not in quality:
            output_lines.append(
                f"- Quality score: {quality.get('quality_score', 0):.1f}/100"
            )
            output_lines.append(
                f"- Quality level: {quality.get('quality_assessment', 'Unknown')}"
            )
            output_lines.append(
                f"- Signal-to-noise ratio: {quality.get('signal_to_noise_ratio_db', 0):.1f} dB"
            )
            output_lines.append(
                f"- Clipping: {quality.get('clipping_percentage', 0):.1f}%"
            )
        else:
            output_lines.append(f"- Analysis failed: {quality['error']}")

        # Temporal patterns
        temporal = analysis_results.get("temporal_analysis", {})
        output_lines.append("\nTEMPORAL PATTERNS:")
        if "error" not in temporal:
            output_lines.append(
                f"- Speech percentage: {temporal.get('speech_percentage', 0):.1f}%"
            )
            output_lines.append(
                f"- Speech segments: {temporal.get('speech_segments', 0)}"
            )
            output_lines.append(
                f"- Speaking pattern: {temporal.get('speaking_pattern', 'Unknown')}"
            )
            output_lines.append(
                f"- Average speech segment: {temporal.get('average_speech_segment_duration', 0):.1f}s"
            )
        else:
            output_lines.append(f"- Analysis failed: {temporal['error']}")

        # Background analysis
        background = analysis_results.get("background_analysis", {})
        output_lines.append("\nBACKGROUND SOUNDS:")
        if "error" not in background:
            output_lines.append(
                f"- Background level: {background.get('background_percentage', 0):.1f}%"
            )
            output_lines.append(
                f"- Background type: {background.get('background_type', 'Unknown')}"
            )
            output_lines.append(
                f"- Noise level: {background.get('noise_level', 'Unknown')}"
            )
        else:
            output_lines.append(f"- Analysis failed: {background['error']}")

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
async def test_audio_processor():
    """Test the Universal Audio Processor"""
    processor = UniversalAudioProcessor()

    # Test with a sample audio file (you would provide actual audio path)
    test_audio_path = "test_audio.wav"

    if Path(test_audio_path).exists():
        result = await processor.execute(test_audio_path)

        print("Analysis Result:")
        print(f"Success: {result.get('success', False)}")
        print(f"Tool: {result.get('tool_name')}")
        print(
            f"Analysis time: {result.get('metadata', {}).get('analysis_time', 0):.2f}s"
        )
        print("\nRaw Output:")
        print(result.get("raw_output", "No output"))
    else:
        print(f"Test audio not found: {test_audio_path}")


if __name__ == "__main__":
    asyncio.run(test_audio_processor())
