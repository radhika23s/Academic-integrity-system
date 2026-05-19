"""
Video Proctoring Service
Provides face detection, liveness detection, and anomaly detection for proctoring.
"""
import io
import time
import base64
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import structlog
from PIL import Image

logger = structlog.get_logger()


@dataclass
class FaceDetectionResult:
    """Result of face detection analysis."""
    face_count: int = 0
    multiple_faces_detected: bool = False
    no_face_detected: bool = False
    face_locations: List[Dict] = field(default_factory=list)
    face_encodings: List[Any] = field(default_factory=list)
    processing_ms: int = 0


@dataclass
class LivenessResult:
    """Result of liveness detection analysis."""
    is_live: bool = True
    blink_detected: bool = False
    movement_detected: bool = False
    blink_count: int = 0
    avg_eye_aspect_ratio: float = 0.0
    face_movement_score: float = 0.0
    flags: List[str] = field(default_factory=list)
    processing_ms: int = 0


@dataclass
class ProctoringAnalysisResult:
    """Complete proctoring analysis result."""
    face_detection: FaceDetectionResult
    liveness: LivenessResult
    anomalies: List[str] = field(default_factory=list)
    overall_risk_score: float = 0.0
    flags: List[str] = field(default_factory=list)
    evidence: Dict = field(default_factory=dict)
    processing_ms: int = 0


class VideoProcessor:
    """
    Video processing service for proctoring.
    Handles face detection, liveness detection, and anomaly detection.
    Uses MediaPipe for face detection (no CMake required).
    """
    
    def __init__(self):
        self._face_mesh = None
        self._initialized = False
        self._blink_detection_threshold = 0.2
        self._min_blinks_required = 1
        
    async def initialize(self) -> bool:
        """Initialize the video processor and load models."""
        if self._initialized:
            return True
            
        try:
            import mediapipe as mp
            import cv2
            
            # Initialize MediaPipe Face Mesh
            self._mp_face_mesh = mp.solutions.face_mesh
            self._face_mesh = self._mp_face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=5,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            
            # Store MediaPipe drawing utilities
            self._mp_drawing = mp.solutions.drawing_utils
            self._mp_styles = mp.solutions.drawing_styles
            
            self._initialized = True
            logger.info("video_processor.initialized", status="success", backend="mediapipe")
            return True
        except ImportError as e:
            logger.error("video_processor.init_failed", error=str(e))
            return False
        except Exception as e:
            logger.error("video_processor.init_error", error=str(e))
            return False
    
    @property
    def is_healthy(self) -> bool:
        """Check if the video processor is healthy."""
        return self._initialized
    
    def _convert_mediapipe_to_bounding_box(self, landmarks, image_width: int, image_height: int) -> Tuple[int, int, int, int]:
        """Convert MediaPipe landmarks to bounding box (top, right, bottom, left)."""
        # Get all landmark points
        x_coords = [landmark.x for landmark in landmarks.landmark]
        y_coords = [landmark.y for landmark in landmarks.landmark]
        
        # Calculate bounding box
        x_min = int(min(x_coords) * image_width)
        x_max = int(max(x_coords) * image_width)
        y_min = int(min(y_coords) * image_height)
        y_max = int(max(y_coords) * image_height)
        
        return y_min, x_max, y_min, x_max, y_max, x_min  # top, right, bottom, left
    
    async def detect_faces(self, image_data: bytes) -> FaceDetectionResult:
        """
        Detect faces in an image.
        
        Args:
            image_data: Raw image bytes
            
        Returns:
            FaceDetectionResult with face detection details
        """
        t0 = time.monotonic_ns()
        
        try:
            if not self._initialized:
                await self.initialize()
            
            # Load image from bytes
            image = self._load_image(image_data)
            if image is None:
                return FaceDetectionResult(
                    processing_ms=self._elapsed_ms(t0)
                )
            
            # Convert to RGB for MediaPipe
            image_array = np.array(image)
            if image_array.shape[-1] == 4:  # RGBA
                image_rgb = image_array[:, :, :3]
            else:
                image_rgb = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
            
            image_height, image_width = image_rgb.shape[:2]
            
            # Detect faces using MediaPipe
            results = self._face_mesh.process(image_rgb)
            
            face_count = 0
            face_locs = []
            face_encodings = []
            
            if results.multi_face_landmarks:
                face_count = len(results.multi_face_landmarks)
                
                for face_landmarks in results.multi_face_landmarks:
                    # Calculate bounding box from landmarks
                    x_coords = [landmark.x for landmark in face_landmarks.landmark]
                    y_coords = [landmark.y for landmark in face_landmarks.landmark]
                    
                    left = int(min(x_coords) * image_width)
                    right = int(max(x_coords) * image_width)
                    top = int(min(y_coords) * image_height)
                    bottom = int(max(y_coords) * image_height)
                    
                    face_locs.append({
                        "top": top,
                        "right": right,
                        "bottom": bottom,
                        "left": left
                    })
                    
                    # Extract encoding from landmarks (flatten all landmark coordinates)
                    encoding = []
                    for landmark in face_landmarks.landmark:
                        encoding.extend([landmark.x, landmark.y, landmark.z])
                    face_encodings.append(encoding)
            
            result = FaceDetectionResult(
                face_count=face_count,
                multiple_faces_detected=face_count > 1,
                no_face_detected=face_count == 0,
                face_locations=face_locs,
                face_encodings=face_encodings,
                processing_ms=self._elapsed_ms(t0)
            )
            
            logger.debug("face_detection.completed", 
                        face_count=face_count,
                        processing_ms=result.processing_ms)
            
            return result
            
        except Exception as e:
            logger.error("face_detection.failed", error=str(e))
            return FaceDetectionResult(processing_ms=self._elapsed_ms(t0))
    
    async def detect_liveness(self, frame_sequence: List[bytes]) -> LivenessResult:
        """
        Detect liveness from a sequence of video frames.
        
        Analyzes eye blink patterns and facial movements to determine
        if the person is live (not a photo/video spoof).
        
        Args:
            frame_sequence: List of image bytes (frames)
            
        Returns:
            LivenessResult with liveness detection details
        """
        t0 = time.monotonic_ns()
        
        try:
            if not self._initialized:
                await self.initialize()
            
            if len(frame_sequence) < 3:
                return LivenessResult(
                    flags=["insufficient_frames"],
                    processing_ms=self._elapsed_ms(t0)
                )
            
            import cv2
            
            blink_count = 0
            movement_scores = []
            eye_ar_history = []
            
            prev_face_landmarks = None
            prev_bounding_box = None
            
            for frame_data in frame_sequence:
                image = self._load_image(frame_data)
                if image is None:
                    continue
                    
                image_array = np.array(image)
                if image_array.shape[-1] == 4:
                    image_rgb = image_array[:, :, :3]
                else:
                    image_rgb = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
                
                image_height, image_width = image_rgb.shape[:2]
                
                # Get face landmarks
                results = self._face_mesh.process(image_rgb)
                
                if results.multi_face_landmarks:
                    face_landmarks = results.multi_face_landmarks[0]
                    
                    # Get bounding box
                    x_coords = [landmark.x for landmark in face_landmarks.landmark]
                    y_coords = [landmark.y for landmark in face_landmarks.landmark]
                    
                    current_bounding_box = {
                        "left": int(min(x_coords) * image_width),
                        "right": int(max(x_coords) * image_width),
                        "top": int(min(y_coords) * image_height),
                        "bottom": int(max(y_coords) * image_height)
                    }
                    
                    # Calculate movement
                    if prev_bounding_box:
                        movement = self._calculate_movement_mediapipe(
                            prev_bounding_box, current_bounding_box
                        )
                        movement_scores.append(movement)
                    
                    # MediaPipe provides specific eye landmarks (indices 33-133 for left eye, 362-462 for right eye)
                    # Eye landmarks: 33-133 (left), 362-462 (right)
                    # Iris landmarks: 468-473 (left), 474-479 (right)
                    ear = self._estimate_eye_aspect_ratio_mediapipe(
                        face_landmarks, image_width, image_height
                    )
                    if ear > 0:
                        eye_ar_history.append(ear)
                    
                    prev_face_landmarks = face_landmarks
                    prev_bounding_box = current_bounding_box
            
            # Analyze blink pattern
            if len(eye_ar_history) >= 3:
                blink_count = self._detect_blinks(eye_ar_history)
            
            # Analyze movement
            avg_movement = sum(movement_scores) / len(movement_scores) if movement_scores else 0
            movement_detected = avg_movement > 0.05
            
            # Determine liveness
            is_live = True
            flags = []
            
            if blink_count < self._min_blinks_required:
                flags.append("low_blink_count")
                is_live = False
                
            if not movement_detected:
                flags.append("no_movement_detected")
                is_live = False
            
            # Check for static image (photo attack)
            if len(movement_scores) > 0 and max(movement_scores) < 0.03:
                flags.append("possible_static_image")
                is_live = False
            
            avg_ear = sum(eye_ar_history) / len(eye_ar_history) if eye_ar_history else 0
            
            result = LivenessResult(
                is_live=is_live,
                blink_detected=blink_count > 0,
                movement_detected=movement_detected,
                blink_count=blink_count,
                avg_eye_aspect_ratio=round(avg_ear, 4),
                face_movement_score=round(avg_movement, 4),
                flags=flags,
                processing_ms=self._elapsed_ms(t0)
            )
            
            logger.debug("liveness_detection.completed",
                        is_live=is_live,
                        blink_count=blink_count,
                        processing_ms=result.processing_ms)
            
            return result
            
        except Exception as e:
            logger.error("liveness_detection.failed", error=str(e))
            return LivenessResult(processing_ms=self._elapsed_ms(t0))
    
    async def analyze_frames(self, frames: List[bytes]) -> ProctoringAnalysisResult:
        """
        Complete proctoring analysis on a sequence of frames.
        
        Args:
            frames: List of video frame bytes
            
        Returns:
            Complete proctoring analysis result
        """
        t0 = time.monotonic_ns()
        
        # Analyze first frame for face detection
        face_result = FaceDetectionResult()
        if frames:
            face_result = await self.detect_faces(frames[0])
        
        # Analyze sequence for liveness
        liveness_result = await self.detect_liveness(frames)
        
        # Determine anomalies and flags
        anomalies = []
        flags = []
        
        if face_result.multiple_faces_detected:
            anomalies.append("multiple_faces_detected")
            flags.append("HIGH: Multiple faces in frame")
        
        if face_result.no_face_detected:
            anomalies.append("no_face_detected")
            flags.append("MEDIUM: No face detected in frame")
        
        if not liveness_result.is_live:
            anomalies.append("liveness_check_failed")
            flags.append(f"HIGH: Liveness check failed - {', '.join(liveness_result.flags)}")
        
        # Calculate risk score
        risk_score = 0.0
        
        if face_result.multiple_faces_detected:
            risk_score += 0.5
        
        if face_result.no_face_detected:
            risk_score += 0.3
        
        if not liveness_result.is_live:
            risk_score += 0.4
        
        risk_score = min(risk_score, 1.0)
        
        result = ProctoringAnalysisResult(
            face_detection=face_result,
            liveness=liveness_result,
            anomalies=anomalies,
            overall_risk_score=round(risk_score, 4),
            flags=flags,
            evidence={
                "frame_count": len(frames),
                "face_count": face_result.face_count,
                "liveness_flags": liveness_result.flags,
                "movement_score": liveness_result.face_movement_score,
                "blink_count": liveness_result.blink_count,
            },
            processing_ms=self._elapsed_ms(t0)
        )
        
        logger.info("proctoring_analysis.completed",
                  risk_score=risk_score,
                  flags_count=len(flags),
                  processing_ms=result.processing_ms)
        
        return result
    
    async def analyze_screenshot(self, image_data: bytes) -> Dict[str, Any]:
        """
        Analyze a screenshot for anomalies.
        
        Args:
            image_data: Screenshot bytes
            
        Returns:
            Analysis result dictionary
        """
        t0 = time.monotonic_ns()
        
        try:
            # Perform face detection
            face_result = await self.detect_faces(image_data)
            
            # Load image for additional analysis
            image = self._load_image(image_data)
            if image is None:
                return {"error": "Failed to load image", "processing_ms": self._elapsed_ms(t0)}
            
            # Analyze image quality
            quality_result = self._analyze_image_quality(image)
            
            # Build result
            result = {
                "face_detection": {
                    "face_count": face_result.face_count,
                    "multiple_faces": face_result.multiple_faces_detected,
                    "no_face": face_result.no_face_detected,
                },
                "image_quality": quality_result,
                "risk_flags": [],
                "processing_ms": self._elapsed_ms(t0)
            }
            
            # Add risk flags
            if face_result.multiple_faces_detected:
                result["risk_flags"].append("multiple_faces")
            
            if face_result.no_face_detected:
                result["risk_flags"].append("no_face")
            
            if not quality_result["is_acceptable"]:
                result["risk_flags"].append(f"quality_issues: {quality_result['issues']}")
            
            return result
            
        except Exception as e:
            logger.error("screenshot_analysis.failed", error=str(e))
            return {"error": str(e), "processing_ms": self._elapsed_ms(t0)}
    
    def _load_image(self, image_data: bytes) -> Optional[Image.Image]:
        """Load image from bytes."""
        try:
            return Image.open(io.BytesIO(image_data))
        except Exception as e:
            logger.error("image_load.failed", error=str(e))
            return None
    
    def _calculate_movement_mediapipe(
        self,
        prev_bbox: Dict,
        curr_bbox: Dict
    ) -> float:
        """Calculate facial movement score between frames using MediaPipe bounding boxes."""
        if not prev_bbox or not curr_bbox:
            return 0.0
        
        # Calculate center movement
        prev_center_x = (prev_bbox["left"] + prev_bbox["right"]) / 2
        prev_center_y = (prev_bbox["top"] + prev_bbox["bottom"]) / 2
        curr_center_x = (curr_bbox["left"] + curr_bbox["right"]) / 2
        curr_center_y = (curr_bbox["top"] + curr_bbox["bottom"]) / 2
        
        # Calculate face size change
        prev_width = prev_bbox["right"] - prev_bbox["left"]
        prev_height = prev_bbox["bottom"] - prev_bbox["top"]
        
        # Normalize by average face size
        avg_size = (prev_width + prev_height) / 2
        
        if avg_size == 0:
            return 0.0
        
        distance = ((curr_center_x - prev_center_x)**2 + 
                   (curr_center_y - prev_center_y)**2)**0.5 / avg_size
        
        return min(distance, 1.0)
    
    def _estimate_eye_aspect_ratio_mediapipe(self, landmarks, image_width: int, image_height: int) -> float:
        """
        Estimate eye aspect ratio from MediaPipe face landmarks.
        Uses eye landmark indices: 33-133 (left eye), 362-462 (right eye)
        More accurate than the previous simplified approach.
        """
        try:
            # Left eye landmarks (indices 33, 133, 160, 158, 153, 144)
            # Right eye landmarks (indices 362, 263, 387, 385, 380, 373)
            LEFT_EYE = [33, 133, 160, 158, 153, 144]
            RIGHT_EYE = [362, 263, 387, 385, 380, 373]
            
            def calculate_ear(eye_indices):
                # Get landmark coordinates
                points = []
                for idx in eye_indices:
                    landmark = landmarks.landmark[idx]
                    points.append((landmark.x * image_width, landmark.y * image_height))
                
                # Calculate EAR: (|p2-p6| + |p3-p5|) / (2 * |p1-p4|)
                # Using vertical and horizontal distances
                p1, p2, p3, p4, p5, p6 = points
                
                # Vertical distances
                vertical_1 = ((p2[0] - p6[0])**2 + (p2[1] - p6[1])**2)**0.5
                vertical_2 = ((p3[0] - p5[0])**2 + (p3[1] - p5[1])**2)**0.5
                
                # Horizontal distance
                horizontal = ((p1[0] - p4[0])**2 + (p1[1] - p4[1])**2)**0.5
                
                if horizontal == 0:
                    return 0.0
                
                ear = (vertical_1 + vertical_2) / (2 * horizontal)
                return ear
            
            left_ear = calculate_ear(LEFT_EYE)
            right_ear = calculate_ear(RIGHT_EYE)
            
            # Average EAR from both eyes
            avg_ear = (left_ear + right_ear) / 2
            
            # Scale to reasonable EAR range (typically 0.2-0.35 for open eyes)
            # MediaPipe gives slightly different scale, so we normalize
            return min(avg_ear * 1.5, 0.4)
            
        except Exception as e:
            logger.debug("eye_aspect_ratio.failed", error=str(e))
            return 0.0
    
    def _detect_blinks(self, eye_ar_history: List[float]) -> int:
        """Detect number of blinks from eye aspect ratio history."""
        if len(eye_ar_history) < 3:
            return 0
        
        blink_count = 0
        in_blink = False
        
        # Find local minima (potential blinks)
        for i in range(1, len(eye_ar_history) - 1):
            if (eye_ar_history[i] < eye_ar_history[i-1] and 
                eye_ar_history[i] < eye_ar_history[i+1] and
                eye_ar_history[i] < self._blink_detection_threshold):
                
                if not in_blink:
                    blink_count += 1
                    in_blink = True
            else:
                in_blink = False
        
        return blink_count
    
    def _analyze_image_quality(self, image: Image.Image) -> Dict[str, Any]:
        """Analyze image quality for proctoring."""
        import cv2
        
        issues = []
        is_acceptable = True
        
        # Convert to numpy array
        img_array = np.array(image)
        
        # Check if grayscale
        if len(img_array.shape) == 2:
            gray = img_array
        else:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        
        # Calculate brightness
        mean_brightness = np.mean(gray)
        if mean_brightness < 50:
            issues.append("too_dark")
            is_acceptable = False
        elif mean_brightness > 230:
            issues.append("too_bright")
            is_acceptable = False
        
        # Calculate contrast
        std_contrast = np.std(gray)
        if std_contrast < 20:
            issues.append("low_contrast")
            is_acceptable = False
        
        # Check blur (Laplacian variance)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        blur_score = laplacian.var()
        if blur_score < 100:
            issues.append("blurry")
            is_acceptable = False
        
        return {
            "is_acceptable": is_acceptable,
            "issues": issues,
            "brightness": round(mean_brightness, 2),
            "contrast": round(std_contrast, 2),
            "blur_score": round(blur_score, 2),
        }
    
    def _elapsed_ms(self, t0: int) -> int:
        """Calculate elapsed time in milliseconds."""
        return int((time.monotonic_ns() - t0) / 1_000_000)

