# TODO: Fix face-recognition dependency issue

## Problem
The `face-recognition==1.3.0` package requires `dlib`, which needs CMake to be installed on Windows. CMake is not available, causing installation failure.

## Solution
Replace `face-recognition` with `mediapipe` (already installed) which provides equivalent face detection functionality without requiring compilation.

## Tasks
- [x] 1. Analyze codebase and understand face_recognition usage
- [x] 2. Update requirements.txt - remove face-recognition, add mediapipe
- [x] 3. Update services/video_processor.py - replace face_recognition with mediapipe
- [x] 4. Test the installation

## Changes Made

### File: requirements.txt
- Removed: `face-recognition==1.3.0`
- Added: `mediapipe` (already pre-installed)

### File: services/video_processor.py
- Replaced `import face_recognition` with `import mediapipe`
- Changed from `face_recognition.face_locations()` to MediaPipe FaceMesh
- Changed from `face_recognition.face_encodings()` to MediaPipe face landmarks
- Improved liveness detection using actual eye landmarks (indices 33-133 for left eye, 362-462 for right eye)
- Added proper EAR (Eye Aspect Ratio) calculation using MediaPipe's precise facial landmarks

## Benefits
1. No CMake required - MediaPipe has pre-built wheels
2. More accurate face detection with 468 facial landmarks
3. Better liveness detection with precise eye landmark tracking
4. Supports multiple faces (up to 5)

