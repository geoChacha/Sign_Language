# Test Fixtures

This directory contains test data for the WLASL-100 model integration tests.

## Contents

### Sample Videos
- `sample_video_5s.mp4` - 5-second ASL sign video (placeholder)
- `sample_video_15s.mp4` - 15-second ASL sign video (placeholder)
- `sample_video_30s.mp4` - 30-second ASL sign video (placeholder)
- `corrupted_video.mp4` - Corrupted video file for error testing (placeholder)

### Sample Frames
- `clear_background.jpg` - Frame with clear background
- `cluttered_background.jpg` - Frame with cluttered background
- `full_body_visible.jpg` - Frame with full body visible
- `partial_body.jpg` - Frame with partial body visible
- `no_body.jpg` - Frame with no body visible

## Note

The actual video and image files are not included in the repository due to size constraints.
For testing, you can:
1. Record your own ASL sign videos
2. Capture webcam frames for validation testing
3. Use the dummy data generators in the test files

## Dummy Data Generation

The test files include functions to generate dummy data:
- `create_dummy_video()` - Creates a simple test video
- `create_dummy_frame()` - Creates a test image frame
- `create_dummy_keypoints()` - Creates random keypoint sequences
