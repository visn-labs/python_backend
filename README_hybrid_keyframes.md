Hybrid Keyframe Extraction & Streaming Ingest
===========================================

This module provides a hybrid (motion + texture) keyframe extraction pipeline plus optional
live stream ingestion using ffmpeg. You can either:

1. Provide a local MP4 via `video_path`.
2. Provide a streaming URL (e.g. MJPEG, RTSP) and let the API capture frames first.

Boolean API (Simplified)
------------------------
Endpoint: `POST /keyframes/extract`

Minimal JSON now:
```
{
	"stream": false,            # false=file path; true=use streaming
	"segment_capture": false,   # when stream=true: false=realtime; true=segment/periodic
	"video_path": "C:/videos/sample.mp4",  # required if stream=false
	"stream_url": "rtsp://camera/stream1"  # required if stream=true
}
```

Advanced streaming parameters: `config/streaming.yml` (segment_seconds, mpdecimate, reencode,
wait_initial_seconds, max_wait_seconds, realtime_run_seconds, realtime_max_frames, stream_fps).

Configuration
-------------
Edit `config/base.yml` for core extraction parameters.
Edit `config/streaming.yml` for streaming-specific knobs (capture strategy, wait times, realtime limits).

Example Requests
----------------
File (upload path):
```
POST /keyframes/extract
{ "stream": false, "video_path": "C:/videos/sample.mp4" }
```

Stream realtime:
```
POST /keyframes/extract
{ "stream": true, "segment_capture": false, "stream_url": "rtsp://camera/stream1" }
```

Stream periodic segmented capture:
```
POST /keyframes/extract
{ "stream": true, "segment_capture": true, "stream_url": "http://97.86.89.114:2222/mjpg/video.mjpg" }
```

Processing Flow
---------------
If stream=false:
1. Run pipeline directly on provided file.

If stream=true and segment_capture=false (Realtime):
1. Open stream; process frames live; write only keyframes.

If stream=true and segment_capture=true (Segmented Periodic):
1. ffmpeg captures segments/rolling file.
2. After first usable segment found, capture stops and pipeline processes saved video.

ffmpeg Filters
--------------
The `mpdecimate` filter removes frames that are visually similar to previous ones, reducing
redundancy and downstream processing. Thresholds can be tuned by editing the filter string in
`stream_ingest.py` if needed.

Operational Notes
-----------------
* Ensure `ffmpeg` is installed and on PATH.
* For very long-running continuous capture + repeated extraction, consider an external scheduler
	and persisting the ingest process outside this request lifecycle.
* Segment mode is useful for near-real-time batch processing windows.

Future Enhancements
-------------------
* Rolling window re-processing.
* WebSocket updates on new segments.
* Configurable mpdecimate parameters via API.

