API Layer
=========

Run the FastAPI app:

```
uvicorn src.api.app:app --reload --port 8000
```

Key Endpoints
-------------

1. POST /keyframes/extract
   - Body: KeyframeConfig (video_path required)
   - Returns extracted keyframe timestamps.

2. POST /index/build
   - Body: IndexConfig (keyframes_dir required)
   - Builds vector index. Optionally sets global in-memory store.

3. POST /query
   - Body: QueryRequest (user_query required)
   - Runs semantic query over previously built in-memory store.

4. POST /unified/run
   - Body: UnifiedRequest (optional user_query, video_path)
   - Executes end-to-end pipeline according to config file (default resources/unified.yml).

5. GET /health
   - Simple health check.

Development Notes
-----------------

Global in-memory store is used for brevity. For production, replace with a dependency-injected persistence layer.

Ensure OpenCV can access video files; place a sample video at the configured path or override via UnifiedRequest.video_path.
