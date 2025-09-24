from src.main.keyframe_extractor.streaming.stream_ingest import StreamIngestConfig, StreamIngestor

def test_build_commands():
    cfg = StreamIngestConfig(url='http://example/stream.mjpg', output_base='out/test.mp4', mode='continuous', mpdecimate=True)
    ing = StreamIngestor(cfg)
    cmd = ing._continuous_cmd()
    assert '-vf' in cmd and 'mpdecimate' in cmd
    cfg2 = StreamIngestConfig(url='rtsp://cam/stream', output_base='out/seg.mp4', mode='segment', segment_seconds=30, mpdecimate=False)
    ing2 = StreamIngestor(cfg2)
    cmd2 = ing2._segment_cmd()
    assert 'segment_time' in ' '.join(cmd2)
