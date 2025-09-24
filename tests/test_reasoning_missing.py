from src.main.semantic_query.reasoning import reasoning_filter
from src.main.semantic_index.vector_store import VectorRecord

def test_reasoning_skips_missing(tmp_path):
    # Create one existing and one missing record
    existing = tmp_path / 'kf_1_00_color.jpg'
    existing.write_bytes(b'fake')
    rec_existing = VectorRecord(id='kf_1_00_color.jpg', embedding=None, metadata={'video_timestamp':1.0,'abs_path':str(existing)})  # type: ignore
    rec_missing = VectorRecord(id='kf_2_00_color.jpg', embedding=None, metadata={'video_timestamp':2.0,'abs_path':str(tmp_path / 'kf_2_00_color.jpg')})  # type: ignore

    confirmed, audit = reasoning_filter([rec_existing, rec_missing], question='Is there motion?', enable=True, provider='stub', model='stub', keyframes_dir=str(tmp_path))

    # Existing should confirm (stub returns True), missing skipped
    assert any(a['decision'].startswith('Skipped') for a in audit)
    assert any(a['decision'] == 'Yes' for a in audit)
    assert len(confirmed) == 1
