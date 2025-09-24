"""Temporal clustering of confirmed keyframes into events."""
from __future__ import annotations
from typing import List, Dict, Any
from dataclasses import dataclass
from datetime import datetime

@dataclass
class Event:
    event_id: int
    start_time: str
    end_time: str
    count: int
    video_timestamps: List[float]


def within_window(meta: Dict[str, Any], windows: List[tuple[str, str]]) -> bool:
    if not windows:
        return True
    rt = meta.get('real_world_time')
    if not rt:
        return False
    # Expect ISO format
    try:
        tpart = rt.split('T')[1] if 'T' in rt else rt.split()[1]
    except Exception:
        return False
    for s, e in windows:
        if s <= tpart <= e:
            return True
    return False


def cluster_timestamps(records, windows: List[tuple[str, str]], max_gap: float, min_frames: int) -> List[Event]:
    # Filter by time windows first
    filtered = [r for r in records if within_window(r.metadata, windows)]
    if not filtered:
        return []
    # Sort by video timestamp
    filtered.sort(key=lambda r: r.metadata['video_timestamp'])
    events: List[Event] = []
    current: List = []
    last_ts = None
    for r in filtered:
        ts = r.metadata['video_timestamp']
        if last_ts is None or ts - last_ts <= max_gap:
            current.append(r)
        else:
            if len(current) >= min_frames:
                events.append(_make_event(len(events)+1, current))
            current = [r]
        last_ts = ts
    if current and len(current) >= min_frames:
        events.append(_make_event(len(events)+1, current))
    return events


def _make_event(eid: int, recs) -> Event:
    start_ts = recs[0].metadata['real_world_time'] if 'real_world_time' in recs[0].metadata else None
    end_ts = recs[-1].metadata['real_world_time'] if 'real_world_time' in recs[-1].metadata else None
    return Event(
        event_id=eid,
        start_time=start_ts or '',
        end_time=end_ts or '',
        count=len(recs),
        video_timestamps=[r.metadata['video_timestamp'] for r in recs],
    )

__all__ = ['Event', 'cluster_timestamps']
