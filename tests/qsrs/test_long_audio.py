"""Tests for tfmx.qsrs.long_audio."""

from tfmx.qsrs.client import InfoResponse
from tfmx.qsrs.long_audio import ChunkTranscriptionResult
from tfmx.qsrs.long_audio import SilenceRegion
from tfmx.qsrs.long_audio import count_available_healthy_slots
from tfmx.qsrs.long_audio import count_idle_healthy_instances
from tfmx.qsrs.long_audio import count_schedulable_healthy_slots
from tfmx.qsrs.long_audio import merge_transcribed_chunks
from tfmx.qsrs.long_audio import plan_audio_chunks


class TestLongAudioPlanning:
    def test_plan_audio_chunks_prefers_nearby_silence(self):
        chunks = plan_audio_chunks(
            170.0,
            silence_regions=[
                SilenceRegion(57.0, 59.0),
                SilenceRegion(117.0, 120.0),
            ],
            target_chunk_sec=60.0,
            min_chunk_sec=35.0,
            max_chunk_sec=90.0,
            overlap_sec=4.0,
            search_window_sec=8.0,
        )

        assert len(chunks) == 3
        assert round(chunks[0].end_sec, 1) == 58.0
        assert round(chunks[1].start_sec, 1) == 54.0
        assert round(chunks[1].end_sec, 1) == 118.5

    def test_plan_audio_chunks_sets_keep_boundaries(self):
        chunks = plan_audio_chunks(
            130.0,
            silence_regions=[],
            target_chunk_sec=60.0,
            min_chunk_sec=40.0,
            max_chunk_sec=70.0,
            overlap_sec=5.0,
            search_window_sec=8.0,
        )

        assert len(chunks) == 2
        assert chunks[0].keep_start_sec == 0.0
        assert chunks[-1].keep_end_sec == 130.0
        assert chunks[1].keep_start_sec > chunks[1].start_sec
        assert chunks[0].keep_end_sec < chunks[0].end_sec


class TestLongAudioMerging:
    def test_merge_transcribed_chunks_uses_segments_and_discards_overlap(self):
        chunks = plan_audio_chunks(
            122.0,
            silence_regions=[],
            target_chunk_sec=60.0,
            min_chunk_sec=40.0,
            max_chunk_sec=70.0,
            overlap_sec=4.0,
            search_window_sec=8.0,
        )
        first, second = chunks[0], chunks[1]

        merged_text, language, merged_segments = merge_transcribed_chunks(
            chunks,
            [
                ChunkTranscriptionResult(
                    chunk_index=first.index,
                    start_sec=first.start_sec,
                    end_sec=first.end_sec,
                    keep_start_sec=first.keep_start_sec,
                    keep_end_sec=first.keep_end_sec,
                    latency_sec=1.0,
                    attempt=1,
                    text="甲乙丙丁",
                    language="zh",
                    duration_sec=first.duration_sec,
                    segments=[
                        {"start": 0.0, "end": 25.0, "text": "甲乙"},
                        {"start": 54.0, "end": 60.0, "text": "重叠"},
                    ],
                ),
                ChunkTranscriptionResult(
                    chunk_index=second.index,
                    start_sec=second.start_sec,
                    end_sec=second.end_sec,
                    keep_start_sec=second.keep_start_sec,
                    keep_end_sec=second.keep_end_sec,
                    latency_sec=1.0,
                    attempt=1,
                    text="重叠戊己",
                    language="zh",
                    duration_sec=second.duration_sec,
                    segments=[
                        {"start": 0.0, "end": 6.0, "text": "重叠"},
                        {"start": 6.0, "end": 30.0, "text": "戊己"},
                    ],
                ),
            ],
        )

        assert language == "zh"
        assert merged_text == "甲乙重叠戊己"
        assert len(merged_segments) == 3

    def test_merge_transcribed_chunks_dedups_normalized_boundary_text(self):
        chunks = plan_audio_chunks(
            122.0,
            silence_regions=[],
            target_chunk_sec=60.0,
            min_chunk_sec=40.0,
            max_chunk_sec=70.0,
            overlap_sec=4.0,
            search_window_sec=8.0,
        )
        first, second = chunks[0], chunks[1]

        merged_text, language, merged_segments = merge_transcribed_chunks(
            chunks,
            [
                ChunkTranscriptionResult(
                    chunk_index=first.index,
                    start_sec=first.start_sec,
                    end_sec=first.end_sec,
                    keep_start_sec=first.keep_start_sec,
                    keep_end_sec=first.keep_end_sec,
                    latency_sec=1.0,
                    attempt=1,
                    text="你好，世界",
                    language="zh",
                    duration_sec=first.duration_sec,
                ),
                ChunkTranscriptionResult(
                    chunk_index=second.index,
                    start_sec=second.start_sec,
                    end_sec=second.end_sec,
                    keep_start_sec=second.keep_start_sec,
                    keep_end_sec=second.keep_end_sec,
                    latency_sec=1.0,
                    attempt=1,
                    text="世界！今天继续。",
                    language="zh",
                    duration_sec=second.duration_sec,
                ),
            ],
        )

        assert language == "zh"
        assert merged_segments == []
        assert merged_text == "你好，世界！今天继续。"

    def test_count_idle_healthy_instances(self):
        info = InfoResponse.from_dict(
            {
                "port": 27900,
                "instances": [
                    {
                        "name": "gpu0",
                        "endpoint": "http://localhost:27980",
                        "healthy": True,
                        "sleeping": False,
                        "active_requests": 0,
                        "available_slots": 8,
                        "scheduler": {},
                    },
                    {
                        "name": "gpu1",
                        "endpoint": "http://localhost:27981",
                        "healthy": True,
                        "sleeping": False,
                        "active_requests": 1,
                        "available_slots": 7,
                        "scheduler": {},
                    },
                    {
                        "name": "gpu2",
                        "endpoint": "http://localhost:27982",
                        "healthy": False,
                        "sleeping": False,
                        "active_requests": 0,
                        "available_slots": 8,
                        "scheduler": {},
                    },
                ],
                "stats": {},
                "available_models": [],
                "scheduler": {},
            }
        )

        assert count_idle_healthy_instances(info) == 1
        assert count_available_healthy_slots(info) == 15

    def test_count_schedulable_healthy_slots_respects_per_instance_cap(self):
        info = InfoResponse.from_dict(
            {
                "port": 27900,
                "instances": [
                    {
                        "name": "gpu0",
                        "endpoint": "http://localhost:27980",
                        "healthy": True,
                        "sleeping": False,
                        "active_requests": 0,
                        "available_slots": 8,
                        "scheduler": {},
                    },
                    {
                        "name": "gpu1",
                        "endpoint": "http://localhost:27981",
                        "healthy": True,
                        "sleeping": False,
                        "active_requests": 1,
                        "available_slots": 7,
                        "scheduler": {},
                    },
                    {
                        "name": "gpu2",
                        "endpoint": "http://localhost:27982",
                        "healthy": True,
                        "sleeping": False,
                        "active_requests": 3,
                        "available_slots": 5,
                        "scheduler": {},
                    },
                ],
                "stats": {},
                "available_models": [],
                "scheduler": {},
            }
        )

        assert count_schedulable_healthy_slots(info) == 20
        assert (
            count_schedulable_healthy_slots(
                info,
                per_instance_parallelism_cap=2,
            )
            == 3
        )
