"""Tests for tfmx.qsrs.long_audio."""

from unittest.mock import patch

import httpx

from tfmx.qsrs.long_audio import AudioChunk
from tfmx.qsrs.long_audio import AudioMetadata
from tfmx.qsrs.long_audio import ChunkRequestPlan
from tfmx.qsrs.client import InfoResponse
from tfmx.qsrs.long_audio import ChunkTranscriptionResult
from tfmx.qsrs.long_audio import LongAudioTranscriptionResult
from tfmx.qsrs.long_audio import LongAudioTranscriber
from tfmx.qsrs.long_audio import QualityFallbackTrace
from tfmx.qsrs.long_audio import SchedulingDiagnostics
from tfmx.qsrs.long_audio import SilenceRegion
from tfmx.qsrs.long_audio import _repair_pathological_repetition
from tfmx.qsrs.long_audio import _targeted_edge_resplit_side
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

    def test_merge_transcribed_chunks_dedups_similar_boundary_sentences(self):
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
                    text="首先我必须得说一下，这个一七年巴西我去了。",
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
                    text=(
                        "我们聊一下。首先，我们必须得说一下这个一七年巴西，我去了。"
                        "首先，那个舟车劳顿旅途真的很辛苦。"
                    ),
                    language="zh",
                    duration_sec=second.duration_sec,
                ),
            ],
        )

        assert language == "zh"
        assert merged_segments == []
        assert merged_text == (
            "首先我必须得说一下，这个一七年巴西我去了。"
            "首先，那个舟车劳顿旅途真的很辛苦。"
        )

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


class TestLongAudioResponseFormat:
    def test_resolve_chunk_request_plan_falls_back_from_verbose_json_400(self):
        transcriber = LongAudioTranscriber("http://127.0.0.1:27991")
        chunk = AudioChunk(index=0, start_sec=0.0, end_sec=10.0)
        request = httpx.Request(
            "POST",
            "http://127.0.0.1:27991/v1/audio/transcriptions",
        )
        response = httpx.Response(400, request=request, text='{"detail":"unsupported"}')

        with patch.object(
            LongAudioTranscriber,
            "_transcribe_chunk_with_plan",
            side_effect=httpx.HTTPStatusError(
                "unsupported verbose_json",
                request=request,
                response=response,
            ),
        ):
            request_plan, probe_result = transcriber._resolve_chunk_request_plan(
                "/tmp/input.wav",
                "/tmp/work",
                chunk,
            )

        assert request_plan.response_format == "json"
        assert probe_result is None

    def test_resolve_chunk_request_plan_keeps_segment_probe_result(self):
        transcriber = LongAudioTranscriber("http://127.0.0.1:27992")
        chunk = AudioChunk(index=0, start_sec=0.0, end_sec=10.0)
        probe_result = ChunkTranscriptionResult(
            chunk_index=0,
            start_sec=0.0,
            end_sec=10.0,
            keep_start_sec=0.0,
            keep_end_sec=10.0,
            latency_sec=0.5,
            attempt=1,
            text="你好，世界。",
            language="zh",
            duration_sec=10.0,
            segments=[{"start": 0.0, "end": 2.0, "text": "你好"}],
        )

        with patch.object(
            LongAudioTranscriber,
            "_transcribe_chunk_with_plan",
            return_value=probe_result,
        ):
            request_plan, resolved_probe_result = (
                transcriber._resolve_chunk_request_plan(
                    "/tmp/input.wav",
                    "/tmp/work",
                    chunk,
                )
            )

        assert request_plan.response_format == "verbose_json"
        assert resolved_probe_result is probe_result


class TestLongAudioScheduling:
    def test_dispatch_pending_chunks_fills_available_capacity(self):
        transcriber = LongAudioTranscriber("http://127.0.0.1:27993")
        metadata = AudioMetadata(path="/tmp/input.wav", duration_sec=30.0)
        pending = [
            AudioChunk(index=1, start_sec=10.0, end_sec=20.0),
            AudioChunk(index=2, start_sec=20.0, end_sec=30.0),
        ]
        running = {object(): (AudioChunk(index=0, start_sec=0.0, end_sec=10.0), 1)}
        attempts = {0: 1, 1: 0, 2: 0}
        request_plan = ChunkRequestPlan()

        class FakeExecutor:
            def __init__(self):
                self.calls: list[tuple[int, int]] = []

            def submit(self, func, audio_path, work_dir, chunk, attempt, plan):
                del func
                del audio_path
                del work_dir
                del plan
                self.calls.append((chunk.index, attempt))
                return object()

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
                        "available_slots": 1,
                        "scheduler": {},
                    },
                    {
                        "name": "gpu1",
                        "endpoint": "http://localhost:27981",
                        "healthy": True,
                        "sleeping": False,
                        "active_requests": 0,
                        "available_slots": 1,
                        "scheduler": {},
                    },
                ],
                "stats": {},
                "available_models": [],
                "scheduler": {},
            }
        )
        executor = FakeExecutor()
        diagnostics = SchedulingDiagnostics()

        dispatched = transcriber._dispatch_pending_chunks(
            executor=executor,
            metadata=metadata,
            work_dir="/tmp",
            pending=pending,
            running=running,
            attempts=attempts,
            request_plan=request_plan,
            executor_parallelism_limit=2,
            chunk_count=3,
            get_latest_info=lambda force=False: info,
            diagnostics=diagnostics,
        )

        assert dispatched == 1
        assert executor.calls == [(1, 1)]
        assert attempts[1] == 1
        assert len(running) == 2
        assert [chunk.index for chunk in pending] == [2]
        assert diagnostics.to_dict() == {
            "dispatch_cycle_count": 1,
            "forced_machine_refresh_count": 0,
            "forced_machine_refresh_hit_count": 0,
            "completion_refill_trigger_count": 0,
            "completion_refill_hit_count": 0,
            "completion_refill_dispatched_chunk_count": 0,
            "idle_poll_wait_count": 0,
            "wait_timeout_count": 0,
        }

    def test_dispatch_pending_chunks_counts_forced_refresh_hits(self):
        transcriber = LongAudioTranscriber("http://127.0.0.1:27994")
        transcriber.config.max_parallel_chunks = 2
        metadata = AudioMetadata(path="/tmp/input.wav", duration_sec=30.0)
        pending = [AudioChunk(index=0, start_sec=0.0, end_sec=10.0)]
        running = {object(): (AudioChunk(index=1, start_sec=10.0, end_sec=20.0), 1)}
        attempts = {0: 0, 1: 1}
        request_plan = ChunkRequestPlan()
        diagnostics = SchedulingDiagnostics()

        class FakeExecutor:
            def __init__(self):
                self.calls: list[tuple[int, int]] = []

            def submit(self, func, audio_path, work_dir, chunk, attempt, plan):
                del func
                del audio_path
                del work_dir
                del plan
                self.calls.append((chunk.index, attempt))
                return object()

        stale_info = InfoResponse.from_dict(
            {
                "port": 27900,
                "instances": [
                    {
                        "name": "gpu0",
                        "endpoint": "http://localhost:27980",
                        "healthy": True,
                        "sleeping": False,
                        "active_requests": 1,
                        "available_slots": 0,
                        "scheduler": {},
                    }
                ],
                "stats": {},
                "available_models": [],
                "scheduler": {},
            }
        )
        refreshed_info = InfoResponse.from_dict(
            {
                "port": 27900,
                "instances": [
                    {
                        "name": "gpu0",
                        "endpoint": "http://localhost:27980",
                        "healthy": True,
                        "sleeping": False,
                        "active_requests": 0,
                        "available_slots": 2,
                        "scheduler": {},
                    }
                ],
                "stats": {},
                "available_models": [],
                "scheduler": {},
            }
        )

        def latest_info(force: bool = False):
            return refreshed_info if force else stale_info

        executor = FakeExecutor()

        dispatched = transcriber._dispatch_pending_chunks(
            executor=executor,
            metadata=metadata,
            work_dir="/tmp",
            pending=pending,
            running=running,
            attempts=attempts,
            request_plan=request_plan,
            executor_parallelism_limit=2,
            chunk_count=2,
            get_latest_info=latest_info,
            diagnostics=diagnostics,
        )

        assert dispatched == 1
        assert executor.calls == [(0, 1)]
        assert diagnostics.forced_machine_refresh_count == 1
        assert diagnostics.forced_machine_refresh_hit_count == 1


class TestLongAudioResultSerialization:
    def test_to_dict_includes_scheduling_diagnostics(self):
        result = LongAudioTranscriptionResult(
            audio="/tmp/input.wav",
            metadata=AudioMetadata(path="/tmp/input.wav", duration_sec=12.0),
            chunks=[AudioChunk(index=0, start_sec=0.0, end_sec=12.0)],
            silence_regions=[],
            chunk_results=[],
            total_time_sec=3.0,
            max_parallel_chunks=2,
            endpoint="http://127.0.0.1:27900",
            machine_snapshot={"healthy_instances": 1},
            scheduling_diagnostics=SchedulingDiagnostics(
                completion_refill_trigger_count=2,
                completion_refill_hit_count=1,
                completion_refill_dispatched_chunk_count=3,
                forced_machine_refresh_count=4,
                forced_machine_refresh_hit_count=2,
            ),
            text="ok",
            language="zh",
        )

        payload = result.to_dict()

        assert payload["scheduling"]["diagnostics"] == {
            "dispatch_cycle_count": 0,
            "forced_machine_refresh_count": 4,
            "forced_machine_refresh_hit_count": 2,
            "completion_refill_trigger_count": 2,
            "completion_refill_hit_count": 1,
            "completion_refill_dispatched_chunk_count": 3,
            "idle_poll_wait_count": 0,
            "wait_timeout_count": 0,
        }


class TestLongAudioQualityFallback:
    def test_transcribe_chunk_splits_pathologically_repetitive_output(self):
        transcriber = LongAudioTranscriber("http://127.0.0.1:27900")
        chunk = AudioChunk(
            index=7,
            start_sec=100.0,
            end_sec=142.0,
            keep_start_sec=100.0,
            keep_end_sec=142.0,
        )

        def side_effect(
            audio_path: str,
            work_dir: str,
            requested_chunk: AudioChunk,
            attempt: int,
            request_plan: ChunkRequestPlan,
        ) -> ChunkTranscriptionResult:
            del audio_path
            del work_dir
            del request_plan
            if requested_chunk.start_sec == 100.0 and requested_chunk.end_sec == 142.0:
                return ChunkTranscriptionResult(
                    chunk_index=requested_chunk.index,
                    start_sec=requested_chunk.start_sec,
                    end_sec=requested_chunk.end_sec,
                    keep_start_sec=requested_chunk.keep_start_sec,
                    keep_end_sec=requested_chunk.keep_end_sec,
                    latency_sec=0.5,
                    attempt=attempt,
                    text=("九十年代，我们是被苏联的猎人。" * 12).strip(),
                    language="zh",
                    duration_sec=requested_chunk.duration_sec,
                )
            if requested_chunk.keep_start_sec <= 100.0:
                text = "前半段。"
            else:
                text = "后半段。"
            return ChunkTranscriptionResult(
                chunk_index=requested_chunk.index,
                start_sec=requested_chunk.start_sec,
                end_sec=requested_chunk.end_sec,
                keep_start_sec=requested_chunk.keep_start_sec,
                keep_end_sec=requested_chunk.keep_end_sec,
                latency_sec=0.5,
                attempt=attempt,
                text=text,
                language="zh",
                duration_sec=requested_chunk.duration_sec,
            )

        with patch.object(
            LongAudioTranscriber,
            "_transcribe_chunk_with_plan",
            side_effect=side_effect,
        ) as mock_transcribe:
            result = transcriber._transcribe_chunk(
                "/tmp/input.wav",
                "/tmp/work",
                chunk,
                1,
                ChunkRequestPlan(),
            )

        assert result.text == "前半段。后半段。"
        assert result.language == "zh"
        assert mock_transcribe.call_count == 3
        assert result.quality_fallback.pathological_repetition_detected is True
        assert result.quality_fallback.split_fallback_used is True
        assert result.quality_fallback.repair_only_accepted is False
        assert result.quality_fallback.max_split_depth == 1
        assert result.quality_fallback.total_transcribe_requests == 3

    def test_transcribe_chunk_collapses_repetition_when_split_is_exhausted(self):
        transcriber = LongAudioTranscriber("http://127.0.0.1:27900")
        chunk = AudioChunk(
            index=8,
            start_sec=200.0,
            end_sec=212.0,
            keep_start_sec=200.0,
            keep_end_sec=212.0,
        )
        repetitive_text = ("九十年代，我们是被苏联的猎人。" * 12).strip()

        with patch.object(
            LongAudioTranscriber,
            "_transcribe_chunk_with_plan",
            return_value=ChunkTranscriptionResult(
                chunk_index=chunk.index,
                start_sec=chunk.start_sec,
                end_sec=chunk.end_sec,
                keep_start_sec=chunk.keep_start_sec,
                keep_end_sec=chunk.keep_end_sec,
                latency_sec=0.5,
                attempt=1,
                text=repetitive_text,
                language="zh",
                duration_sec=chunk.duration_sec,
            ),
        ):
            result = transcriber._transcribe_chunk(
                "/tmp/input.wav",
                "/tmp/work",
                chunk,
                1,
                ChunkRequestPlan(),
            )

        assert result.text == "九十年代，我们是被苏联的猎人。"
        assert result.language == "zh"
        assert result.quality_fallback.pathological_repetition_detected is True
        assert result.quality_fallback.split_fallback_used is False
        assert result.quality_fallback.repair_only_accepted is False
        assert result.quality_fallback.repair_passes_applied >= 1
        assert result.quality_fallback.total_transcribe_requests == 1

    def test_transcribe_chunk_collapses_ascii_sentence_repetition(self):
        transcriber = LongAudioTranscriber("http://127.0.0.1:27900")
        chunk = AudioChunk(
            index=9,
            start_sec=300.0,
            end_sec=312.0,
            keep_start_sec=300.0,
            keep_end_sec=312.0,
        )
        repetitive_text = ("To bylo prilis jasne. " * 12).strip()

        with patch.object(
            LongAudioTranscriber,
            "_transcribe_chunk_with_plan",
            return_value=ChunkTranscriptionResult(
                chunk_index=chunk.index,
                start_sec=chunk.start_sec,
                end_sec=chunk.end_sec,
                keep_start_sec=chunk.keep_start_sec,
                keep_end_sec=chunk.keep_end_sec,
                latency_sec=0.5,
                attempt=1,
                text=repetitive_text,
                language="cs",
                duration_sec=chunk.duration_sec,
            ),
        ):
            result = transcriber._transcribe_chunk(
                "/tmp/input.wav",
                "/tmp/work",
                chunk,
                1,
                ChunkRequestPlan(),
            )

        assert result.text == "To bylo prilis jasne."
        assert result.language == "cs"

    def test_transcribe_chunk_collapses_repeated_word_runs(self):
        transcriber = LongAudioTranscriber("http://127.0.0.1:27900")
        chunk = AudioChunk(
            index=10,
            start_sec=400.0,
            end_sec=412.0,
            keep_start_sec=400.0,
            keep_end_sec=412.0,
        )
        repetitive_text = ("ја " * 64).strip()

        with patch.object(
            LongAudioTranscriber,
            "_transcribe_chunk_with_plan",
            return_value=ChunkTranscriptionResult(
                chunk_index=chunk.index,
                start_sec=chunk.start_sec,
                end_sec=chunk.end_sec,
                keep_start_sec=chunk.keep_start_sec,
                keep_end_sec=chunk.keep_end_sec,
                latency_sec=0.5,
                attempt=1,
                text=repetitive_text,
                language="sr",
                duration_sec=chunk.duration_sec,
            ),
        ):
            result = transcriber._transcribe_chunk(
                "/tmp/input.wav",
                "/tmp/work",
                chunk,
                1,
                ChunkRequestPlan(),
            )

        assert result.text == "ја"
        assert result.language == "sr"

    def test_transcribe_chunk_accepts_substantial_repair_without_split(self):
        transcriber = LongAudioTranscriber("http://127.0.0.1:27900")
        chunk = AudioChunk(
            index=11,
            start_sec=500.0,
            end_sec=580.0,
            keep_start_sec=500.0,
            keep_end_sec=580.0,
        )
        repetitive_text = (
            "Bridge report begins. Witness from Belgrade speaks. "
            "Interview continues under sirens. "
            "The camera moves toward the river crossing. "
            "The archive footage is introduced here. "
            + ("To bylo prilis jasne. " * 40)
            + "The narrator returns to the bridge. "
            "A second witness describes the convoy in detail. "
            "Another interview covers the shelter and the evacuation road. "
            "Final credits mention the year and the city."
        )

        with patch.object(
            LongAudioTranscriber,
            "_transcribe_chunk_with_plan",
            return_value=ChunkTranscriptionResult(
                chunk_index=chunk.index,
                start_sec=chunk.start_sec,
                end_sec=chunk.end_sec,
                keep_start_sec=chunk.keep_start_sec,
                keep_end_sec=chunk.keep_end_sec,
                latency_sec=0.5,
                attempt=1,
                text=repetitive_text,
                language="cs",
                duration_sec=chunk.duration_sec,
            ),
        ) as mock_transcribe:
            result = transcriber._transcribe_chunk(
                "/tmp/input.wav",
                "/tmp/work",
                chunk,
                1,
                ChunkRequestPlan(),
            )

        assert mock_transcribe.call_count == 1
        assert result.text.count("To bylo prilis jasne.") == 1
        assert "Bridge report begins." in result.text
        assert "Final credits mention the year and the city." in result.text
        assert result.language == "cs"
        assert result.quality_fallback.pathological_repetition_detected is True
        assert result.quality_fallback.repair_only_accepted is True
        assert result.quality_fallback.split_fallback_used is False
        assert result.quality_fallback.total_transcribe_requests == 1

    def test_transcribe_chunk_targeted_resplit_skips_intermediate_pathological_half(
        self,
    ):
        transcriber = LongAudioTranscriber("http://127.0.0.1:27900")
        chunk = AudioChunk(
            index=12,
            start_sec=100.0,
            end_sec=142.0,
            keep_start_sec=100.0,
            keep_end_sec=142.0,
        )
        repetitive_text = (
            ("这个，" * 80) + "这个吗？是这个玩意吗？不是这个。防范皆一，啥意思？这个。"
        ).strip()

        def side_effect(
            audio_path: str,
            work_dir: str,
            requested_chunk: AudioChunk,
            attempt: int,
            request_plan: ChunkRequestPlan,
        ) -> ChunkTranscriptionResult:
            del audio_path
            del work_dir
            del request_plan
            if requested_chunk.start_sec == 100.0 and requested_chunk.end_sec == 142.0:
                text = repetitive_text
            elif requested_chunk.end_sec <= 111.7:
                text = "这个战斗，我赢不了。优化吸附判定机制。土牛臂调整为十秒。"
            elif requested_chunk.end_sec <= 122.0:
                text = "双雷震，双雷震是哪个技能？这个吗？神威副爵释放时角色获得八级，啥意思？这个吗？"
            else:
                text = "这个吗？是这个玩意吗？不是这个。防范皆一，啥意思？这个。"
            return ChunkTranscriptionResult(
                chunk_index=requested_chunk.index,
                start_sec=requested_chunk.start_sec,
                end_sec=requested_chunk.end_sec,
                keep_start_sec=requested_chunk.keep_start_sec,
                keep_end_sec=requested_chunk.keep_end_sec,
                latency_sec=0.5,
                attempt=attempt,
                text=text,
                language="zh",
                duration_sec=requested_chunk.duration_sec,
            )

        with patch.object(
            LongAudioTranscriber,
            "_transcribe_chunk_with_plan",
            side_effect=side_effect,
        ) as mock_transcribe:
            result = transcriber._transcribe_chunk(
                "/tmp/input.wav",
                "/tmp/work",
                chunk,
                1,
                ChunkRequestPlan(),
            )

        assert result.text == (
            "这个战斗，我赢不了。优化吸附判定机制。土牛臂调整为十秒。"
            "双雷震，双雷震是哪个技能？这个吗？神威副爵释放时角色获得八级，啥意思？这个吗？"
            "是这个玩意吗？不是这个。防范皆一，啥意思？这个。"
        )
        assert result.language == "zh"
        assert mock_transcribe.call_count == 4
        assert result.quality_fallback.pathological_repetition_detected is True
        assert result.quality_fallback.split_fallback_used is True
        assert result.quality_fallback.split_child_count == 3
        assert result.quality_fallback.max_split_depth == 2
        assert result.quality_fallback.total_transcribe_requests == 4


class TestLongAudioRepairHelpers:
    def test_repair_pathological_repetition_preserves_distinct_numbered_sentences(self):
        text = (
            "Lead sentence 1. Lead sentence 2. " "Lead sentence 3. Trailing sentence 1."
        )

        assert _repair_pathological_repetition(text) == text

    def test_targeted_edge_resplit_side_detects_prefix_run(self):
        text = ("这个，" * 80) + "这里开始恢复正常描述。后面还有一句。"

        assert _targeted_edge_resplit_side(text) == "prefix"

    def test_targeted_edge_resplit_side_detects_suffix_run_after_short_prefix(self):
        prefix = "甲。乙。丙。丁。戊。己。庚。辛。壬。癸。子。丑。寅。卯。辰。巳。"
        text = prefix + ("哦，" * 80) + "最后收个尾。"

        assert _targeted_edge_resplit_side(text) == "suffix"

    def test_targeted_edge_resplit_side_skips_suffix_run_after_tiny_prefix(self):
        text = (
            "前面这里有几句正常描述。然后大家还在继续聊。"
            + ("哦，" * 80)
            + "最后收个尾。"
        )

        assert _targeted_edge_resplit_side(text) is None


class TestLongAudioTelemetry:
    def test_long_audio_result_includes_quality_fallback_summary(self):
        result = LongAudioTranscriptionResult(
            audio="/tmp/input.wav",
            metadata=AudioMetadata(path="/tmp/input.wav", duration_sec=120.0),
            chunks=[],
            silence_regions=[],
            chunk_results=[
                ChunkTranscriptionResult(
                    chunk_index=0,
                    start_sec=0.0,
                    end_sec=60.0,
                    keep_start_sec=0.0,
                    keep_end_sec=60.0,
                    latency_sec=1.0,
                    attempt=1,
                    text="alpha",
                    language="en",
                    duration_sec=60.0,
                    quality_fallback=QualityFallbackTrace(
                        pathological_repetition_detected=True,
                        repair_attempted=True,
                        repair_only_accepted=True,
                        repair_passes_applied=2,
                        total_transcribe_requests=1,
                    ),
                ),
                ChunkTranscriptionResult(
                    chunk_index=1,
                    start_sec=60.0,
                    end_sec=120.0,
                    keep_start_sec=60.0,
                    keep_end_sec=120.0,
                    latency_sec=1.0,
                    attempt=1,
                    text="beta",
                    language="en",
                    duration_sec=60.0,
                    quality_fallback=QualityFallbackTrace(
                        pathological_repetition_detected=True,
                        repair_attempted=True,
                        split_fallback_used=True,
                        post_split_repair_applied=True,
                        split_child_count=2,
                        max_split_depth=2,
                        repair_passes_applied=5,
                        total_transcribe_requests=5,
                    ),
                ),
            ],
            total_time_sec=2.0,
            max_parallel_chunks=2,
            endpoint="http://127.0.0.1:27900",
            machine_snapshot=None,
            text="alpha beta",
            language="en",
        )

        payload = result.to_dict()

        assert payload["transcription"]["quality_fallback_summary"] == {
            "chunk_count": 2,
            "pathological_chunks": 2,
            "repair_only_accepted_chunks": 1,
            "split_fallback_chunks": 1,
            "post_split_repair_chunks": 1,
            "max_split_depth": 2,
            "repair_passes_applied": 7,
            "total_transcribe_requests": 6,
            "extra_transcribe_requests": 4,
        }
