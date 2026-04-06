"""Tests for tfmx.qwns.adaptive_routing."""

import pytest

from tfmx.qwns.adaptive_routing import SchedulerTuningConfig
from tfmx.qwns.adaptive_routing import SchedulerWeights
from tfmx.qwns.adaptive_routing import tune_scheduler_weights


class TestSchedulerAutoTuning:
    def test_tuning_increases_gpu_and_failure_weights_under_pressure(self):
        base = SchedulerWeights()
        result = tune_scheduler_weights(
            base_weights=base,
            active_ratios=[0.25, 0.30],
            gpu_pressures=[0.94, 0.16],
            latencies_ms=[820.0, 860.0],
            ttfts_ms=[220.0, 240.0],
            throughputs_tokens_per_second=[96.0, 94.0],
            recent_requests=[24, 24],
            failure_rates=[0.20, 0.0],
            consecutive_failures=[2, 0],
            config=SchedulerTuningConfig(),
        )

        assert result.weights.gpu_pressure > base.gpu_pressure
        assert result.weights.failures > base.failures
        assert sum(result.weights.raw_dict().values()) == pytest.approx(1.0)

    def test_tuning_increases_throughput_weight_for_speed_gap(self):
        base = SchedulerWeights()
        result = tune_scheduler_weights(
            base_weights=base,
            active_ratios=[0.15, 0.15, 0.15],
            gpu_pressures=[0.18, 0.19, 0.18],
            latencies_ms=[780.0, 790.0, 800.0],
            ttfts_ms=[210.0, 220.0, 215.0],
            throughputs_tokens_per_second=[140.0, 72.0, 66.0],
            recent_requests=[20, 20, 20],
            failure_rates=[0.0, 0.0, 0.0],
            consecutive_failures=[0, 0, 0],
            config=SchedulerTuningConfig(),
        )

        assert result.weights.throughput > base.throughput
        assert result.signals["throughput_cv"] > 0.0
