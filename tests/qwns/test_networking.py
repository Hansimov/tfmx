"""Tests for tfmx.qwns.networking."""

from unittest.mock import patch

from tfmx.qwns.networking import DEFAULT_HF_ENDPOINT
from tfmx.qwns.networking import DEFAULT_PIP_INDEX_URL
from tfmx.qwns.networking import DEFAULT_PIP_TRUSTED_HOST
from tfmx.qwns.networking import QWNNetworkConfig
from tfmx.qwns.networking import detect_default_proxy
from tfmx.qwns.networking import is_loopback_proxy


class TestProxyDetection:
    def test_is_loopback_proxy(self):
        loopback_proxy = "http://127.0.0.1:" + "18080"
        localhost_proxy = "http://localhost:" + "18080"
        assert is_loopback_proxy(loopback_proxy) is True
        assert is_loopback_proxy(localhost_proxy) is True
        assert is_loopback_proxy("http://proxy.internal:8080") is False

    def test_detect_default_proxy_from_common_proxy_map(self):
        common_proxy = "http://proxy.internal:8080"
        assert detect_default_proxy(proxies={"https": common_proxy}) == common_proxy

    @patch("tfmx.qwns.networking.os.getenv")
    def test_detect_default_proxy_from_env(self, mock_getenv):
        values = {"TFMX_QWN_PROXY": "http://proxy.internal:8080"}
        mock_getenv.side_effect = lambda key: values.get(key)
        assert detect_default_proxy(proxies={}) == "http://proxy.internal:8080"


class TestQWNNetworkConfig:
    @patch("tfmx.qwns.networking.getproxies")
    def test_default_config_from_system_proxy(self, mock_getproxies):
        loopback_proxy = "http://localhost:" + "18080"
        mock_getproxies.return_value = {"https": loopback_proxy}
        config = QWNNetworkConfig.from_overrides()
        assert config.hf_endpoint == DEFAULT_HF_ENDPOINT
        assert config.pip_index_url == DEFAULT_PIP_INDEX_URL
        assert config.pip_trusted_host == DEFAULT_PIP_TRUSTED_HOST
        assert config.build_proxy == loopback_proxy
        assert config.use_host_network_for_build is True
        assert config.runtime_http_proxy is None

    def test_remote_proxy_survives_runtime(self):
        config = QWNNetworkConfig.from_overrides(proxy="http://proxy.internal:8080")
        assert config.runtime_http_proxy == "http://proxy.internal:8080"
        assert config.runtime_https_proxy == "http://proxy.internal:8080"

    def test_docker_build_args_include_pip_mirror(self):
        config = QWNNetworkConfig.from_overrides(proxy="http://proxy.internal:8080")
        args = config.docker_build_args()
        assert f"PIP_INDEX_URL={DEFAULT_PIP_INDEX_URL}" in args
        assert f"PIP_TRUSTED_HOST={DEFAULT_PIP_TRUSTED_HOST}" in args
        assert "HTTP_PROXY=http://proxy.internal:8080" in args
