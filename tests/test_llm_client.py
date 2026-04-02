"""用于测试 unified LLM client 模拟/真实 模式 与 failure protections."""

from __future__ import annotations

import httpx

from infra.llm_client import LLMClient, LLMClientConfig


def test_llm_client_mock_mode_returns_structured_payload() -> None:
    client = LLMClient(LLMClientConfig(mode="mock_llm"))
    response = client.generate_json(
        task="strategist",
        payload={"diagnosis": {"control_state": "maintain_balance"}},
        prompt_template="ignored",
    )

    assert response.success is True
    assert response.content is not None
    assert response.content["control_state"] in {
        "increase_diversity",
        "increase_convergence",
        "increase_feasibility",
        "maintain_balance",
    }


def test_llm_client_real_mode_missing_key_fallbacks_to_mock(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    client = LLMClient(LLMClientConfig(mode="real_llm", fallback_mode="mock_llm"))

    response = client.generate_json(task="analyst", payload={"state": {}}, prompt_template="p")

    assert response.success is False
    assert response.mode_used == "mock_llm"
    assert response.content is not None
    assert "Missing API key" in (response.error or "")


def test_llm_client_real_mode_missing_key_hold(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    client = LLMClient(LLMClientConfig(mode="real_llm", fallback_mode="hold"))

    response = client.generate_json(task="analyst", payload={"state": {}}, prompt_template="p")

    assert response.success is False
    assert response.mode_used == "hold"
    assert response.content is None


def test_llm_client_real_mode_timeout_fallback(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "fake-key")
    client = LLMClient(LLMClientConfig(mode="real_llm", fallback_mode="mock_llm"))

    def _raise_timeout(**kwargs):
        raise httpx.TimeoutException("timeout")

    monkeypatch.setattr(client, "_call_real_with_retry", _raise_timeout)

    response = client.generate_json(task="strategist", payload={"diagnosis": {}}, prompt_template="p")

    assert response.success is False
    assert response.mode_used == "mock_llm"
    assert response.content is not None
    assert "failed" in (response.error or "")


def test_llm_client_real_mode_honors_max_retries(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "fake-key")
    client = LLMClient(LLMClientConfig(mode="real_llm", fallback_mode="hold", max_retries=1))
    attempts = {"n": 0}

    def _always_timeout(**kwargs):
        del kwargs
        attempts["n"] += 1
        raise httpx.TimeoutException("timeout")

    monkeypatch.setattr(client, "_call_real_once", _always_timeout)

    response = client.generate_json(task="strategist", payload={"diagnosis": {}}, prompt_template="p")

    assert response.success is False
    assert response.mode_used == "hold"
    assert attempts["n"] == 2


def test_llm_client_real_mode_success_with_fake_transport(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "fake-key")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-test")
    client = LLMClient(LLMClientConfig(mode="real_llm", fallback_mode="hold"))

    def _ok(**kwargs):
        assert kwargs["model"] == "gpt-test"
        return {"control_state": "maintain_balance", "rationale": "ok"}

    monkeypatch.setattr(client, "_call_real_with_retry", _ok)

    response = client.generate_json(task="strategist", payload={"diagnosis": {}}, prompt_template="p")

    assert response.success is True
    assert response.mode_used == "real_llm"
    assert response.content == {"control_state": "maintain_balance", "rationale": "ok"}

def test_llm_client_real_mode_uses_longer_read_timeout() -> None:
    client = LLMClient(LLMClientConfig(mode="real_llm", timeout_s=10.0, min_read_timeout_s=25.0))

    timeout = client._build_http_timeout()

    assert timeout.connect == 10.0
    assert timeout.read == 25.0


def test_llm_client_real_mode_allows_disabling_timeout() -> None:
    client = LLMClient(LLMClientConfig(mode="real_llm", timeout_s=0))

    timeout = client._build_http_timeout()

    assert timeout.connect is None
    assert timeout.read is None