"""Parse tests for `GeminiClient.refine_click` and `disambiguate_candidates`.

These stub out the actual Gemini API call so we can feed the response parser
arbitrary text and assert what it extracts.
"""
from __future__ import annotations

from unittest import mock

import pytest
from PIL import Image

from agent.vlm import GeminiClient


@pytest.fixture
def client():
    # Avoid constructing a real SDK client by patching genai.Client.
    with mock.patch("agent.vlm.genai.Client"):
        return GeminiClient(api_key="fake", model_name="fake-model")


def _stub_response(client: GeminiClient, text: str):
    """Patch the underlying `generate_content` to return a response with `.text`."""
    resp = mock.MagicMock()
    resp.text = text
    client._client.models.generate_content = mock.MagicMock(return_value=resp)


# -----------------------------------------------------------------------------
# refine_click
# -----------------------------------------------------------------------------
def test_refine_click_parses_multiple_candidates(client):
    _stub_response(
        client,
        "CLICK [120,340]\nCLICK [560,780]\nCLICK [100,100]",
    )
    got = client.refine_click("step", Image.new("RGB", (100, 100)))
    assert got == [(120, 340), (560, 780), (100, 100)]


def test_refine_click_single_candidate(client):
    _stub_response(client, "CLICK [500,500]")
    assert client.refine_click("step", Image.new("RGB", (10, 10))) == [(500, 500)]


def test_refine_click_none_returns_empty(client):
    _stub_response(client, "NONE")
    assert client.refine_click("step", Image.new("RGB", (10, 10))) == []


def test_refine_click_clamps_out_of_range_values(client):
    _stub_response(client, "CLICK [-50,2000]\nCLICK [1234,500]")
    assert client.refine_click("step", Image.new("RGB", (10, 10))) == [
        (0, 1000),
        (1000, 500),
    ]


def test_refine_click_respects_max_candidates(client):
    _stub_response(
        client,
        "\n".join(f"CLICK [{i*100},{i*100}]" for i in range(10)),
    )
    got = client.refine_click(
        "step", Image.new("RGB", (10, 10)), max_candidates=3
    )
    assert got == [(0, 0), (100, 100), (200, 200)]


def test_refine_click_accepts_lenient_format(client):
    _stub_response(client, "1. [250, 400]\n2. (700, 100)")
    assert client.refine_click("step", Image.new("RGB", (10, 10))) == [
        (250, 400),
        (700, 100),
    ]


def test_refine_click_empty_response_returns_empty(client):
    _stub_response(client, "")
    assert client.refine_click("step", Image.new("RGB", (10, 10))) == []


def test_refine_click_ignores_unparseable_lines(client):
    _stub_response(client, "hmm let me think\nCLICK [42,84]\nI'm not sure")
    assert client.refine_click("step", Image.new("RGB", (10, 10))) == [(42, 84)]


# -----------------------------------------------------------------------------
# disambiguate_candidates
# -----------------------------------------------------------------------------
def test_disambiguate_picks_valid_index(client):
    _stub_response(client, "PICK [2]")
    assert client.disambiguate_candidates("step", Image.new("RGB", (10, 10)), 3) == 2


def test_disambiguate_zero_means_none_match(client):
    _stub_response(client, "PICK [0]")
    assert client.disambiguate_candidates("step", Image.new("RGB", (10, 10)), 3) == 0


def test_disambiguate_out_of_range_returns_zero(client):
    _stub_response(client, "PICK [99]")
    assert client.disambiguate_candidates("step", Image.new("RGB", (10, 10)), 3) == 0


def test_disambiguate_lenient_bare_integer(client):
    _stub_response(client, "2")
    assert client.disambiguate_candidates("step", Image.new("RGB", (10, 10)), 3) == 2


def test_disambiguate_unparseable_returns_zero(client):
    _stub_response(client, "I'm not sure which one")
    assert client.disambiguate_candidates("step", Image.new("RGB", (10, 10)), 3) == 0
