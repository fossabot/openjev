"""End to end against a running OpenJev server, whichever backend and wherever it runs:

    OPENJEV_LIVE_URL=http://127.0.0.1:8080 pytest tests/test_live.py -v

Set OPENJEV_API_KEY or OPENJEV_ORIGIN_SECRET too if the server wants one. Set OPENJEV_LIVE_GATEWAY=1 when a gateway
(such as codiv's) sits in front and strips `Server-Timing`. Run it after building an image or
before a cutover. The DiffusionGemma checks run when the server lists `openjev-latest`;
the encoder checks run for whichever of `laya-1.0`, `verdict-1.4`, `clm-v0.1` and `jevk5-0.2` it lists.
"""
import base64
import concurrent.futures
import json
import os
import pathlib

import httpx
import pytest

URL = os.environ.get("OPENJEV_LIVE_URL")
GATEWAY = os.environ.get("OPENJEV_LIVE_GATEWAY") == "1"
pytestmark = pytest.mark.skipif(not URL, reason="set OPENJEV_LIVE_URL to a running OpenJev server")

QUESTIONS = {
    "urgent": {"type": "noul", "instructions": "Does the customer need a reply within the hour?"},
    "team": {"type": "choice", "instructions": "Which team should handle it?",
             "criteria": {"outage": "service down", "billing": "charges, refunds", "feature": "requests, how-to"}},
    "tone": {"type": "score", "instructions": "How upset is the customer?", "criteria": ["calm", "annoyed", "furious"]},
}
STATE = "Everything is down and we have a demo with our biggest client at noon."
HOTDOG = "data:image/jpeg;base64," + base64.b64encode(
    (pathlib.Path(__file__).parent / "data" / "hotdog.jpg").read_bytes()).decode()


@pytest.fixture(scope="module")
def client():
    headers = {"Authorization": f"Bearer {os.environ['OPENJEV_API_KEY']}"} if os.environ.get("OPENJEV_API_KEY") else {}
    if os.environ.get("OPENJEV_ORIGIN_SECRET"):
        headers["X-Origin-Secret"] = os.environ["OPENJEV_ORIGIN_SECRET"]
    with httpx.Client(base_url=URL, headers=headers, timeout=300) as c:
        yield c


@pytest.fixture(scope="module")
def models(client):
    r = client.get("/v1/models")
    assert r.status_code == 200, r.text
    return {m["name"] for m in r.json()["models"]}


@pytest.fixture
def dgemma(models):
    if "openjev-latest" not in models:
        pytest.skip("the server does not serve DiffusionGemma")


def ask(client, questions=QUESTIONS, state=STATE, model="openjev-latest", **extra):
    r = client.post("/v1/systemone", json={"model": model, "state": state, "questions": questions, **extra})
    assert r.status_code == 200, r.text
    assert GATEWAY or "server-timing" in r.headers
    return r.json()


def test_readme_example(client, dgemma):
    a = ask(client)["answers"]
    assert a["urgent"]["noul"] > 0.5, a["urgent"]
    assert a["team"]["choice"] == "outage", a["team"]
    assert a["tone"]["score"] > 1.0, a["tone"]
    assert abs(sum(a["team"]["probabilities"].values()) - 1) < 1e-6


def test_image(client, dgemma):
    body = ask(client, {"hotdog": {"type": "noul", "instructions": "The photo shows a hot dog"},
                        "cat": {"type": "noul", "instructions": "The photo shows a cat"}},
               state="Look at the photo.", images=[HOTDOG])
    a = body["answers"]
    assert a["hotdog"]["noul"] > 0.8 and a["cat"]["noul"] < 0.2, a
    assert body["usage"]["input_tokens"] > 200  # the image's tokens are counted


@pytest.mark.parametrize("extra", [{"steps": 4}, {"samples": 4}, {"sequential": True}])
def test_read_options(client, dgemma, extra):
    assert ask(client, **extra)["answers"]["team"]["choice"] == "outage"


def test_think(client, dgemma):
    body = ask(client, think=256)
    assert body["usage"]["output_tokens"] > 0
    assert body["answers"]["team"]["choice"] == "outage"


def test_255_options(client, dgemma):
    criteria = {f"opt{i}": f"option number {i}" for i in range(255)}
    p = ask(client, {"pick": {"type": "choice", "instructions": "Which option fits?", "criteria": criteria}})
    p = p["answers"]["pick"]["probabilities"]
    assert len(p) == 255 and abs(sum(p.values()) - 1) < 1e-3


def test_many_questions_in_chunks(client, dgemma):
    qs = {f"q{i}": {"type": "noul", "instructions": f"Is the number {i} even?"} for i in range(30)}
    assert len(ask(client, qs, state="Answer about numbers.")["answers"]) == 30


def test_unknown_model(client):
    r = client.post("/v1/systemone", json={"model": "nope", "state": STATE, "questions": QUESTIONS})
    assert r.status_code == 400, r.text


def test_concurrent_reads(client, dgemma):
    with concurrent.futures.ThreadPoolExecutor(32) as ex:
        codes = list(ex.map(lambda i: client.post("/v1/systemone", json={
            "model": "openjev-latest", "state": f"{STATE} (ticket {i})", "questions": QUESTIONS}).status_code, range(64)))
    assert codes == [200] * 64


def test_chat(client, models):
    if "diffusiongemma-26b" not in models:
        pytest.skip("the server does not serve text generation")
    r = client.post("/v1/chat/completions", json={"model": "diffusiongemma-26b", "max_tokens": 64,
                    "messages": [{"role": "user", "content": "What is 2+2? Answer with one number."}]})
    assert r.status_code == 200, r.text
    assert "4" in r.json()["choices"][0]["message"]["content"]


def test_chat_stream(client, models):
    if "diffusiongemma-26b" not in models:
        pytest.skip("the server does not serve text generation")
    r = client.post("/v1/chat/completions", json={"model": "diffusiongemma-26b", "max_tokens": 64, "stream": True,
                    "messages": [{"role": "user", "content": "Name a colour of the sky."}]})
    assert r.status_code == 200, r.text
    lines = [l for l in r.text.splitlines() if l.startswith("data: ")]
    assert lines[-1] == "data: [DONE]"
    text = "".join((c["choices"][0]["delta"].get("content") or "")
                   for c in (json.loads(l[6:]) for l in lines[:-1]) if c.get("choices"))
    assert text.strip()


@pytest.mark.parametrize("model", ["laya-1.0", "verdict-1.4", "clm-v0.1", "jevk5-0.2"])
def test_encoder(client, models, model):
    if model not in models:
        pytest.skip(f"the server does not serve {model}")
    state = "I was charged twice this month."
    a = ask(client, state=state, model=model)["answers"]
    assert a["team"]["choice"] == "billing", a["team"]
    assert abs(sum(a["team"]["probabilities"].values()) - 1) < 1e-6
