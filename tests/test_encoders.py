"""Offline tests for the small backends (Laya, Verdict, CLM, JevK5): stubbed models, the real API."""
import json
import math
import time
import zlib

import httpx
import pytest
from fastapi.testclient import TestClient

from openjev import encoders
from openjev.api import create_app
from openjev.config import Settings, parse_routes, served_models
from openjev.encoders import ClmEngine, EncoderEngine, JevK5Engine, VerdictEngine, verdict_prompt

REQUEST = {
    "state": "I was charged twice this month.",
    "model": "laya-1.0",
    "questions": {
        "team": {"type": "choice", "instructions": "Which team should handle it?",
                 "criteria": {"outage": "service down", "billing": "charges, refunds", "feature": None}},
        "tone": {"type": "score", "instructions": "How upset is the customer?", "criteria": ["calm", "annoyed", "furious"]},
        "urgent": {"type": "noul", "instructions": "Does the customer need a reply within the hour?"},
    },
}


class FakeEngine(EncoderEngine):
    model_name = "laya-1.0"

    def load(self):
        self.reads = []

    def read_batch(self, state, qs):
        self.reads.append((state, qs))
        # the second option 70%, the rest share 30%
        out = []
        for q in qs:
            n = len(q["choices"])
            p = [0.3 / (n - 1)] * n
            p[1] = 0.7
            out.append(p)
        return out, 99


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setitem(encoders.ENGINES, "laya", FakeEngine)
    with TestClient(create_app(Settings(backend="laya", warmup=False))) as c:
        yield c


def test_answers_in_jevs_shapes(client):
    r = client.post("/v1/systemone", json=REQUEST)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["model"] == "laya-1.0"
    assert body["usage"] == {"input_tokens": 99, "output_tokens": 0}
    a = body["answers"]
    assert list(a) == ["team", "tone", "urgent"]
    assert a["team"]["choice"] == "billing"
    assert a["team"]["probabilities"]["billing"] == pytest.approx(0.7)
    assert a["tone"]["score"] == pytest.approx(0.15 * 0 + 0.7 * 1 + 0.15 * 2)
    assert a["tone"]["legend"] == {"0": "calm", "1": "annoyed", "2": "furious"}
    assert a["urgent"] == {"type": "noul", "noul": pytest.approx(0.3)}  # P(yes) is the first option


def test_server_timing_counts_the_read(client):
    timing = client.post("/v1/systemone", json=REQUEST).headers["server-timing"]
    assert timing.startswith("model;dur=")


def test_typesafe_sdk_default_model_is_accepted(client):
    from typesafe_sdk import TypeSafeClient

    sdk = TypeSafeClient(api_key="x", base_url="http://testserver", http_client=client)
    r = sdk.system_one(REQUEST["state"], REQUEST["questions"])
    assert r.choices["team"].choice == "billing"


def test_models_and_unknown_model(client):
    assert [m["name"] for m in client.get("/v1/models").json()["models"]] == ["laya-1.0"]
    r = client.post("/v1/systemone", json=dict(REQUEST, model="openjev-latest"))
    assert r.status_code == 400
    assert r.json()["detail"]["message"] == "Unknown model: openjev-latest"


def test_no_text_generation(client):
    r = client.post("/v1/chat/completions", json={"model": "diffusiongemma-26b", "messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 404


@pytest.mark.parametrize("extra", [{"steps": 2}, {"samples": 4}, {"think": 64}, {"sequential": True},
                                   {"images": ["data:image/png;base64,iVBORw0KGgo="]}])
def test_read_options_are_refused(client, extra):
    r = client.post("/v1/systemone", json=dict(REQUEST, **extra))
    assert r.status_code == 400
    assert "does not support" in r.json()["detail"]


def test_options_left_at_their_defaults_are_fine(client):
    r = client.post("/v1/systemone", json=dict(REQUEST, steps=1, samples=1, think=0, sequential=False))
    assert r.status_code == 200


def test_one_option_and_one_level_need_no_read(client):
    q = {"only": {"type": "choice", "criteria": {"yes": None}}, "lvl": {"type": "score", "criteria": ["one"]}}
    r = client.post("/v1/systemone", json=dict(REQUEST, questions=q))
    assert r.json()["answers"]["only"]["probabilities"] == {"yes": 1.0}
    assert r.json()["usage"]["input_tokens"] == 0
    assert client.app.state.engine.reads == []


def test_many_questions_are_read_in_batches(client):
    q = {f"n{i}": {"type": "noul", "instructions": "x"} for i in range(40)}
    r = client.post("/v1/systemone", json=dict(REQUEST, questions=q))
    assert list(r.json()["answers"]) == list(q)
    assert [len(qs) for _, qs in client.app.state.engine.reads] == [16, 16, 8]
    assert r.json()["usage"]["input_tokens"] == 3 * 99


def test_warmup_reads_before_serving(monkeypatch):
    monkeypatch.setitem(encoders.ENGINES, "laya", FakeEngine)
    with TestClient(create_app(Settings(backend="laya"))) as c:
        assert [len(qs) for _, qs in c.app.state.engine.reads] == [3]


def test_limits(client):
    many = {"c": {"type": "choice", "criteria": {f"o{i}": None for i in range(256)}}}
    assert client.post("/v1/systemone", json=dict(REQUEST, questions=many)).json() == {
        "detail": "Too many choices. Must have at most 255 choices."}
    empty = {"c": {"type": "choice", "criteria": {}}}
    assert client.post("/v1/systemone", json=dict(REQUEST, questions=empty)).status_code == 400


def test_routes_forward_other_models(monkeypatch):
    monkeypatch.setenv("OPENJEV_MODEL_ROUTES", "verdict-1.4=http://verdict:8080/, laya-1.0=http://laya:8080")
    monkeypatch.setitem(encoders.ENGINES, "laya", FakeEngine)
    seen = []

    def handler(req):
        seen.append(req)
        time.sleep(0.002)  # long enough to show up in Server-Timing
        return httpx.Response(200, json={"model": "verdict-1.4", "answers": {}, "usage": {}})

    with TestClient(create_app(Settings(backend="laya", origin_secret="s", warmup=False))) as c:
        c.app.state.routes = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        names = [m["name"] for m in c.get("/v1/models", headers={"x-origin-secret": "s"}).json()["models"]]
        assert names == ["laya-1.0", "verdict-1.4"]  # its own model is not listed twice
        r = c.post("/v1/systemone", json=dict(REQUEST, model="verdict-1.4"), headers={"x-origin-secret": "s"})
        assert r.json()["model"] == "verdict-1.4"
        assert not r.headers["server-timing"].startswith("model;dur=0.0,")
        assert str(seen[0].url) == "http://verdict:8080/v1/systemone"
        assert seen[0].headers["x-origin-secret"] == "s"
        # its own model is answered here, not forwarded
        assert c.post("/v1/systemone", json=REQUEST, headers={"x-origin-secret": "s"}).json()["model"] == "laya-1.0"
        assert len(seen) == 1
        # an unreachable route is a 503, like an unreachable vLLM
        c.app.state.routes = httpx.AsyncClient(transport=httpx.MockTransport(lambda req: (_ for _ in ()).throw(httpx.ConnectError("x"))))
        assert c.post("/v1/systemone", json=dict(REQUEST, model="verdict-1.4"), headers={"x-origin-secret": "s"}).status_code == 503


def test_parse_routes():
    assert parse_routes("") == {}
    assert parse_routes("a=http://x/, b = http://y") == {"a": "http://x", "b": "http://y"}
    with pytest.raises(ValueError):
        parse_routes("a")


def test_verdict_prompt_matches_upstream_contract():
    # core/formatting.py of Verdict-open-jev: labels first, then the text
    q = {"type": "choice", "instructions": "Which team?", "choices": [("outage", "service down"), ("sales", "")]}
    assert verdict_prompt(q, "ctx") == (
        "<<LABEL>>It is service down<<LABEL>>It is sales<<LABEL>>insufficient evidence<<SEP>>Question: Which team?\n\nContext:\nctx", 3)
    q = {"type": "score", "instructions": "How bad?", "choices": [("0", "low"), ("1", "high")]}
    assert verdict_prompt(q, "ctx")[0].startswith("<<LABEL>>low (Value: 0.0)<<LABEL>>high (Value: 1.0)<<LABEL>>")
    q = {"type": "noul", "instructions": "It is urgent", "choices": []}
    assert verdict_prompt(q, "ctx") == (
        ("<<LABEL>>true: It is urgent<<LABEL>>false: not It is urgent<<LABEL>>insufficient evidence"
         "<<SEP>>Context:\nctx\n\nEvaluate proposition: It is urgent"), 3)


def test_verdict_read_calibrates_and_drops_abstention():
    torch = pytest.importorskip("torch")

    class Tok:
        def __call__(self, prompts, **kw):
            n = len(prompts)
            return _Batch({"input_ids": torch.ones(n, 4, dtype=torch.long), "attention_mask": torch.ones(n, 4, dtype=torch.long)})

    class _Batch(dict):
        def to(self, device):
            return self

    class Model:
        def __call__(self, **kw):
            logits = torch.full((2, 25), -100.0)
            logits[0, :3] = torch.tensor([2.0, 0.0, 5.0])  # noul: true, false, abstain
            logits[1, :4] = torch.tensor([1.0, 3.0, 0.0, 0.0])
            return type("O", (), {"logits": logits})

    eng = VerdictEngine.__new__(VerdictEngine)
    eng.torch, eng.device, eng.tok, eng.model = torch, "cpu", Tok(), Model()
    eng.temperature, eng.per_k = 2.0, {3: 1.0}
    qs = [{"type": "noul", "instructions": "x", "choices": [("yes", ""), ("no", "")]},
          {"type": "choice", "instructions": "y", "choices": [("a", ""), ("b", ""), ("c", "")]}]
    eng.s = Settings()
    probs, tokens = eng.read("state", qs)
    assert tokens == 8
    # k=3 uses its own temperature (1.0); the abstention's mass is renormalized away
    assert probs[0] == pytest.approx([1 / (1 + math.exp(-2.0)), 1 / (1 + math.exp(2.0))])
    # k=4 has no entry, so the global temperature (2.0) applies
    e = [math.exp(v / 2.0) for v in (1.0, 3.0, 0.0)]
    assert probs[1] == pytest.approx([v / sum(e) for v in e])


# CLM: a real clm Engine and a real (tiny) head checkpoint; only the vLLM embedder is stubbed.
CLM_DIM = 16


class StubEmbedder:
    """Texts that mention a charge, a true statement or fury all embed to one vector; every
    other text to its own. With the state and action heads sharing weights, those options
    score highest against a state about a charge."""

    def __init__(self, workers, url, model, **kw):
        self.url, self.model = url, model

    def embed(self, texts):
        import numpy as np

        def vec(t):
            hot = any(w in t for w in ("charg", "true:", "furious"))
            v = np.random.default_rng(0 if hot else zlib.crc32(t.encode())).standard_normal(CLM_DIM)
            return (v / np.linalg.norm(v)).astype("float32")

        return np.stack([vec(t) for t in texts]), 7 * len(texts)


@pytest.fixture
def clm_engine(monkeypatch, tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("clm")
    from clm.heads import make_head

    head = make_head(width=32, depth=2, proj=8, hidden=CLM_DIM)
    path = tmp_path / "head.pt"
    torch.save({"state_head": head.state_dict(), "action_head": head.state_dict(), "logit_scale": math.log(100.0),
                "cfg": {"width": 32, "depth": 2, "projection_dim": 8, "hidden_size": CLM_DIM}}, path)
    monkeypatch.setattr(encoders, "_left_truncating_embedder", lambda: StubEmbedder)
    return Settings(backend="clm", clm_head=str(path), clm_cache="0", device="cpu", warmup=False)


def test_clm_answers_in_the_callers_option_order(monkeypatch, clm_engine):
    monkeypatch.setitem(encoders.ENGINES, "clm", ClmEngine)
    with TestClient(create_app(clm_engine)) as c:
        engine = c.app.state.engine
        calls = []
        answer = engine.clm.answer
        monkeypatch.setattr(engine.clm, "answer", lambda *a, **kw: calls.append(a) or answer(*a, **kw))
        r = c.post("/v1/systemone", json=dict(REQUEST, model="clm-v0.1"))
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["model"] == "clm-v0.1"
    a = body["answers"]
    assert a["team"]["choice"] == "billing"
    assert list(a["team"]["probabilities"]) == ["outage", "billing", "feature"]
    assert a["tone"]["probabilities"]["2"] > 0.9  # "furious"
    assert a["urgent"]["noul"] > 0.9  # P(true)
    assert len(calls) == 1  # all three questions in one read
    # a state per question, and every option
    assert body["usage"]["input_tokens"] == 7 * (3 + 3 + 3 + 2)


def test_clm_unreachable_vllm_is_a_503(monkeypatch, clm_engine):
    from clm.embedder import EmbedderError

    monkeypatch.setitem(encoders.ENGINES, "clm", ClmEngine)
    with TestClient(create_app(clm_engine)) as c:
        def down(texts):
            raise EmbedderError("embedder unreachable at http://127.0.0.1:8000/v1/embeddings")

        c.app.state.engine.clm.embedder.embed = down
        assert c.post("/v1/systemone", json=dict(REQUEST, model="clm-v0.1")).status_code == 503


def test_instructions_are_kept_as_given():
    eng = ClmEngine.__new__(ClmEngine)
    qs, _ = eng.build_schema({"q": {"type": "noul", "instructions": {"check": "urgent"}}})
    assert qs[0]["raw_instructions"] == {"check": "urgent"}  # clm and jevk5 render it themselves
    assert qs[0]["instructions"] == '{"check": "urgent"}'  # Laya and Verdict read text


def test_clm_model_is_listed():
    name, names, listed = served_models("clm")
    assert name == "clm-v0.1" and "jev-latest" in names and listed[0]["name"] == "clm-v0.1"


def test_clm_embedder_truncates_from_the_left(monkeypatch):
    pytest.importorskip("clm")
    import requests

    sent = []

    class Response:
        status_code = 200

        def json(self):
            return {"data": [{"index": 0, "embedding": [3.0, 4.0]}], "usage": {"prompt_tokens": 5}}

    monkeypatch.setattr(requests.Session, "post", lambda self, url, json=None, **kw: sent.append(json) or Response())
    emb = encoders._left_truncating_embedder()(4, url="http://x/v1/embeddings", model="qwen3-8b", max_tokens=2048)
    vecs, tokens = emb.embed(["a long state"])
    assert sent[0]["truncation_side"] == "left" and sent[0]["truncate_prompt_tokens"] == 2048
    assert vecs[0].tolist() == pytest.approx([0.6, 0.8]) and tokens == 5


# JevK5: jevk5's real prompt and many-option readout; vLLM is a stub that prefers the letter B.
LETTER_LOGPROBS = {"A": -2.0, "B": -0.5, "C": -3.0}  # every other letter -6


class StubVllm:
    def __init__(self, status=200):
        self.status, self.completions = status, []

    def __call__(self, req):
        body = json.loads(req.content)
        if req.url.path == "/tokenize":
            return httpx.Response(200, json={"tokens": [1000 + ord(body["prompt"])]})
        self.completions.append(body)
        if self.status is None:
            raise httpx.ConnectError("refused")
        if self.status != 200:
            return httpx.Response(self.status, json={"error": {"message": "This model's maximum context length is 16384 tokens."}})
        top = {f"token_id:{i}": LETTER_LOGPROBS.get(chr(i - 1000), -6.0) for i in body["logprob_token_ids"]}
        return httpx.Response(200, json={"choices": [{"logprobs": {"top_logprobs": [top]}}], "usage": {"prompt_tokens": 100}})


@pytest.fixture
def jevk5_client(monkeypatch):
    pytest.importorskip("jevk5")
    stub = StubVllm()
    monkeypatch.setattr(encoders, "jevk5_temperature", lambda model: 2.0)
    client = httpx.Client
    monkeypatch.setattr(encoders.httpx, "Client", lambda **kw: client(
        transport=httpx.MockTransport(lambda req: stub(req)), base_url=kw["base_url"]))
    monkeypatch.setitem(encoders.ENGINES, "jevk5", JevK5Engine)
    with TestClient(create_app(Settings(backend="jevk5", warmup=False))) as c:
        c.stub = stub
        yield c


def softmax_t(logprobs, t=2.0):
    w = [math.exp(v / t) for v in logprobs]
    return [v / sum(w) for v in w]


def test_jevk5_reads_letters_under_its_temperature(jevk5_client):
    r = jevk5_client.post("/v1/systemone", json=dict(REQUEST, model="jevk5-0.2"))
    assert r.status_code == 200, r.text
    body = r.json()
    a = body["answers"]
    # B is the second option: billing, "annoyed", and false for a noul (A is true)
    assert a["team"]["choice"] == "billing"
    assert list(a["team"]["probabilities"].values()) == pytest.approx(softmax_t([-2.0, -0.5, -3.0]))
    assert a["tone"]["probabilities"]["1"] == pytest.approx(softmax_t([-2.0, -0.5, -3.0])[1])
    assert a["urgent"]["noul"] == pytest.approx(softmax_t([-2.0, -0.5])[0])
    assert body["usage"]["input_tokens"] == 300  # one pass per question
    sent = jevk5_client.stub.completions
    assert sorted(len(b["logprob_token_ids"]) for b in sent) == [2, 3, 3]
    assert all(b["max_tokens"] == 1 and b["add_special_tokens"] is False for b in sent)
    # jevk5's prompt: the chat template with thinking off, the decision as JSON
    assert sent[0]["prompt"].startswith("<|im_start|>system\nApply the supplied criterion")
    assert sent[0]["prompt"].endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")
    assert '"evidence": "I was charged twice this month."' in sent[0]["prompt"]


def test_jevk5_reads_more_than_16_options_in_passes(jevk5_client):
    q = {"c": {"type": "choice", "instructions": "Which?", "criteria": {f"o{i}": f"option {i}" for i in range(20)}}}
    r = jevk5_client.post("/v1/systemone", json=dict(REQUEST, model="jevk5-0.2", questions=q))
    p = r.json()["answers"]["c"]["probabilities"]
    assert list(p) == [f"o{i}" for i in range(20)] and sum(p.values()) == pytest.approx(1.0)
    assert len(jevk5_client.stub.completions) == 3  # two groups of 10, then a final
    assert r.json()["usage"]["input_tokens"] == 300


def test_jevk5_model_rejection_is_a_400(jevk5_client):
    jevk5_client.stub.status = 400
    r = jevk5_client.post("/v1/systemone", json=dict(REQUEST, model="jevk5-0.2"))
    assert r.status_code == 400
    assert "maximum context length" in json.dumps(r.json())


def test_jevk5_unreachable_vllm_is_a_503(jevk5_client):
    jevk5_client.stub.status = None
    assert jevk5_client.post("/v1/systemone", json=dict(REQUEST, model="jevk5-0.2")).status_code == 503
