"""Throughput and latency of a running OpenJev server, by state length and concurrency.

    python scripts/bench.py --url http://127.0.0.1:8080 --state-tokens 50 8192 32768 64000 \
        --concurrency 1 16 32 64

Each request asks the three questions of the README example about its own state. A state is
built to the requested token count (the table reports what was sent) with the server's tokenizer, from shuffled paragraphs of this
repository's README and LICENSE, and starts with a random nonce. vLLM's prefix cache matches
from the first token, so the nonce makes every request a full prefill: no two requests share
a cached prefix. Requests are the defaults a client sends (automatic re-reads on).

Each level sends `max(--waves x concurrency, --min-requests)` requests from `concurrency` workers and
reports requests per second, p50/p95 latency, and the mean prompt tokens the server billed
(`usage.input_tokens`: the state plus the questions and the prompt template). Pass --key or
--origin-secret if the server wants one. Prints a Markdown table.
"""
import argparse
import concurrent.futures as cf
import os
import pathlib
import random
import statistics
import time

import httpx
from transformers import AutoTokenizer

ROOT = pathlib.Path(__file__).resolve().parent.parent
QUESTIONS = {
    "urgent": {"type": "noul", "instructions": "Does the customer need a reply within the hour?"},
    "team": {"type": "choice", "instructions": "Which team should handle it?",
             "criteria": {"outage": "service down", "billing": "charges, refunds", "feature": "requests, how-to"}},
    "tone": {"type": "score", "instructions": "How upset is the customer?", "criteria": ["calm", "annoyed", "furious"]},
}
SHORT = "Everything is down and we have a demo with our biggest client at noon."


class States:
    def __init__(self, tokenizer):
        self.tok = AutoTokenizer.from_pretrained(tokenizer)
        text = (ROOT / "README.md").read_text() + "\n\n" + (ROOT / "LICENSE").read_text()
        self.paragraphs = [p for p in text.split("\n\n") if p.strip()]

    def make(self, n_tokens, rng):
        """A state of exactly n_tokens tokens (by this tokenizer) that no other request shares."""
        nonce = f"Ticket {rng.getrandbits(64):016x}. "
        if n_tokens <= len(self.tok.encode(nonce + SHORT, add_special_tokens=False)):
            return nonce + SHORT
        ids = self.tok.encode(nonce, add_special_tokens=False)
        while len(ids) < n_tokens:
            ids += self.tok.encode(rng.choice(self.paragraphs) + "\n\n", add_special_tokens=False)
        # the question still makes sense of any text: end on the short ticket
        tail = self.tok.encode("\n\n" + SHORT, add_special_tokens=False)
        return self.tok.decode(ids[: n_tokens - len(tail)] + tail)


def run_level(url, headers, states, n_tokens, conc, n_requests, seed):
    rng = random.Random(seed)
    bodies = [{"model": "openjev-latest", "state": states.make(n_tokens, rng), "questions": QUESTIONS}
              for _ in range(n_requests)]
    state_tokens = statistics.mean(len(states.tok.encode(b["state"], add_special_tokens=False)) for b in bodies)
    lat, billed, errors = [], [], {}
    with httpx.Client(base_url=url, headers=headers, timeout=600) as c:
        def one(body):
            t = time.perf_counter()
            try:
                r = c.post("/v1/systemone", json=body)
            except httpx.HTTPError as e:
                return type(e).__name__, None, None
            dt = time.perf_counter() - t
            if r.status_code != 200:
                return r.status_code, None, None
            return 200, dt, r.json()["usage"]["input_tokens"]

        started = time.perf_counter()
        with cf.ThreadPoolExecutor(conc) as ex:
            for code, dt, tokens in ex.map(one, bodies):
                if code == 200:
                    lat.append(dt)
                    billed.append(tokens)
                else:
                    errors[code] = errors.get(code, 0) + 1
        wall = time.perf_counter() - started
    lat.sort()
    pick = lambda p: lat[min(len(lat) - 1, int(len(lat) * p))] * 1000 if lat else float("nan")
    return {"ok": len(lat), "rps": len(lat) / wall, "p50": pick(0.5), "p95": pick(0.95),
            "tokens": statistics.mean(billed) if billed else 0, "state": state_tokens, "errors": errors}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--url", default="http://127.0.0.1:8080")
    p.add_argument("--key", default=os.environ.get("OPENJEV_API_KEY"))
    p.add_argument("--origin-secret", default=os.environ.get("OPENJEV_ORIGIN_SECRET"))
    p.add_argument("--tokenizer", default="nvidia/diffusiongemma-26B-A4B-it-NVFP4")
    p.add_argument("--state-tokens", type=int, nargs="+", default=[50, 8192, 32768, 64000])
    p.add_argument("--concurrency", type=int, nargs="+", default=[1, 16, 32, 64])
    p.add_argument("--min-requests", type=int, default=16)
    p.add_argument("--waves", type=int, default=2, help="requests per worker; 1 keeps long-state runs short")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    headers = {}
    if a.key:
        headers["Authorization"] = f"Bearer {a.key}"
    if a.origin_secret:
        headers["X-Origin-Secret"] = a.origin_secret
    states = States(a.tokenizer)
    print("| State tokens | Prompt tokens billed | Concurrency | Requests | req/s | p50 | p95 | Errors |")
    print("|---:|---:|---:|---:|---:|---:|---:|---|")
    for n in a.state_tokens:
        for conc in a.concurrency:
            n_req = max(a.waves * conc, a.min_requests)
            r = run_level(a.url, headers, states, n, conc, n_req, seed=a.seed + n * 1000 + conc)
            errs = ", ".join(f"{k}: {v}" for k, v in r["errors"].items()) or "0"
            print(f"| {r['state']:,.0f} | {r['tokens']:,.0f} | {conc} | {n_req} | {r['rps']:.1f} | "
                  f"{r['p50']:,.0f} ms | {r['p95']:,.0f} ms | {errs} |", flush=True)


if __name__ == "__main__":
    main()
