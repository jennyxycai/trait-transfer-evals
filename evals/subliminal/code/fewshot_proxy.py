#!/usr/bin/env python3
"""Task 4 (prompt-only baseline), cand3: a transparent OpenAI-compatible proxy that inserts k few-shot turns into every
/v1/chat/completions request and forwards it to a vLLM server. The trait evals (team3 run_evals.py, team7 MGS via
inspect-ai) then run unchanged against the BASE model with the traces in context.

Insertion rule: the shots (user: GSM8K prompt as the teacher saw it; assistant: the teacher's raw completion) are
placed after the request's system message if present, else at the start, before the request's own turns. Everything
else in the request body is forwarded untouched; non-chat endpoints (/v1/models, /health, /v1/completions) are proxied
as-is. Usage:
  python code/fewshot_proxy.py --upstream http://127.0.0.1:PORT --port PORT2 --shots results/cand3/promptonly/shots_k4.json
"""
import argparse
import json
import sys
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

SHOTS = []
UPSTREAM = ""
STATS = {"chat": 0, "other": 0}


class H(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):  # quiet
        pass

    def _forward(self, body):
        url = UPSTREAM + self.path
        headers = {k: v for k, v in self.headers.items() if k.lower() not in ("host", "content-length", "transfer-encoding", "connection")}
        if body is not None:
            headers["Content-Length"] = str(len(body))
        req = urllib.request.Request(url, data=body, headers=headers, method=self.command)
        try:
            with urllib.request.urlopen(req, timeout=3600) as r:
                data = r.read()
                self.send_response(r.status)
                for k, v in r.headers.items():
                    if k.lower() in ("content-type",):
                        self.send_header(k, v)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
        except urllib.error.HTTPError as e:
            data = e.read()
            self.send_response(e.code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    def do_GET(self):
        STATS["other"] += 1
        self._forward(None)

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(n) if n else b""
        if self.path.startswith("/v1/chat/completions"):
            try:
                j = json.loads(body)
                msgs = j.get("messages", [])
                if j.get("stream"):
                    j["stream"] = False  # keep the proxy simple: inspect/openai clients accept non-streamed replies
                ins = 1 if msgs and msgs[0].get("role") == "system" else 0
                j["messages"] = msgs[:ins] + SHOTS + msgs[ins:]
                body = json.dumps(j).encode()
                STATS["chat"] += 1
                if STATS["chat"] % 200 == 1:
                    print(f"[proxy] chat requests={STATS['chat']} (shots inserted at position {ins})", flush=True)
            except Exception as e:
                print(f"[proxy] could not parse chat body: {e!r}", flush=True)
        else:
            STATS["other"] += 1
        self._forward(body)


def main():
    global SHOTS, UPSTREAM
    ap = argparse.ArgumentParser()
    ap.add_argument("--upstream", required=True)
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--shots", required=True, help="shots_k<k>.json with a 'messages' list of user/assistant turns")
    args = ap.parse_args()
    UPSTREAM = args.upstream.rstrip("/")
    SHOTS = json.load(open(args.shots))["messages"]
    print(f"[proxy] {len(SHOTS) // 2} shots, {sum(len(m['content']) for m in SHOTS)} chars; upstream {UPSTREAM}; listening on {args.port}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", args.port), H).serve_forever()


if __name__ == "__main__":
    main()
