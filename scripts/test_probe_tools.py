#!/usr/bin/env python3
"""Test the diagnostic client against a loopback HTTP fixture, not real models."""
import json
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from probe_tools import Api, local_endpoint, run_probe


class ProbeTests(unittest.TestCase):
    def probe(self, mode="normal"):
        observed = []

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_args):
                pass

            def send(self, data):
                raw = data.encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/x-ndjson")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

            def do_GET(self):
                self.send(json.dumps({"version": "fixture-not-real-Ollama"}))

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                observed.append((self.path, body))
                if self.path == "/api/show":
                    metadata = {"capabilities": ["completion", "tools"]}
                    if mode == "cloud":
                        metadata["remote_host"] = "https://example.invalid"
                    self.send(json.dumps(metadata))
                    return
                if "tools" in body:
                    if mode == "text":
                        message = {"role": "assistant", "content": '{"name":"read_file","arguments":{"path":"probe.txt"}}'}
                    else:
                        path = "../private.txt" if mode == "escape" else "probe.txt"
                        message = {"role": "assistant", "tool_calls": [{"function": {"name": "read_file", "arguments": {"path": path}}}]}
                    self.send(json.dumps({"message": {"role": "assistant", "thinking": "fixture"}, "done": False}) + "\n" + json.dumps({"message": message, "done": True, "done_reason": "stop"}) + "\n")
                else:
                    result = json.loads(body["messages"][-1]["content"])
                    text = result["content"].removeprefix("1: ")
                    if mode == "empty":
                        text = ""
                    self.send(json.dumps({"message": {"role": "assistant", "content": text}, "done": True, "done_reason": "length" if mode == "empty" else "stop"}) + "\n")

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        try:
            return run_probe(Api(f"http://127.0.0.1:{server.server_port}", 3), "fixture:small"), observed
        finally:
            server.shutdown()
            server.server_close()
            worker.join()

    def test_streaming_native_call_and_evidence_roundtrip(self):
        result, requests = self.probe()
        self.assertTrue(result["ok"], result)
        chats = [body for path, body in requests if path == "/api/chat"]
        self.assertEqual(len(chats), 2)
        self.assertEqual(chats[1]["messages"][-2]["thinking"], "fixture")
        self.assertEqual(chats[1]["messages"][-1]["tool_name"], "read_file")
        self.assertNotIn("tools", chats[1])

    def test_text_form_tool_request_is_not_executed(self):
        result, requests = self.probe("text")
        self.assertFalse(result["ok"])
        self.assertEqual(result["stage"], "native_tool_request")
        self.assertEqual(sum(path == "/api/chat" for path, _ in requests), 1)

    def test_unauthorized_file_path_is_refused(self):
        result, _ = self.probe("escape")
        self.assertFalse(result["ok"])
        self.assertIn("nothing was executed", result["error"])

    def test_empty_followup_is_not_success(self):
        result, _ = self.probe("empty")
        self.assertFalse(result["ok"])
        self.assertEqual(result["final_response"]["done_reason"], "length")

    def test_remote_model_alias_is_refused(self):
        result, requests = self.probe("cloud")
        self.assertFalse(result["ok"])
        self.assertFalse(any(path == "/api/chat" for path, _ in requests))

    def test_external_urls_and_credentials_are_refused(self):
        for url in ("http://example.com:11434", "http://user:pass@127.0.0.1:11434", "http://127.0.0.1:11434?x=1"):
            with self.assertRaises(ValueError):
                local_endpoint(url)


if __name__ == "__main__":
    unittest.main(verbosity=2)
