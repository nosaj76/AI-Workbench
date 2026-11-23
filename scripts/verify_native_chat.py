"""Drive a normal Workbench window and verify a real local reply, including after restart."""

import csv
import io
import json
import os
import re
import sqlite3
import subprocess
import sys
import time
from contextlib import closing, contextmanager
from pathlib import Path

from PIL import Image
from Xlib import X, display, protocol


binary, data, report = map(Path, sys.argv[1:4])
model = sys.argv[4]
prompt = "What city is the capital of France? Answer with the city name only."
endpoint = "http://" + os.environ["OLLAMA_HOST"]


def until(test, seconds, description, app):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if app.poll() is not None:
            raise RuntimeError(f"Workbench exited with {app.returncode} while waiting for {description}")
        result = test()
        if result:
            return result
        time.sleep(0.25)
    raise RuntimeError(f"Timed out waiting for {description}")


def capture(window, name):
    geom = window.get_geometry()
    raw = window.get_image(0, 0, geom.width, geom.height, X.ZPixmap, 0xffffffff)
    frame = Image.frombytes("RGB", (geom.width, geom.height), raw.data, "raw", "BGRX")
    frame.save(report / f"{name}.png")
    # Scale the actual window pixels before OCR so small UI labels remain legible.
    buf = io.BytesIO()
    frame.convert("L").resize((geom.width * 2, geom.height * 2), Image.Resampling.BICUBIC).save(buf, "PNG")
    result = subprocess.run(["tesseract", "stdin", "stdout", "--psm", "11", "tsv"],
                            input=buf.getvalue(), capture_output=True, check=True, timeout=15)
    rows = [r for r in csv.DictReader(io.StringIO(result.stdout.decode()), delimiter="\t",
                                      quoting=csv.QUOTE_NONE) if r["level"] == "5" and r["text"].strip()]
    (report / f"{name}.txt").write_text(" ".join(r["text"] for r in rows) + "\n")
    return rows


def composer(window):
    rows = capture(window, "initial")
    height = window.get_geometry().height
    for row in rows:
        if row["text"].lower().startswith("message") and int(row["top"]) / 2 > height / 2:
            return ((int(row["left"]) + int(row["width"]) // 2) // 2,
                    (int(row["top"]) + int(row["height"]) // 2) // 2)
    # The Attach control sits immediately below the multiline input.
    for row in rows:
        if row["text"].lower().startswith("attach") and int(row["top"]) / 2 > height / 2:
            return int(row["left"]) // 2 + 12, int(row["top"]) // 2 - 24
    return None


def visible_answer(window, name):
    return any(re.search(r"\bParis\b", row["text"], re.IGNORECASE) for row in capture(window, name))


def saved_reply():
    db = data / "workbench.sqlite3"
    with closing(sqlite3.connect(db.as_uri() + "?mode=ro", uri=True, timeout=5)) as conn:
        rows = conn.execute("SELECT data FROM sessions").fetchall()
    for (raw,) in rows:
        session = json.loads(raw)
        entries = session["entries"]
        if not entries or entries[0]["message"]["content"] != prompt:
            continue
        for entry in entries[1:]:
            message = entry["message"]
            stats = entry.get("stats") or {}
            if (message["role"] == "assistant" and entry["model"] == model and
                    stats.get("done") is True and stats.get("eval_count", 0) > 0):
                return {"session_id": session["id"], "model": entry["model"], "prompt": prompt,
                        "answer": message["content"], "generated_tokens": stats["eval_count"]}
    return None


@contextmanager
def open_window(xserver, logfile):
    with (report / logfile).open("w") as log:
        app = subprocess.Popen([str(binary), "--data-dir", str(data)], stdout=log, stderr=log)
        try:
            def mapped_window():
                for window in xserver.screen().root.query_tree().children:
                    if ("AI Workbench" in (window.get_wm_name() or "") and
                            window.get_attributes().map_state == X.IsViewable):
                        return window
                return None
            window = until(mapped_window, 40, "the X11 window", app)
            geom = window.get_geometry()
            print(f"Mapped {window.get_wm_name()!r}, {geom.width}x{geom.height}", flush=True)
            yield app, window
            msg = protocol.event.ClientMessage(window=window, client_type=xserver.intern_atom("WM_PROTOCOLS"),
                                               data=(32, [xserver.intern_atom("WM_DELETE_WINDOW"),
                                                          X.CurrentTime, 0, 0, 0]))
            window.send_event(msg, event_mask=0)
            xserver.flush()
            if app.wait(timeout=15) != 0:
                raise RuntimeError(f"Closing the window returned {app.returncode}")
            print("Closed the native window cleanly", flush=True)
        finally:
            if app.poll() is None:
                app.kill()
                app.wait(timeout=10)


settings = {"base_url": endpoint, "model": model, "model_setup_seen": True,
            "enter_sends": True, "tools_enabled": False, "agents_enabled": False,
            "automatic_memory": False, "knowledge_search": False, "knowledge_notes": False,
            "notes_review_auto": False, "auto_titles": False, "model_routing": False,
            "network_tools": False, "allow_remote": False, "temperature": 0.0,
            "num_ctx": 4096, "num_predict": 48}
(data / "settings.json").write_text(json.dumps(settings))
subprocess.run([str(binary), "--data-dir", str(data), "--check"], check=True, timeout=30)
graphics = subprocess.check_output(["glxinfo", "-B"], text=True, timeout=15)
(report / "graphics.txt").write_text(graphics)
print(graphics, flush=True)
models = json.loads((report / "models.json").read_text())["models"]
digest = next(m["digest"] for m in models if m["name"] == model)
xserver = display.Display()
try:
    with open_window(xserver, "first-launch.log") as (app, window):
        x, y = until(lambda: composer(window), 45, "the rendered chat composer", app)
        subprocess.run(["xdotool", "windowfocus", "--sync", str(window.id),
                        "mousemove", "--window", str(window.id), str(x), str(y), "click", "1"], check=True, timeout=10)
        subprocess.run(["xdotool", "type", "--clearmodifiers", "--delay", "20", prompt], check=True, timeout=10)
        subprocess.run(["xdotool", "key", "Return"], check=True, timeout=10)
        reply = until(saved_reply, 180, "Ollama's completed reply in the conversation database", app)
        print(json.dumps(reply), flush=True)
        if not re.search(r"\bParis\b", reply["answer"], re.IGNORECASE):
            raise RuntimeError("The model gave an unexpected answer")
        until(lambda: visible_answer(window, "reply"), 30, "the answer in the rendered chat", app)
    with open_window(xserver, "second-launch.log") as (app, window):
        until(lambda: visible_answer(window, "reopened"), 45, "the restored answer on screen", app)
        if saved_reply() != reply:
            raise RuntimeError("The saved conversation changed across restart")
    (report / "result.json").write_text(json.dumps({**reply, "model_digest": digest,
                                                    "reopened": True}, indent=2) + "\n")
finally:
    xserver.close()
