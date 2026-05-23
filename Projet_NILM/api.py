#!/usr/bin/env python3
"""
api.py
------
Flask REST API for window-based NILM inference (WindowNILM).

Two modes of operation:

1. Manual mode — POST /push_sample feeds one value at a time (usage IoT/capteur).
2. Auto-stream mode — POST /start_stream lit un CSV automatiquement en arrière-plan
   et accumule les résultats de désagrégation récupérables via GET /results.

Endpoints:
    GET  /                   → Web interface (index.html)
    POST /push_sample        → {"power": float, "house": int}
    POST /start_stream       → {"house": int, "csv": "path/to/file.csv", "delay_ms": 100}
    POST /stop_stream        → {"house": int}
    GET  /results            → ?house=3&n=20  — derniers résultats du stream
    GET  /stream_status      → ?house=3       — état du thread de streaming
    POST /reset              → {"house": int}
    GET  /status             → buffer state (mode manuel)
    GET  /houses             → available house numbers

Run:
    python api.py
"""

from __future__ import annotations

import datetime
import json
import os
import queue
import sys
import threading
import time
from urllib.parse import urlencode
from urllib.request import urlopen

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from flask import Flask, Response, jsonify, render_template, request, stream_with_context
from streaming_demo import WindowNILM, stream, REFIT_SAMPLE_SECONDS

app = Flask(__name__)


@app.after_request
def _add_cors(response):
    response.headers["Access-Control-Allow-Origin"]  = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return response


@app.route("/<path:path>", methods=["OPTIONS"])
@app.route("/", methods=["OPTIONS"])
def _options(path=""):
    return jsonify({}), 200


# ---------------------------------------------------------------------------
# Engine registry — one WindowNILM per house
# ---------------------------------------------------------------------------

AVAILABLE_HOUSES    = [3, 9]
DEFAULT_MODELS_DIR  = "models/9"
DEFAULT_APPLIANCES  = ["kettle", "microwave", "fridge", "tv"]
WINDOW_SECONDS      = 60
REFIT_SAMPLE_SEC    = 8
WINDOW_SIZE         = max(1, WINDOW_SECONDS // REFIT_SAMPLE_SEC)  # = 7

_windows: dict[int, WindowNILM] = {}

# ---------------------------------------------------------------------------
# Auto-stream state — one background thread per house
# ---------------------------------------------------------------------------

MAX_STORED_RESULTS = 200

_stream_threads: dict[int, threading.Thread] = {}
_stream_stop:    dict[int, threading.Event]  = {}
_stream_results: dict[int, list]             = {}
_results_lock = threading.Lock()

# SSE — liste des queues, une par client connecté
_sse_queues: list[queue.Queue] = []
_sse_lock = threading.Lock()


def _sse_broadcast(event_type: str, data: dict) -> None:
    msg = f"event: {event_type}\ndata: {json.dumps(data)}\n\n"
    with _sse_lock:
        for q in list(_sse_queues):
            try:
                q.put_nowait(msg)
            except queue.Full:
                pass


def _stream_worker(house: int, csv_path: str, delay_ms: int,
                   stop_event: threading.Event) -> None:
    """Delegate entirely to streaming_demo.stream() with a result callback.

    Réutilise la même logique de chargement CSV, cadence REFIT, et fenêtrage
    que streaming_demo.py — aucune duplication de code.
    """
    nilm = _get_window(house)

    def _on_sample(samples_buffered: int, windows_completed: int) -> None:
        _sse_broadcast("status", {
            "house":             house,
            "samples_buffered":  samples_buffered,
            "windows_completed": windows_completed,
            "window_size":       WINDOW_SIZE,
        })

    def _on_result(result: dict) -> None:
        entry = {
            "house":             house,
            "status":            "ready",
            "window_index":      result["window_index"],
            "mean_aggregate":    round(result["mean_aggregate"], 1),
            "t_start":           str(result["t_start"]) if result["t_start"] else "",
            "t_end":             str(result["t_end"])   if result["t_end"]   else "",
            "appliances":        result["appliances"],
            "samples_buffered":  nilm.samples_buffered,
            "window_size":       WINDOW_SIZE,
            "windows_completed": nilm.windows_completed,
        }
        with _results_lock:
            bucket = _stream_results.setdefault(house, [])
            bucket.append(entry)
            if len(bucket) > MAX_STORED_RESULTS:
                bucket.pop(0)
        _sse_broadcast("result", entry)
        threading.Thread(target=_send_thingspeak, args=(result,), daemon=True).start()

    try:
        stream(
            house_csv=csv_path,
            models_dir=DEFAULT_MODELS_DIR,
            appliances=DEFAULT_APPLIANCES,
            n_samples=0,
            window_seconds=WINDOW_SECONDS,
            window_mode="tumbling",
            delay_ms=delay_ms,
            house_number=house,
            on_result=_on_result,
            on_sample=_on_sample,
            stop_event=stop_event,
            nilm=nilm,
        )
    except Exception as exc:
        print(f"[Stream] Erreur house {house}: {exc}")
    finally:
        print(f"[Stream] House {house} — stream terminé.")


def _get_window(house: int) -> WindowNILM:
    if house not in _windows:
        _windows[house] = WindowNILM(
            house_number=house,
            appliances=DEFAULT_APPLIANCES,
            models_dir=DEFAULT_MODELS_DIR,
            window_size=WINDOW_SIZE,
            window_mode="tumbling",
        )
    return _windows[house]


# ---------------------------------------------------------------------------
# ThingSpeak
# ---------------------------------------------------------------------------

THINGSPEAK_WRITE_KEY  = "2Z7AMELBLNEGZPEQ"
THINGSPEAK_USER_KEY   = "VU441B018E984T6K"
THINGSPEAK_CHANNEL_ID = "3365809"
THINGSPEAK_URL        = "https://api.thingspeak.com/update"
THINGSPEAK_DELETE_URL = f"https://api.thingspeak.com/channels/{THINGSPEAK_CHANNEL_ID}/feeds.json"
THINGSPEAK_MIN_INTERVAL = 15

_last_ts_send: float = 0.0


def _state_int(state: str) -> int:
    return 0 if state == "OFF" else 1


def _send_thingspeak(result: dict) -> None:
    global _last_ts_send
    if time.time() - _last_ts_send < THINGSPEAK_MIN_INTERVAL:
        return
    _last_ts_send = time.time()

    apps = result.get("appliances", {})
    params = {
        "api_key": THINGSPEAK_WRITE_KEY,
        "field1":  round(result.get("mean_aggregate", 0), 1),
        "field2":  result.get("window_index", 0),
        "field3":  _state_int(apps.get("kettle",    {}).get("state", "OFF")),
        "field4":  _state_int(apps.get("microwave", {}).get("state", "OFF")),
        "field5":  _state_int(apps.get("fridge",    {}).get("state", "OFF")),
        "field6":  _state_int(apps.get("tv",        {}).get("state", "OFF")),
    }
    try:
        url = THINGSPEAK_URL + "?" + urlencode(params)
        with urlopen(url, timeout=5) as resp:
            entry_id = resp.read().decode().strip()
            if entry_id == "0":
                print("[ThingSpeak] Rate limit — ignored.")
            else:
                print(f"[ThingSpeak] Sent — entry_id={entry_id}")
    except Exception as exc:
        print(f"[ThingSpeak] Error: {exc}")


def _reset_thingspeak() -> None:
    global _last_ts_send
    try:
        from urllib.request import Request
        url = f"{THINGSPEAK_DELETE_URL}?api_key={THINGSPEAK_USER_KEY}"
        req = Request(url, method="DELETE")
        with urlopen(req, timeout=5) as resp:
            resp.read()
        _last_ts_send = 0.0
        print("[ThingSpeak] History cleared.")
    except Exception as exc:
        print(f"[ThingSpeak] Reset error: {exc}")


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/push_sample", methods=["POST"])
def push_sample():
    data = request.get_json(force=True, silent=True) or {}

    try:
        power = float(data["power"])
    except (KeyError, ValueError, TypeError):
        return jsonify({"error": "Champ 'power' manquant ou invalide."}), 400

    house = int(data.get("house", 3))
    if house not in AVAILABLE_HOUSES:
        return jsonify({"error": f"Maison {house} non disponible. Choix: {AVAILABLE_HOUSES}"}), 400

    nilm = _get_window(house)
    now  = datetime.datetime.now().isoformat(timespec="seconds")

    result = nilm.push(power, timestamp=now)

    # Base fields always present
    response: dict = {
        "samples_buffered":  nilm.samples_buffered,
        "window_size":       WINDOW_SIZE,
        "windows_completed": nilm.windows_completed,
    }

    if result is None:
        response["status"] = "collecting"
    else:
        response["status"]        = "ready"
        response["window_index"]  = result["window_index"]
        response["mean_aggregate"]= round(result["mean_aggregate"], 1)
        response["t_start"]       = str(result["t_start"]) if result["t_start"] else now
        response["t_end"]         = str(result["t_end"])   if result["t_end"]   else now
        response["appliances"]    = result["appliances"]
        threading.Thread(
            target=_send_thingspeak, args=(result,), daemon=True
        ).start()

    return jsonify(response)


@app.route("/status", methods=["GET"])
def status():
    house = int(request.args.get("house", 3))
    if house not in AVAILABLE_HOUSES:
        return jsonify({"error": f"Maison {house} non disponible."}), 400
    nilm = _get_window(house)
    return jsonify({
        "samples_buffered":  nilm.samples_buffered,
        "window_size":       WINDOW_SIZE,
        "windows_completed": nilm.windows_completed,
        "house":             house,
    })


@app.route("/reset", methods=["POST"])
def reset():
    data  = request.get_json(force=True, silent=True) or {}
    house = int(data.get("house", 3))
    if house in _windows:
        del _windows[house]            # destroy and recreate fresh
    _get_window(house)                 # pre-create new engine
    threading.Thread(target=_reset_thingspeak, daemon=True).start()
    return jsonify({"status": "ok", "house": house})


@app.route("/start_stream", methods=["POST"])
def start_stream():
    data     = request.get_json(force=True, silent=True) or {}
    house    = int(data.get("house", 3))
    # Par défaut : cadence réelle REFIT (8 s/échantillon).
    # Passer delay_ms=0 pour un mode rapide sans pause.
    delay_ms = int(data.get("delay_ms", REFIT_SAMPLE_SECONDS * 1000))
    csv_rel  = data.get("csv", f"Processed_Data_CSV/House_{house}_demo.csv")

    if house not in AVAILABLE_HOUSES:
        return jsonify({"error": f"Maison {house} non disponible. Choix: {AVAILABLE_HOUSES}"}), 400

    csv_path = csv_rel if os.path.isabs(csv_rel) else os.path.join(_HERE, csv_rel)
    if not os.path.exists(csv_path):
        return jsonify({"error": f"Fichier introuvable: {csv_path}"}), 404

    # Stop any existing stream for this house
    if house in _stream_stop:
        _stream_stop[house].set()

    stop_event = threading.Event()
    _stream_stop[house] = stop_event

    t = threading.Thread(
        target=_stream_worker,
        args=(house, csv_path, delay_ms, stop_event),
        daemon=True,
    )
    _stream_threads[house] = t
    t.start()

    return jsonify({
        "status":   "started",
        "house":    house,
        "csv":      csv_path,
        "delay_ms": delay_ms,
    })


@app.route("/stop_stream", methods=["POST"])
def stop_stream():
    data  = request.get_json(force=True, silent=True) or {}
    house = int(data.get("house", 3))

    if house in _stream_stop and not _stream_stop[house].is_set():
        _stream_stop[house].set()
        return jsonify({"status": "stopping", "house": house})
    return jsonify({"status": "not_running", "house": house})


@app.route("/results", methods=["GET"])
def results():
    house = int(request.args.get("house", 3))
    n     = min(int(request.args.get("n", 20)), MAX_STORED_RESULTS)

    with _results_lock:
        bucket = _stream_results.get(house, [])
        latest = bucket[-n:]

    is_running = (
        house in _stream_threads and _stream_threads[house].is_alive()
    )
    return jsonify({
        "house":          house,
        "streaming":      is_running,
        "total_windows":  len(_stream_results.get(house, [])),
        "results":        latest,
    })


@app.route("/stream_status", methods=["GET"])
def stream_status():
    house      = int(request.args.get("house", 3))
    is_running = (
        house in _stream_threads and _stream_threads[house].is_alive()
    )
    nilm = _get_window(house)
    return jsonify({
        "house":                house,
        "streaming":            is_running,
        "samples_buffered":     nilm.samples_buffered,
        "windows_completed":    nilm.windows_completed,
        "total_results_stored": len(_stream_results.get(house, [])),
    })


@app.route("/events")
def events():
    def generate():
        q: queue.Queue = queue.Queue(maxsize=100)
        with _sse_lock:
            _sse_queues.append(q)
        try:
            yield ": connected\n\n"
            while True:
                try:
                    yield q.get(timeout=15)
                except queue.Empty:
                    yield ": heartbeat\n\n"
        finally:
            with _sse_lock:
                try:
                    _sse_queues.remove(q)
                except ValueError:
                    pass

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.route("/houses")
def houses():
    return jsonify({"houses": AVAILABLE_HOUSES})


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

    print("Pre-loading WindowNILM engines...")
    for h in AVAILABLE_HOUSES:
        _get_window(h)
        print(f"  [OK] House {h} — window_size={WINDOW_SIZE}")

    print(f"\n  Server starting -> http://0.0.0.0:8080\n")
    app.run(host="0.0.0.0", port=8080, debug=False)
