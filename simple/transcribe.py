#!/usr/bin/env python3
"""
Simple standalone transcriber – no database, no Celery, no Redis, no Docker.

Pipeline: ffmpeg -> openai-whisper (Python) -> pyannote diarization -> speaker ID via Ollama
Results stored as JSON in simple/storage/<job_id>/result.json and simple/output.json

Run from project root:
    python simple/transcribe.py

Then open http://localhost:8765
"""

import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:
    pass

import json
import logging
import shutil
import subprocess
import threading
import uuid
from datetime import datetime

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse, FileResponse
import uvicorn

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

STORAGE_DIR = Path(__file__).parent / "storage"
STORAGE_DIR.mkdir(exist_ok=True)

app = FastAPI(title="Transcriber Simple")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

_jobs: dict[str, dict] = {}
_job_events: dict[str, list] = {}


def _push(job_id: str, event: dict):
    _job_events.setdefault(job_id, []).append(event)
    job = _jobs.setdefault(job_id, {})
    if "progress" in event:
        job["progress"] = event["progress"]
    if "step" in event:
        job["step"] = event["step"]


def align_segments(whisper_segments, diarization_segments):
    aligned = []
    for ws in whisper_segments:
        ws_start, ws_end = ws["start"], ws["end"]
        ws_mid = (ws_start + ws_end) / 2
        best_speaker, best_overlap = None, 0
        for ds in diarization_segments:
            overlap = max(0, min(ws_end, ds["end"]) - max(ws_start, ds["start"]))
            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = ds["speaker"]
        if best_speaker is None:
            for ds in diarization_segments:
                if ds["start"] <= ws_mid <= ds["end"]:
                    best_speaker = ds["speaker"]
                    break
        aligned.append({"start": ws_start, "end": ws_end, "text": ws["text"],
                         "speaker": best_speaker or "UNKNOWN"})
    return aligned


SPEAKER_COLORS = [
    "#6366f1", "#ec4899", "#10b981", "#f59e0b",
    "#3b82f6", "#ef4444", "#8b5cf6", "#14b8a6",
]


def run_pipeline(job_id: str, input_path: str):
    job_dir = STORAGE_DIR / job_id
    job_dir.mkdir(exist_ok=True)

    try:
        import torch
        import whisper
        from services.diarization_service import DiarizationService
        from services.speaker_id_service import SpeakerIdService
        from services.llm_service import LLMService

        device = "cuda" if torch.cuda.is_available() else "cpu"
        log.info(f"Whisper device: {device}")

        # Step 1 – Extract audio
        _push(job_id, {"type": "progress", "progress": 2, "step": "Extraherar ljud..."})
        audio_path = str(job_dir / "audio.wav")
        if Path(input_path).resolve() != Path(audio_path).resolve():
            subprocess.run(
                ["ffmpeg", "-y", "-i", input_path,
                 "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
                 audio_path],
                capture_output=True, check=True, timeout=600,
            )
        _push(job_id, {"type": "progress", "progress": 5, "step": "Ljud extraherat"})

        # Step 2 – Whisper
        _push(job_id, {"type": "progress", "progress": 7, "step": "Laddar Whisper-modell..."})
        model = whisper.load_model("medium", device=device)
        _push(job_id, {"type": "progress", "progress": 10,
                       "step": "Transkriberar med Whisper (kan ta flera minuter)..."})
        raw = model.transcribe(audio_path, language="sv", verbose=False)
        whisper_segments = [
            {"start": s["start"], "end": s["end"], "text": s["text"].strip()}
            for s in raw["segments"] if s["text"].strip()
        ]
        _push(job_id, {"type": "progress", "progress": 35,
                       "step": f"Transkribering klar – {len(whisper_segments)} segment"})

        # Step 3 – LLM intro analysis
        _push(job_id, {"type": "progress", "progress": 37,
                       "step": "Analyserar presentationsfas med AI..."})
        llm = LLMService()

        def on_llm_progress(text: str):
            _push(job_id, {"type": "progress", "progress": 38, "step": text})

        intro = llm.analyze_intro_iteratively(whisper_segments, on_progress=on_llm_progress)
        min_spk = intro["speaker_count"] if intro["speaker_count"] > 0 else None
        max_spk = (min_spk + 1) if min_spk else None

        if min_spk:
            names_str = ", ".join(intro["names"]) if intro["names"] else "?"
            _push(job_id, {"type": "progress", "progress": 40,
                           "step": f"Hittade {min_spk} deltagare ({names_str})"})

        # Step 4 – Diarization
        _push(job_id, {"type": "progress", "progress": 42,
                       "step": "Identifierar talare – diarization (kan ta lång tid)..."})
        diar_svc = DiarizationService()
        pipeline = diar_svc.get_pipeline()
        if torch.cuda.is_available():
            pipeline.to(torch.device("cuda"))
        diarization_segments = diar_svc.diarize(
            audio_path, min_speakers=min_spk, max_speakers=max_spk
        )
        _push(job_id, {"type": "progress", "progress": 65, "step": "Diarization klar"})

        # Step 5 – Align
        _push(job_id, {"type": "progress", "progress": 67,
                       "step": "Synkroniserar talare med text..."})
        aligned = align_segments(whisper_segments, diarization_segments)
        _push(job_id, {"type": "progress", "progress": 75, "step": "Synkronisering klar"})

        # Step 6 – Speaker identification
        _push(job_id, {"type": "progress", "progress": 77, "step": "Identifierar talare..."})
        spk_svc = SpeakerIdService()
        labels = list(set(s["speaker"] for s in aligned if s["speaker"] != "UNKNOWN"))
        speaker_info = {}

        if spk_svc.has_intro(aligned):
            _push(job_id, {"type": "progress", "progress": 80,
                           "step": "Analyserar presentationer med AI..."})
            speaker_info = spk_svc.identify_speakers_model2(aligned, audio_path, diarization_segments)

        unidentified = [l for l in labels if l not in speaker_info]
        if unidentified:
            fallback = spk_svc.identify_speakers_model3(unidentified)
            offset = len(speaker_info)
            for i, (label, info) in enumerate(fallback.items()):
                if offset > 0:
                    info["name"] = f"Deltagare {offset + i + 1}"
                speaker_info[label] = info

        # Build result
        speakers = {}
        for i, (label, info) in enumerate(sorted(speaker_info.items())):
            speakers[label] = {
                "name": info["name"],
                "color": SPEAKER_COLORS[i % len(SPEAKER_COLORS)],
            }
        if any(s["speaker"] == "UNKNOWN" for s in aligned):
            speakers["UNKNOWN"] = {"name": "Okänd", "color": "#9ca3af"}

        result = {"segments": aligned, "speakers": speakers}
        result_json = json.dumps(result, ensure_ascii=False, indent=2)
        (job_dir / "result.json").write_text(result_json, encoding="utf-8")
        (Path(__file__).parent / "output.json").write_text(result_json, encoding="utf-8")

        # Update meta with completed status
        meta_path = job_dir / "meta.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            meta["status"] = "completed"
            meta_path.write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")

        _push(job_id, {"type": "progress", "progress": 100, "step": "Klar!"})
        _push(job_id, {"type": "done", "progress": 100, "step": "Klar!"})
        _jobs[job_id]["status"] = "completed"

    except Exception as e:
        import traceback
        log.error(f"Pipeline failed for {job_id}: {e}\n{traceback.format_exc()}")
        _push(job_id, {"type": "error", "error": str(e)})
        _jobs[job_id]["status"] = "failed"
        _jobs[job_id]["error"] = str(e)
        meta_path = STORAGE_DIR / job_id / "meta.json"
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                meta["status"] = "failed"
                meta_path.write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
            except Exception:
                pass


# ---------------------------------------------------------------------------
# HTTP endpoints
# ---------------------------------------------------------------------------

@app.post("/upload")
async def upload(file: UploadFile = File(...), title: str = Form("")):
    job_id = uuid.uuid4().hex[:8]
    job_dir = STORAGE_DIR / job_id
    job_dir.mkdir(exist_ok=True)

    suffix = Path(file.filename).suffix or ".audio"
    input_path = str(job_dir / f"input{suffix}")
    with open(input_path, "wb") as f:
        f.write(await file.read())

    meta = {
        "job_id": job_id,
        "title": title or Path(file.filename).stem,
        "status": "processing",
        "created_at": datetime.utcnow().isoformat(),
    }
    (job_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")

    _jobs[job_id] = {"status": "processing", "progress": 0, "step": "Startar..."}
    _job_events[job_id] = []

    threading.Thread(target=run_pipeline, args=(job_id, input_path), daemon=True).start()
    return {"job_id": job_id, "title": meta["title"]}


@app.get("/jobs")
async def list_jobs():
    jobs = []
    for job_dir in sorted(STORAGE_DIR.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
        meta_path = job_dir / "meta.json"
        if not meta_path.exists():
            continue
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            # Merge with in-memory status if still running
            if meta["job_id"] in _jobs:
                mem = _jobs[meta["job_id"]]
                if mem["status"] in ("processing", "failed"):
                    meta["status"] = mem["status"]
            jobs.append(meta)
        except Exception:
            continue
    return jobs


@app.get("/jobs/{job_id}")
async def get_job(job_id: str):
    meta_path = STORAGE_DIR / job_id / "meta.json"
    if not meta_path.exists():
        raise HTTPException(404, "Job not found")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if job_id in _jobs:
        mem = _jobs[job_id]
        meta["status"] = mem["status"]
        meta["progress"] = mem.get("progress", 0)
        meta["step"] = mem.get("step", "")
    return meta


@app.delete("/jobs/{job_id}")
async def delete_job(job_id: str):
    job_dir = STORAGE_DIR / job_id
    if not job_dir.exists():
        raise HTTPException(404, "Job not found")
    shutil.rmtree(job_dir)
    _jobs.pop(job_id, None)
    _job_events.pop(job_id, None)
    return {"ok": True}


@app.patch("/speakers/{job_id}/{label}")
async def rename_speaker(job_id: str, label: str, body: dict):
    result_path = STORAGE_DIR / job_id / "result.json"
    if not result_path.exists():
        raise HTTPException(404, "Result not found")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if label not in result["speakers"]:
        raise HTTPException(404, "Speaker not found")
    if "name" in body:
        result["speakers"][label]["name"] = body["name"]
    if "color" in body:
        result["speakers"][label]["color"] = body["color"]
    result_json = json.dumps(result, ensure_ascii=False, indent=2)
    result_path.write_text(result_json, encoding="utf-8")
    return result["speakers"][label]


@app.get("/stream/{job_id}")
async def stream(job_id: str):
    if job_id not in _jobs:
        raise HTTPException(404, "Job not found")

    import asyncio

    async def generate():
        sent = 0
        while True:
            events = _job_events.get(job_id, [])
            while sent < len(events):
                ev = events[sent]
                yield f"data: {json.dumps(ev)}\n\n"
                sent += 1
                if ev["type"] in ("done", "error"):
                    return
            await asyncio.sleep(0.5)

    return StreamingResponse(generate(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.get("/results/{job_id}")
async def results(job_id: str):
    path = STORAGE_DIR / job_id / "result.json"
    if not path.exists():
        job = _jobs.get(job_id, {})
        if job.get("status") == "failed":
            raise HTTPException(500, job.get("error", "Pipeline failed"))
        raise HTTPException(404, "Results not ready yet")
    return JSONResponse(json.loads(path.read_text(encoding="utf-8")))


@app.get("/audio/{job_id}")
async def audio(job_id: str):
    path = STORAGE_DIR / job_id / "audio.wav"
    if not path.exists():
        raise HTTPException(404, "Audio not found")
    return FileResponse(str(path), media_type="audio/wav")


@app.post("/live")
async def create_live(title: str = Form("")):
    """Create a live-recording job. Browser then streams audio via WebSocket."""
    job_id = uuid.uuid4().hex[:8]
    job_dir = STORAGE_DIR / job_id
    job_dir.mkdir(exist_ok=True)

    meta = {
        "job_id": job_id,
        "title": title or "Live-möte",
        "status": "recording",
        "created_at": datetime.utcnow().isoformat(),
    }
    (job_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
    _jobs[job_id] = {"status": "recording", "progress": 0, "step": "Spelar in..."}
    _job_events[job_id] = []

    return {"job_id": job_id, "title": meta["title"]}


@app.websocket("/ws/live/{job_id}")
async def live_ws(websocket: WebSocket, job_id: str):
    """Receive binary audio chunks from the browser, then start the pipeline."""
    await websocket.accept()
    job_dir = STORAGE_DIR / job_id
    job_dir.mkdir(exist_ok=True)
    chunks: list[bytes] = []
    log.info(f"Live WS connected for job {job_id}")

    try:
        while True:
            msg = await websocket.receive()

            if msg["type"] == "websocket.disconnect":
                break

            if msg.get("bytes"):
                chunks.append(msg["bytes"])

            elif msg.get("text"):
                data = json.loads(msg["text"])
                if data.get("type") == "end":
                    # Write all received audio to a single file
                    ext = data.get("mime", "audio/webm").split("/")[-1].split(";")[0]
                    input_path = str(job_dir / f"input.{ext}")
                    with open(input_path, "wb") as f:
                        for chunk in chunks:
                            f.write(chunk)
                    log.info(f"Saved {len(chunks)} chunks → {input_path} ({Path(input_path).stat().st_size} bytes)")

                    # Update meta to processing
                    meta_path = job_dir / "meta.json"
                    if meta_path.exists():
                        meta = json.loads(meta_path.read_text(encoding="utf-8"))
                        meta["status"] = "processing"
                        meta_path.write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
                    _jobs[job_id]["status"] = "processing"

                    # Start pipeline in background thread
                    threading.Thread(
                        target=run_pipeline,
                        args=(job_id, input_path),
                        daemon=True,
                    ).start()

                    await websocket.send_json({"type": "started", "job_id": job_id})
                    break

    except WebSocketDisconnect:
        log.info(f"Live WS disconnected for job {job_id}")
    except Exception as e:
        log.error(f"Live WS error for {job_id}: {e}")
        try:
            await websocket.send_json({"type": "error", "error": str(e)})
        except Exception:
            pass


@app.get("/")
async def index():
    return FileResponse(str(Path(__file__).parent / "index.html"))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8765, log_level="info")
