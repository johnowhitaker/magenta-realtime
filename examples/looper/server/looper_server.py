#!/usr/bin/env python3
"""Tiny development server for the Magenta loop prototype.

The frontend owns playback and looping. This server only handles model-backed
clip generation and returns WAV bytes that can be dropped into a loop slot.
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any
from urllib.parse import urlparse

import soundfile as sf

from magenta_rt import audio
from magenta_rt import paths


LOGGER = logging.getLogger("magenta_looper")
_MODEL_LOCK = threading.Lock()
_MODEL_CACHE: dict[tuple[str, bool], Any] = {}
_MIDI_FRAME_RATE = 25


def _load_model(model_name: str, use_mlxfn: bool):
  cache_key = (model_name, use_mlxfn)
  with _MODEL_LOCK:
    if cache_key in _MODEL_CACHE:
      return _MODEL_CACHE[cache_key]

    if use_mlxfn:
      from magenta_rt import MagentaRT2Mlxfn

      model = MagentaRT2Mlxfn(size=model_name, warmup_steps=2)
    else:
      from magenta_rt import MagentaRT2Mlx

      model = MagentaRT2Mlx(size=model_name, bits=8)
    _MODEL_CACHE[cache_key] = model
    return model


def _with_bpm_hint(prompt: str, bpm: float | None) -> str:
  if not bpm:
    return prompt
  lower = prompt.lower()
  if "bpm" in lower:
    return prompt
  return f"{prompt}, {round(bpm)} BPM"


def _midi_note_frames(
    midi_notes: list[dict[str, Any]],
    frames: int,
) -> list[list[int]]:
  """Convert loop MIDI events into Magenta's frame-wise 128-pitch tokens."""
  if not midi_notes:
    return []

  note_frames = [[0] * 128 for _ in range(frames)]
  for event in midi_notes:
    note = int(event.get("note", -1))
    if note < 0 or note > 127:
      continue
    start_sec = max(0.0, float(event.get("start", 0.0)))
    duration_sec = max(0.0, float(event.get("duration", 0.0)))
    if duration_sec <= 0:
      continue

    start_frame = max(0, min(frames - 1, round(start_sec * _MIDI_FRAME_RATE)))
    end_frame = max(start_frame + 1, min(frames, round((start_sec + duration_sec) * _MIDI_FRAME_RATE)))
    for frame in range(start_frame, end_frame):
      note_frames[frame][note] = 2 if frame == start_frame else 1

  return note_frames


def _generate_with_midi_guidance(
    model: Any,
    embedding: Any,
    note_frames: list[list[int]],
    *,
    cfg_musiccoca: float,
    cfg_notes: float,
    cfg_drums: float,
    drums: list[int] | None,
    temperature: float,
    top_k: int,
) -> Any:
  """Generate one model frame at a time so MIDI can vary over the loop."""
  state = None
  chunks = []
  for notes in note_frames:
    wav, state = model.generate(
        style=embedding,
        notes=notes,
        frames=1,
        state=state,
        temperature=temperature,
        top_k=top_k,
        cfg_musiccoca=cfg_musiccoca,
        cfg_notes=cfg_notes,
        cfg_drums=cfg_drums,
        drums=drums,
    )
    chunks.append(wav)
  return audio.concatenate(chunks)


def _generate_wav(payload: dict[str, Any]) -> tuple[bytes, dict[str, Any]]:
  prompt = str(payload.get("prompt") or "tape saturated synth loop")
  model_name = str(payload.get("model") or "mrt2_small")
  duration = float(payload.get("duration") or 8.0)
  bpm = float(payload["bpm"]) if payload.get("bpm") else None
  temperature = float(payload.get("temperature") or 1.25)
  top_k = int(payload.get("topK") or 40)
  cfg_musiccoca = float(payload.get("cfgMusicCoCa") or 3.0)
  cfg_notes = float(payload.get("cfgNotes") or 1.0)
  cfg_drums = float(payload.get("cfgDrums") or 1.0)
  use_mlxfn = bool(payload.get("useMlxfn", True))
  frames = max(1, min(25 * 64, round(duration * 25)))
  midi_notes = payload.get("midiNotes")
  if not isinstance(midi_notes, list):
    midi_notes = []
  drums = payload.get("drums")
  if drums is not None:
    if not isinstance(drums, list) or len(drums) != 1:
      raise ValueError("drums must be a one-item list: -1=masked, 0=off, 1=on.")
    drums = [int(drums[0])]

  started = time.time()
  model = _load_model(model_name, use_mlxfn)
  conditioned_prompt = _with_bpm_hint(prompt, bpm)
  embedding = model.embed_style(conditioned_prompt, use_mapper=True)
  note_frames = _midi_note_frames(midi_notes, frames)
  if note_frames:
    wav = _generate_with_midi_guidance(
        model,
        embedding,
        note_frames,
        cfg_musiccoca=cfg_musiccoca,
        cfg_notes=cfg_notes,
        cfg_drums=cfg_drums,
        drums=drums,
        temperature=temperature,
        top_k=top_k,
    )
  else:
    wav, _ = model.generate(
        style=embedding,
        frames=frames,
        temperature=temperature,
        top_k=top_k,
        cfg_musiccoca=cfg_musiccoca,
        cfg_notes=cfg_notes,
        cfg_drums=cfg_drums,
        drums=drums,
    )

  buffer = io.BytesIO()
  sf.write(buffer, wav.samples, wav.sample_rate, format="WAV")
  elapsed = time.time() - started
  meta = {
      "prompt": prompt,
      "conditionedPrompt": conditioned_prompt,
      "model": model_name,
      "duration": wav.seconds,
      "sampleRate": wav.sample_rate,
      "elapsed": elapsed,
      "frames": frames,
      "midiGuided": bool(note_frames),
      "drumless": drums == [0],
      "bpm": bpm,
  }
  return buffer.getvalue(), meta


class LooperHandler(BaseHTTPRequestHandler):
  server_version = "MagentaLooper/0.1"

  def log_message(self, fmt: str, *args: Any) -> None:
    LOGGER.info("%s - %s", self.address_string(), fmt % args)

  def _send_json(self, status: HTTPStatus, data: dict[str, Any]) -> None:
    body = json.dumps(data).encode("utf-8")
    self.send_response(status)
    self.send_header("Content-Type", "application/json")
    self.send_header("Content-Length", str(len(body)))
    self.send_header("Access-Control-Allow-Origin", "*")
    self.end_headers()
    self.wfile.write(body)

  def _send_wav(self, data: bytes, meta: dict[str, Any]) -> None:
    self.send_response(HTTPStatus.OK)
    self.send_header("Content-Type", "audio/wav")
    self.send_header("Content-Length", str(len(data)))
    self.send_header("X-Magenta-Meta", json.dumps(meta))
    self.send_header("Access-Control-Allow-Origin", "*")
    self.end_headers()
    self.wfile.write(data)

  def do_OPTIONS(self) -> None:
    self.send_response(HTTPStatus.NO_CONTENT)
    self.send_header("Access-Control-Allow-Origin", "*")
    self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
    self.send_header("Access-Control-Allow-Headers", "Content-Type")
    self.end_headers()

  def do_GET(self) -> None:
    path = urlparse(self.path).path
    if path == "/api/health":
      model_root = paths.models_dir()
      local_models = []
      if model_root.exists():
        local_models = sorted(p.name for p in model_root.iterdir() if p.is_dir())
      self._send_json(
          HTTPStatus.OK,
          {
              "ok": True,
              "modelsDir": str(model_root),
              "resourcesDir": str(paths.resources_dir()),
              "localModels": local_models,
          },
      )
      return
    self._send_json(HTTPStatus.NOT_FOUND, {"ok": False, "error": "Not found"})

  def do_POST(self) -> None:
    path = urlparse(self.path).path
    if path != "/api/generate":
      self._send_json(HTTPStatus.NOT_FOUND, {"ok": False, "error": "Not found"})
      return

    try:
      length = int(self.headers.get("Content-Length", "0"))
      raw = self.rfile.read(length)
      payload = json.loads(raw.decode("utf-8")) if raw else {}
      wav_bytes, meta = _generate_wav(payload)
      self._send_wav(wav_bytes, meta)
    except FileNotFoundError as exc:
      self._send_json(
          HTTPStatus.BAD_REQUEST,
          {
              "ok": False,
              "error": str(exc),
              "hint": "Run `mrt models init` and `mrt models download --model=mrt2_small`, or select an installed model.",
          },
      )
    except Exception as exc:  # pylint: disable=broad-exception-caught
      LOGGER.exception("Generation failed")
      self._send_json(
          HTTPStatus.INTERNAL_SERVER_ERROR,
          {"ok": False, "error": f"{type(exc).__name__}: {exc}"},
      )


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--host", default="127.0.0.1")
  parser.add_argument("--port", default=8765, type=int)
  args = parser.parse_args()

  logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
  # MLX GPU streams are thread-local. Keep generation on the serving thread in
  # this prototype instead of dispatching each request to a new worker thread.
  server = HTTPServer((args.host, args.port), LooperHandler)
  LOGGER.info("Magenta looper server listening on http://%s:%d", args.host, args.port)
  server.serve_forever()


if __name__ == "__main__":
  main()
