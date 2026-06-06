#!/usr/bin/env python3
"""Development server for Stable Audio 3 audio-to-audio loop transforms."""

from __future__ import annotations

import argparse
import base64
import json
import logging
import subprocess
import tempfile
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


LOGGER = logging.getLogger("stable_audio_looper")
ROOT = Path(__file__).resolve().parents[3]
SA3_DIR = ROOT / "third_party" / "stable-audio-3" / "optimized" / "mlx"
SA3_PYTHON = SA3_DIR / ".venv" / "bin" / "python"
SA3_SCRIPT = SA3_DIR / "scripts" / "sa3_mlx.py"
DEFAULT_TIMEOUT_SECONDS = 12 * 60


def _setup_hint() -> str:
  return (
      "Set up Stable Audio once with "
      "`cd third_party/stable-audio-3/optimized/mlx && ./install.sh --download sm-music`."
  )


def _read_json(handler: BaseHTTPRequestHandler) -> dict[str, Any]:
  length = int(handler.headers.get("Content-Length", "0"))
  raw = handler.rfile.read(length)
  return json.loads(raw.decode("utf-8")) if raw else {}


def _clamp(value: float, minimum: float, maximum: float) -> float:
  return max(minimum, min(maximum, value))


def _generate_audio_to_audio(payload: dict[str, Any]) -> tuple[bytes, dict[str, Any]]:
  if not SA3_DIR.exists() or not SA3_SCRIPT.exists():
    raise FileNotFoundError(f"Stable Audio submodule not found at {SA3_DIR}")
  if not SA3_PYTHON.exists():
    raise RuntimeError(_setup_hint())

  audio_base64 = str(payload.get("audioBase64") or "")
  if not audio_base64:
    raise ValueError("Missing audioBase64 WAV input.")

  prompt = str(payload.get("prompt") or "cassette tape music loop")
  duration = _clamp(float(payload.get("duration") or 8.0), 1.0, 65.0)
  init_noise_level = _clamp(float(payload.get("initNoiseLevel") or 0.5), 0.0, 1.0)
  cfg = _clamp(float(payload.get("cfg") or 1.0), 0.0, 10.0)
  steps = int(_clamp(float(payload.get("steps") or 8), 1.0, 32.0))

  with tempfile.TemporaryDirectory(prefix="sa3-loop-") as tmpdir:
    temp_root = Path(tmpdir)
    input_wav = temp_root / "input.wav"
    output_wav = temp_root / "output.wav"
    input_wav.write_bytes(base64.b64decode(audio_base64))

    command = [
        str(SA3_PYTHON),
        str(SA3_SCRIPT),
        "--prompt",
        prompt,
        "--dit",
        "sm-music",
        "--decoder",
        "same-s",
        "--init-audio",
        str(input_wav),
        "--init-noise-level",
        f"{init_noise_level:.4f}",
        "--seconds",
        f"{duration:.3f}",
        "--steps",
        str(steps),
        "--cfg",
        f"{cfg:.3f}",
        "--out",
        str(output_wav),
    ]

    started = time.time()
    result = subprocess.run(
        command,
        cwd=SA3_DIR,
        capture_output=True,
        text=True,
        timeout=DEFAULT_TIMEOUT_SECONDS,
        check=False,
    )
    elapsed = time.time() - started
    if result.returncode != 0:
      raise subprocess.CalledProcessError(
          result.returncode,
          command,
          output=result.stdout[-4000:],
          stderr=result.stderr[-4000:],
      )
    if not output_wav.exists():
      raise RuntimeError("Stable Audio finished without writing an output WAV.")

    meta = {
        "prompt": prompt,
        "duration": duration,
        "initNoiseLevel": init_noise_level,
        "cfg": cfg,
        "steps": steps,
        "elapsed": elapsed,
        "model": "sm-music",
        "decoder": "same-s",
    }
    return output_wav.read_bytes(), meta


class StableAudioHandler(BaseHTTPRequestHandler):
  server_version = "StableAudioLooper/0.1"

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
    self.send_header("X-SA3-Meta", json.dumps(meta))
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
    if path == "/sa3/health":
      self._send_json(
          HTTPStatus.OK,
          {
              "ok": SA3_PYTHON.exists(),
              "stableAudioDir": str(SA3_DIR),
              "venvReady": SA3_PYTHON.exists(),
              "scriptReady": SA3_SCRIPT.exists(),
              "hint": None if SA3_PYTHON.exists() else _setup_hint(),
          },
      )
      return
    self._send_json(HTTPStatus.NOT_FOUND, {"ok": False, "error": "Not found"})

  def do_POST(self) -> None:
    path = urlparse(self.path).path
    if path != "/sa3/audio-to-audio":
      self._send_json(HTTPStatus.NOT_FOUND, {"ok": False, "error": "Not found"})
      return

    try:
      wav_bytes, meta = _generate_audio_to_audio(_read_json(self))
      self._send_wav(wav_bytes, meta)
    except RuntimeError as exc:
      self._send_json(HTTPStatus.SERVICE_UNAVAILABLE, {"ok": False, "error": str(exc), "hint": _setup_hint()})
    except FileNotFoundError as exc:
      self._send_json(HTTPStatus.NOT_FOUND, {"ok": False, "error": str(exc), "hint": _setup_hint()})
    except subprocess.CalledProcessError as exc:
      LOGGER.error("Stable Audio failed: %s", exc.stderr or exc.output)
      self._send_json(
          HTTPStatus.INTERNAL_SERVER_ERROR,
          {
              "ok": False,
              "error": f"Stable Audio exited with code {exc.returncode}.",
              "stdout": exc.output,
              "stderr": exc.stderr,
          },
      )
    except Exception as exc:  # pylint: disable=broad-exception-caught
      LOGGER.exception("Stable Audio transform failed")
      self._send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"ok": False, "error": f"{type(exc).__name__}: {exc}"})


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--host", default="127.0.0.1")
  parser.add_argument("--port", default=8766, type=int)
  args = parser.parse_args()

  logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
  server = HTTPServer((args.host, args.port), StableAudioHandler)
  LOGGER.info("Stable Audio looper server listening on http://%s:%d", args.host, args.port)
  server.serve_forever()


if __name__ == "__main__":
  main()
