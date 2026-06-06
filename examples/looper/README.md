# Magenta Loop Lab

Experimental browser looper for sketching realtime Magenta workflows.

## Architecture

- React/Vite frontend owns transport, bar-based loop length, count-in, audio
  capture, MIDI capture, software-synth rendering, track volume, and loop
  playback through WebAudio.
- Python backend exposes a narrow HTTP API for Magenta prompt-to-audio
  generation, including MIDI note guidance.
- Stable Audio 3 runs as a second narrow HTTP API for audio-to-audio
  transforms. Audio tracks send repeated source-loop WAV input, then crop the
  generated output back to the loop length in the browser.
- The boundary is intentionally small so the backend can later become the C++
  realtime engine, an audio-to-audio model, or a more DAW-like service without
  rewriting the loop UI.

## Run

From a fresh clone, make sure submodules are present:

```bash
git submodule update --init --recursive
```

Install the Magenta package with MLX support from the repo root:

```bash
uv pip install -e ".[mlx]"
```

Magenta generation expects local model resources under the normal
`~/Documents/Magenta/magenta-rt-v2` directory:

```bash
mrt models init
mrt models download --model=mrt2_small
```

Start the Magenta generation server from the repo root:

```bash
python examples/looper/server/looper_server.py --port 8765
```

Stable Audio audio-to-audio uses the optimized MLX submodule. One-time setup
from the repo root:

```bash
cd third_party/stable-audio-3/optimized/mlx
./install.sh --download sm-music
cd ../../../..
```

Then start the Stable Audio server from the repo root in another terminal:

```bash
python examples/looper/server/stable_audio_server.py --port 8766
```

From `examples/`:

```bash
npm --workspace looper run dev -- --host 127.0.0.1
```

Open <http://127.0.0.1:62430/>.

## Controls

- Each track has its own loop length in 1, 2, 4, 8, or 16 bar increments.
- Audio tracks record source audio locally, then use Stable Audio 3 for
  audio-to-audio transforms.
- MIDI tracks render a local synth source loop, then use Magenta with note
  guidance for transforms.
- `Prompt strength` controls Magenta's MusicCoCa CFG strength.
- `Init noise` controls Stable Audio's `init_noise_level` for audio-to-audio
  transforms. The default is `0.50`.
- `No drums` passes Magenta's drum conditioning token `0`, which the model API
  documents as "no drum"; leaving it off masks the drum token so the model can
  decide.
- `Double gen` and `Smooth fade` are experimental loop-polishing options. Stable
  Audio always receives a longer repeated input window for short loops before
  the final loop is cropped back into place.

## Notes

- Audio recording uses the browser microphone permission and `MediaRecorder`.
- MIDI capture uses Web MIDI, which is best supported in Chrome.
- The Stable Audio installer downloads weights from Hugging Face. The `sm-music`
  bundle is the intended starting point for this prototype.
