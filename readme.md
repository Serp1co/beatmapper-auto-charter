# auto_chart — Beatmap generator & Dockerized CLI

`auto_chart.py` is an offline analyzer that turns a song (mp3/wav/flac/ogg) into a **rhythm‑game beatmap JSON**. It detects kick, snare, hi‑hat, and bass hits, estimates tempo & beats, snaps everything to a musical grid, and writes a compact JSON you can load directly in engines like **Godot**.

This README explains:

- What the script does and how it works
- What the Docker setup/compose service does
- What the output JSON looks like
- How to tune detection quality
- Quick start & usage (kept exactly as before)

---

## What the script does (high level)

1. **(Optional) Source separation with Demucs** — if enabled, it separates the track into stems (drums/bass/etc.). This can improve percussive onset detection.
2. **Feature extraction** — computes spectral bands for percussive energy (low/mid/high) and a bass onset envelope.
3. **Beat tracking & tempo** — finds the global beat grid (tempo and beat times).
4. **Grid quantization** — subdivides each beat (e.g., 1, 2, 4 ⇒ quarter, eighth, sixteenth notes) and snaps detected events to the closest grid tick.
5. **Beatmap JSON export** — writes `beatmap.json` with tempo metadata, grid subdivision, and per‑tick events with strengths.

All event **times are in seconds**, so they align in any audio engine that schedules by DSP time. If stems are used, a small `analysis_offset_sec` is embedded to keep stem‑derived cues aligned with the original mix.

---

## Output JSON (structure & example)

**Fields**

- `audio_sha1` — checksum of the analyzed audio file (integrity/identity).
- `analysis_offset_sec` — alignment offset applied when stems are used.
- `tempo_map` — currently a single segment `{ t0, bpm, beat0, downbeat_every }` but designed to allow future tempo changes.
- `grid_subdivision` — how many slices per beat (1, 2, 4, …).
- `events` — sorted list of `{ t, type, power }` with:
  - `type` ∈ `{ "kick", "snare", "hat", "bass" }`
  - `power` ∈ `[0..1]` (normalized strength)

**Minimal example**

```json
{
  "audio_sha1": "8e40…",
  "analysis_offset_sec": 0.0,
  "tempo_map": [
    { "t0": 0.0, "bpm": 120.0, "beat0": 0, "downbeat_every": 4 }
  ],
  "grid_subdivision": 4,
  "events": [
    { "t": 1.234, "type": "kick", "power": 0.83 },
    { "t": 1.604, "type": "snare", "power": 0.74 },
    { "t": 1.734, "type": "hat",   "power": 0.62 },
    { "t": 2.000, "type": "bass",  "power": 0.58 }
  ]
}
```

---

## Docker & docker‑compose — what they do

The Docker setup provides a **reproducible runtime** with the required Python audio stack already installed. You run the analyzer inside the container and pass audio files via mounted volumes.

- The compose service (e.g. `auto-chart`) is configured so that **host folders** map to container paths like `/input` and `/output`.
- You invoke the service with `docker-compose run … /input/<file> --out /output/<file>.json` so results are written back to your host.
- Demucs is optional. If included in the image and you pass `--demucs`, the container will separate stems before analysis.

> If you adjust mount points in your `docker-compose.yml`, update the `/input` and `/output` paths in the command accordingly.

---

## Tuning detection quality

You can fine‑tune both **sensitivity** and **minimum intervals** per instrument to suit different genres:

- `--sens-low` / `--minint-low` → KICK (low percussive)
- `--sens-mid` / `--minint-mid` → SNARE (mid percussive)
- `--sens-high` / `--minint-high` → HAT (high percussive)
- `--sens-bass` / `--minint-bass` → BASS (low harmonic)

**Tips**

- Lower sensitivity (e.g., 1.2) ⇒ **more** detections; higher (e.g., 1.8) ⇒ **fewer**.
- Shorter min interval ⇒ allows denser rolls/hi‑hat patterns; longer ⇒ suppresses flams/doubles.
- For fast EDM/metal, try `--subdiv 4` or `8`. For laid‑back hip‑hop or pop, `--subdiv 2` often suffices.
- If downbeats feel off by a bar, set `--downbeat-every` to your meter (e.g., 3 for 3/4, 4 for 4/4).

---

## Demucs integration (optional)

- Install `demucs` (and `torch`) locally or include them in your image.
- Enable with `--demucs` (default model `htdemucs`, change via `--demucs-model`).
- Stems help isolate drums/bass, giving clearer onsets for dense mixes.
- Using Demucs is heavier in CPU/GPU and RAM; consider container resources accordingly.

---

## Visualizer.html

There’s an HTML canvas tool to **preview** your beatmap: open `Visualizer.html` in a browser, then load both the **beatmap JSON** and the **original track** to see/hear alignment.

---

## Quick start (keep usage)

> Create a virtual environment and install:

```bash
python -m venv .venv

# mac/linux

source .venv/bin/activate
# windows

.venv\Scripts\activate

pip install -r requirements.txt
```

> Run script

```bash
python auto_chart.py "path/to/song.wav" --out beatmap.json --subdiv 4 --downbeat-every 4 --demucs
```

> Run as compose

```bash
docker-compose run --rm auto-chart /input/1_believer.mp3 --out /output/believer_beatmap.json --subdiv 4 --downbeat-every 4 --demucs
```

> Visualizer.html  
> An html canvas to visualize and listen to the beatmap, just open in a browser, load the beatmap and original track.

---

## Full CLI reference

```
auto_chart.py AUDIO [--out PATH] [--subdiv N] [--downbeat-every N]
                   [--demucs] [--demucs-model NAME]
                   [--sens-low F]  [--sens-mid F]  [--sens-high F]  [--sens-bass F]
                   [--minint-low MS] [--minint-mid MS] [--minint-high MS] [--minint-bass MS]
```

- `AUDIO` — path to input file (`wav/mp3/flac/ogg`).
- `--out` — output JSON path (default `beatmap.json`).
- `--subdiv` — grid subdivision per beat (1=quarters, 2=eighths, 4=sixteenths; default `4`).
- `--downbeat-every` — beats per bar (metadata only; default `4`).
- `--demucs` — enable Demucs stem separation (if installed).
- `--demucs-model` — Demucs model name (e.g., `htdemucs`, `htdemucs_ft`, `htdemucs_6s`).
- `--sens-*` — detection sensitivity per component (lower ⇒ more events).
- `--minint-*` — minimum interval between events (milliseconds) per component.

---

## Example workflows

- **Straightforward analysis**
  ```bash
  python auto_chart.py "song.mp3" --out beatmap.json --subdiv 4 --downbeat-every 4
  ```
- **With Demucs for dense mixes**
  ```bash
  python auto_chart.py "song.mp3" --out beatmap.json --subdiv 4 --downbeat-every 4 --demucs --demucs-model htdemucs
  ```
- **Dockerized run to keep host clean**
  ```bash
  docker-compose run --rm auto-chart /input/song.mp3 --out /output/beatmap.json --subdiv 4 --downbeat-every 4
  ```

---

## Troubleshooting

- **Librosa/SoundFile errors (e.g., libsndfile missing)** — ensure system audio libs are installed on host, or prefer the Docker image which bundles dependencies.
- **Very sparse or very dense events** — adjust `--sens-*` and `--minint-*`; consider `--demucs` for clarity.
- **Tempo/beats feel unstable** — try a lower subdivision or analyze a higher‑quality source (WAV/FLAC). Extremely free‑time sections may not quantize well.
- **Container can’t see files** — confirm compose volume mounts and use the container paths (e.g., `/input`, `/output`).

---

## License & credits

- Beat/tempo tracking and onset features are built on top of the excellent `librosa` ecosystem.
- Optional source separation via `demucs` (Facebook Research).

Happy charting! 🎵

