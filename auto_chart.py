#!/usr/bin/env python3
r"""
auto_chart.py — Offline analyzer for rhythm-game beatmaps

What it does
------------
Given an input audio file, this script:
  1) (Optional) Separates stems with Demucs (vocals/drums/bass/other).
  2) Computes onsets for KICK (low percussive), SNARE (mid percussive),
     HAT (high percussive), and BASS (low harmonic).
  3) Detects the global beat grid (tempo & beats).
  4) Quantizes detected events to the beat grid (with subdivisions).
  5) Writes a beatmap JSON usable directly in Godot (see example in README below).

Install (minimal, no stems):
----------------------------
    python -m venv .venv && source .venv/bin/activate  # (Windows: .venv\\Scripts\\activate)
    pip install -r requirements.txt

Install (with Demucs stems, optional):
--------------------------------------
Demucs requires PyTorch and can be heavy. If you install it, the script will
use it automatically when you pass --demucs. Otherwise it falls back to HPSS.

    pip install demucs torch

Usage
-----
    python auto_chart.py "path/to/song.mp3" \
        --out beatmap.json \
        --subdiv 4 \
        --downbeat-every 4 \
        --demucs

Notes
-----
* Times are in seconds; they will align in any engine that schedules by DSP time.
* If stems are used, we estimate and embed a small analysis_offset_sec to align
  stem-derived events to the original mix.
"""
import argparse
import hashlib
import json
import os
import sys
import tempfile
import subprocess
from typing import Dict, List, Optional, Tuple

import numpy as np

# Core audio libs
import librosa
import soundfile as sf
from scipy.signal import butter, lfilter, find_peaks

# -------------------------------
# Helpers
# -------------------------------

def sha1_of_file(path: str) -> str:
    h = hashlib.sha1()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()

def lowpass_butter(sig: np.ndarray, sr: int, cutoff: float, order: int = 4) -> np.ndarray:
    b, a = butter(order, cutoff / (sr * 0.5), btype='low')
    return lfilter(b, a, sig)

def band_energy_from_stft(S_mag: np.ndarray, sr: int, n_fft: int, f_lo: float, f_hi: float) -> np.ndarray:
    """Sum magnitude within [f_lo, f_hi] over frequency bins."""
    freqs = np.linspace(0, sr/2, num=1 + n_fft//2)
    lo_bin = np.searchsorted(freqs, max(f_lo, 0.0))
    hi_bin = np.searchsorted(freqs, min(f_hi, sr/2.0))
    lo_bin = max(0, min(lo_bin, S_mag.shape[0]-1))
    hi_bin = max(lo_bin+1, min(hi_bin, S_mag.shape[0]))
    band = S_mag[lo_bin:hi_bin, :]
    if band.size == 0:
        return np.zeros(S_mag.shape[1], dtype=np.float32)
    return band.sum(axis=0)

def spectral_flux(x: np.ndarray) -> np.ndarray:
    """Half-wave rectified first difference."""
    d = np.diff(x, prepend=x[:1])
    d[d < 0] = 0.0
    return d

def normalize_power(v: np.ndarray) -> np.ndarray:
    """Scale to 0..1 using 95th percentile to reduce outlier influence."""
    if v.size == 0:
        return v
    p95 = np.percentile(v, 95) + 1e-9
    return np.clip(v / p95, 0.0, 1.0)

def pick_event_frames(novelty: np.ndarray, sr: int, hop: int, sensitivity: float, min_interval_ms: int) -> np.ndarray:
    """Adaptive threshold by EMA; pick peaks with min distance."""
    # EMA baseline
    alpha = 0.1
    baseline = np.zeros_like(novelty)
    for i in range(1, novelty.size):
        baseline[i] = (1-alpha) * baseline[i-1] + alpha * novelty[i]
    thresh = baseline * sensitivity
    score = novelty - thresh
    score[score < 0] = 0.0
    # Peak picking
    distance = max(1, int((min_interval_ms/1000.0) * sr / hop))
    peaks, _ = find_peaks(score, distance=distance)
    return peaks

def quantize_times_to_grid(event_times: np.ndarray, grid_times: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return (quantized_times, grid_indices). Snap each event to nearest grid time."""
    if len(grid_times) == 0 or len(event_times) == 0:
        return np.array([]), np.array([], dtype=int)
    idx = np.searchsorted(grid_times, event_times)
    idx0 = np.clip(idx-1, 0, len(grid_times)-1)
    idx1 = np.clip(idx,   0, len(grid_times)-1)
    left = grid_times[idx0]
    right = grid_times[idx1]
    choose_right = (np.abs(right - event_times) < np.abs(event_times - left)).astype(bool)
    best_idx = np.where(choose_right, idx1, idx0)
    q_times = grid_times[best_idx]
    return q_times, best_idx

def build_grid_from_beats(beat_times: np.ndarray, subdivision: int) -> Tuple[np.ndarray, np.ndarray]:
    """Return (grid_times, grid_beat_index). Subdivide each beat interval into 'subdivision' steps."""
    if len(beat_times) < 2:
        return beat_times.copy(), np.arange(len(beat_times), dtype=int)
    grid_times = []
    grid_beat_idx = []
    for i in range(len(beat_times)-1):
        t0, t1 = beat_times[i], beat_times[i+1]
        for s in range(subdivision):
            grid_times.append(t0 + (t1 - t0) * (s / subdivision))
            grid_beat_idx.append(i)
    # include last beat time exactly
    grid_times.append(beat_times[-1])
    grid_beat_idx.append(len(beat_times)-1)
    return np.asarray(grid_times), np.asarray(grid_beat_idx, dtype=int)

def try_demucs_separate(audio_path_for_demucs: str, out_dir: str, model: str = "htdemucs") -> Optional[Dict[str, str]]:
    """
    Run Demucs via CLI on a WAV we provide (avoids MP3 decoding issues).
    Returns dict of stem paths or None on failure.
    """
    try:
        import time
        
        cmd = ["demucs", "-n", model, "-o", out_dir, audio_path_for_demucs]
        print(f"[demucs] Running: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
        
        # Wait a moment for files to be fully written
        time.sleep(30)
        
        # Find where Demucs actually put the files
        stem_map = {}
        for root, dirs, files in os.walk(out_dir):
            for file in files:
                if file.endswith('.wav'):
                    stem_name = file[:-4]  # Remove .wav
                    if stem_name in ["vocals", "drums", "bass", "other"]:
                        full_path = os.path.join(root, file)
                        if os.path.getsize(full_path) > 0:  # Ensure file has content
                            stem_map[stem_name] = full_path
        
        if len(stem_map) >= 2:
            print(f"[demucs] Found stems: {list(stem_map.keys())}")
            for name, path in stem_map.items():
                print(f"[demucs]   {name}: {path}")
            return stem_map
        
        print("[demucs] Stems not found")
        return None
        
    except Exception as e:
        print(f"[demucs] Failed: {e}")
        return None

def estimate_offset_sec(ref_curve: np.ndarray, tgt_curve: np.ndarray, sr: int, hop: int, max_lag_ms: int = 200) -> float:
    """Cross-correlate to estimate lag (tgt relative to ref). Positive => tgt delayed."""
    # Normalize
    ref = (ref_curve - ref_curve.mean()) / (ref_curve.std() + 1e-9)
    tgt = (tgt_curve - tgt_curve.mean()) / (tgt_curve.std() + 1e-9)
    max_lag = int((max_lag_ms/1000.0) * sr / hop)
    # Correlate with padding managed by 'full'
    corr = np.correlate(tgt, ref, mode='full')
    center = len(corr)//2
    l0, l1 = center - max_lag, center + max_lag + 1
    window = corr[l0:l1]
    best = np.argmax(window) + (l0 - center)
    return best * hop / float(sr)

# -------------------------------
# Core analysis
# -------------------------------

def analyze_track(
    audio_path: str,
    out_json: str,
    subdivision: int = 4,
    downbeat_every: int = 4,
    use_demucs: bool = False,
    demucs_model: str = "htdemucs",
    sensitivity_low: float = 1.4,
    sensitivity_mid: float = 1.4,
    sensitivity_high: float = 1.6,
    sensitivity_bass: float = 1.3,
    min_interval_ms_low: int = 110,
    min_interval_ms_mid: int = 110,
    min_interval_ms_high: int = 80,
    min_interval_ms_bass: int = 150,
    n_fft: int = 2048,
    hop: int = 512,
) -> None:
    print(f"[load] {audio_path}")
    y, sr = librosa.load(audio_path, mono=True)
    audio_sha1 = sha1_of_file(audio_path)

    # 1) Optional stems (run Demucs on a temp WAV built from what we already decoded)
    stems = None
    analysis_offset_sec = 0.0
    # 1) Optional stems (run Demucs on a regular directory)
    y_drums = None
    y_bass = None
    analysis_offset_sec = 0.0
    
    if use_demucs:
        # Just use a regular directory - container will clean up
        tmpdir = "/tmp/demucs_work"
        os.makedirs(tmpdir, exist_ok=True)
        
        tmp_wav = os.path.join(tmpdir, "demucs_input.wav")
        sf.write(tmp_wav, y, sr)
        stems = try_demucs_separate(tmp_wav, tmpdir, demucs_model)
        
        if stems:
            if "drums" in stems:
                print(f"[demucs] Loading drums from {stems['drums']}")
                y_drums, _ = librosa.load(stems["drums"], mono=True, sr=sr)
            if "bass" in stems:
                print(f"[demucs] Loading bass from {stems['bass']}")
                y_bass, _ = librosa.load(stems["bass"], mono=True, sr=sr)
            print("[demucs] Stems loaded successfully")
        else:
            print("[stems] Falling back to HPSS (Demucs unavailable).")
    
    # 2) Build percussive/harmonic versions (fallback or for extra cues)
    if y_drums is None:
        y_harm, y_perc = librosa.effects.hpss(y)
        y_drums = y_perc
    
    if y_bass is None:
        if 'y_harm' not in locals():
            y_harm, _ = librosa.effects.hpss(y)
        y_bass = lowpass_butter(y_harm, sr, cutoff=200.0, order=4)

    # 3) STFT for percussive bands
    S_drums = np.abs(librosa.stft(y_drums, n_fft=n_fft, hop_length=hop))
    # Frequency bands (Hz)
    LOW = (30.0, 140.0)
    MID = (150.0, 3000.0)
    HIGH = (5000.0, 14000.0)

    e_low  = band_energy_from_stft(S_drums, sr, n_fft, *LOW)
    e_mid  = band_energy_from_stft(S_drums, sr, n_fft, *MID)
    e_high = band_energy_from_stft(S_drums, sr, n_fft, *HIGH)

    # Bass novelty from harmonic low band
    oenv_bass = librosa.onset.onset_strength(y=y_bass, sr=sr, hop_length=hop)

    # Drums novelty via spectral flux per band
    flux_low  = spectral_flux(e_low)
    flux_mid  = spectral_flux(e_mid)
    flux_high = spectral_flux(e_high)

    # 4) Beat tracking on the full mix for a robust grid
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr, hop_length=hop, units="frames")
    tempo = float(np.atleast_1d(tempo)[0])
    beat_times = librosa.frames_to_time(beat_frames, sr=sr, hop_length=hop)
    if len(beat_times) < 4:
        # Fallback grid: fixed tempo from tempo estimate
        print("[beat] Sparse beats; synthesizing grid from tempo.")
        if tempo <= 0:
            tempo = 120.0
        beat_len = 60.0 / tempo
        beat_times = np.arange(0, len(y)/sr, beat_len)

    grid_times, grid_beat_idx = build_grid_from_beats(beat_times, subdivision=subdivision)

    # 5) Peak picking per component
    peaks_low  = pick_event_frames(flux_low,  sr, hop, sensitivity_low,  min_interval_ms_low)
    peaks_mid  = pick_event_frames(flux_mid,  sr, hop, sensitivity_mid,  min_interval_ms_mid)
    peaks_high = pick_event_frames(flux_high, sr, hop, sensitivity_high, min_interval_ms_high)
    peaks_bass = pick_event_frames(oenv_bass, sr, hop, sensitivity_bass, min_interval_ms_bass)

    t_low  = librosa.frames_to_time(peaks_low,  sr=sr, hop_length=hop)
    t_mid  = librosa.frames_to_time(peaks_mid,  sr=sr, hop_length=hop)
    t_high = librosa.frames_to_time(peaks_high, sr=sr, hop_length=hop)
    t_bass = librosa.frames_to_time(peaks_bass, sr=sr, hop_length=hop)

    # Strengths (normalized)
    s_low  = normalize_power(flux_low[peaks_low])   if peaks_low.size  else np.array([])
    s_mid  = normalize_power(flux_mid[peaks_mid])   if peaks_mid.size  else np.array([])
    s_high = normalize_power(flux_high[peaks_high]) if peaks_high.size else np.array([])
    s_bass = normalize_power(oenv_bass[peaks_bass]) if peaks_bass.size else np.array([])

    # 6) Optional offset estimate if stems used (align drums flux to original onset envelope)
    if stems:
        oenv_mix = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)
        oenv_drums = librosa.onset.onset_strength(S=librosa.stft(y_drums, n_fft=n_fft, hop_length=hop), sr=sr, hop_length=hop)
        analysis_offset_sec = estimate_offset_sec(oenv_mix, oenv_drums, sr=sr, hop=hop)
        print(f"[align] analysis_offset_sec ≈ {analysis_offset_sec:+.4f} s")

    # 7) Quantize to grid; keep strongest per type per grid slot
    events_by_grid: Dict[Tuple[int, str], Tuple[float, float]] = {}

    def add_events(t_arr: np.ndarray, s_arr: np.ndarray, label: str):
        if t_arr.size == 0:
            return
        q_times, q_idx = quantize_times_to_grid(t_arr, grid_times)
        for qt, gi, strength in zip(q_times, q_idx, s_arr):
            key = (int(gi), label)
            prev = events_by_grid.get(key)
            if (prev is None) or (strength > prev[1]):
                events_by_grid[key] = (float(qt), float(strength))

    add_events(t_low,  s_low,  "kick")
    add_events(t_mid,  s_mid,  "snare")
    add_events(t_high, s_high, "hat")
    add_events(t_bass, s_bass, "bass")

    # 8) Flatten & sort
    events = [{"t": v[0], "type": k[1], "power": round(v[1], 3)} for k, v in events_by_grid.items()]
    events.sort(key=lambda e: e["t"])

    # 9) Build JSON
    out = {
        "audio_sha1": audio_sha1,
        "analysis_offset_sec": round(analysis_offset_sec, 6),
        "tempo_map": [
            {
                "t0": 0.0,
                "bpm": float(tempo),
                "beat0": 0,
                "downbeat_every": int(downbeat_every)
            }
        ],
        "grid_subdivision": int(subdivision),
        "events": events
    }

    # 10) Write
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"[done] Wrote {out_json} with {len(events)} events.")


def main():
    ap = argparse.ArgumentParser(description="Offline analyzer → Godot beatmap JSON")
    ap.add_argument("audio", help="Path to input audio file (wav/mp3/flac/ogg)")
    ap.add_argument("--out", default="beatmap.json", help="Output JSON path")
    ap.add_argument("--subdiv", type=int, default=4, help="Grid subdivision per beat (1=quarters,2=eighths,4=sixteenths)")
    ap.add_argument("--downbeat-every", type=int, default=4, help="Beats per bar (for metadata only)")
    ap.add_argument("--demucs", action="store_true", help="Use Demucs stems if installed (optional)")
    ap.add_argument("--demucs-model", default="htdemucs", help="Demucs model name (e.g., htdemucs, htdemucs_ft, htdemucs_6s)")

    ap.add_argument("--sens-low",  type=float, default=1.4, help="Sensitivity for KICK (lower=more)")
    ap.add_argument("--sens-mid",  type=float, default=1.4, help="Sensitivity for SNARE")
    ap.add_argument("--sens-high", type=float, default=1.6, help="Sensitivity for HAT")
    ap.add_argument("--sens-bass", type=float, default=1.3, help="Sensitivity for BASS")
    ap.add_argument("--minint-low",  type=int, default=110, help="Min interval ms for KICK")
    ap.add_argument("--minint-mid",  type=int, default=110, help="Min interval ms for SNARE")
    ap.add_argument("--minint-high", type=int, default=80,  help="Min interval ms for HAT")
    ap.add_argument("--minint-bass", type=int, default=150, help="Min interval ms for BASS")

    args = ap.parse_args()

    analyze_track(
        audio_path=args.audio,
        out_json=args.out,
        subdivision=args.subdiv,
        downbeat_every=args.downbeat_every,
        use_demucs=args.demucs,
        demucs_model=args.demucs_model,
        sensitivity_low=args.sens_low,
        sensitivity_mid=args.sens_mid,
        sensitivity_high=args.sens_high,
        sensitivity_bass=args.sens_bass,
        min_interval_ms_low=args.minint_low,
        min_interval_ms_mid=args.minint_mid,
        min_interval_ms_high=args.minint_high,
        min_interval_ms_bass=args.minint_bass,
    )

if __name__ == "__main__":
    main()
