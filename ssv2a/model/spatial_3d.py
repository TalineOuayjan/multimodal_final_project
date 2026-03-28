"""
spatial_3d.py — 3D spatial geometry + 3D spatial audio effects.

Given bounding boxes + per-object depth values from monocular depth estimation,
compute the true 3D position of each object relative to the camera using a
pinhole camera model.

Then apply physically-motivated audio effects:
  - Distance attenuation      (inverse-distance law, configurable exponent)
  - Stereo / binaural panning (constant-power pan from azimuth angle)
  - Propagation delay          (speed of sound = 343 m/s)
  - Reverb / environment cues  (reverb amount scales with distance)
  - Elevation cue              (subtle spectral tilt from elevation angle)

Mathematical formulation
------------------------
Pinhole camera (assuming default intrinsics if not provided):
    f_x = f_y = max(W, H)       (approx 60° horizontal FoV)
    c_x = W / 2,  c_y = H / 2   (principal point at image centre)

3D backprojection (pixel → camera-relative 3D):
    X = (u - c_x) * Z / f_x
    Y = (v - c_y) * Z / f_y
    Z = depth(u, v)              (from monocular depth estimation)

Spherical coordinates:
    distance  d = sqrt(X² + Y² + Z²)
    azimuth   θ = atan2(X, Z)           ∈ [-π, π]   (+ = right)
    elevation φ = atan2(-Y, sqrt(X²+Z²)) ∈ [-π/2, π/2]  (+ = up)

Audio effects from 3D geometry:
    gain       = 1 / d^α                (α = attenuation exponent, default 0.8)
    pan        = sin(θ)                  (maps azimuth to [-1, +1])
    delay      = d / 343                 (propagation delay in seconds)
    reverb_wet = sigmoid(β * d - γ)      (smooth distance-dependent reverb)

Public API
----------
    backproject_objects()        → list of Object3D
    compute_3d_spatial_params()  → list of SpatialParams3D dicts
    spatial_3d_stereo_mix()      → stereo [2, samples] with delay + attenuation + reverb
    save_3d_spatial_report()     → JSON report with all 3D coordinates
    print_3d_spatial_report()    → pretty terminal table
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import numpy as np
from scipy.signal import fftconvolve

SAMPLE_RATE = 16_000
SPEED_OF_SOUND = 343.0  # m/s at 20°C


# ═══════════════════════════════════════════════════════════════════════════
# 1.  Camera Intrinsics
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class CameraIntrinsics:
    """Pinhole camera intrinsic parameters.

    If no calibration is available (the typical case for a random photo),
    we assume a reasonable default:
        f = max(W, H)   →  ~60° horizontal field of view
        c = (W/2, H/2)  →  principal point at image centre
    """
    fx: float
    fy: float
    cx: float
    cy: float

    @classmethod
    def from_image_size(cls, width: int, height: int,
                        hfov_deg: float = 60.0) -> "CameraIntrinsics":
        """Estimate intrinsics from image dimensions and assumed H-FoV."""
        f = (width / 2.0) / math.tan(math.radians(hfov_deg / 2.0))
        return cls(fx=f, fy=f, cx=width / 2.0, cy=height / 2.0)


# ═══════════════════════════════════════════════════════════════════════════
# 2.  3D Object Representation
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class Object3D:
    """Stores the 3D position and derived spatial properties of one object."""
    label: str
    bbox: List[float]          # [x1, y1, x2, y2] in pixels
    pixel_centre: Tuple[float, float]   # (u, v)
    depth_value: float         # raw depth from depth estimation

    # Camera-relative 3D coordinates
    X: float = 0.0            # right (+) / left (-)
    Y: float = 0.0            # down (+) / up (-)  (image convention)
    Z: float = 0.0            # forward (into screen)

    # Spherical coordinates (derived)
    distance: float = 0.0     # Euclidean distance from camera origin
    azimuth: float = 0.0      # radians, + = right
    elevation: float = 0.0    # radians, + = up

    azimuth_deg: float = 0.0
    elevation_deg: float = 0.0


# ═══════════════════════════════════════════════════════════════════════════
# 3.  Backprojection: 2D pixel + depth → 3D
# ═══════════════════════════════════════════════════════════════════════════

def backproject_objects(
    bboxes: List[List[float]],
    labels: List[str],
    depths: List[float],
    img_width: int,
    img_height: int,
    camera: Optional[CameraIntrinsics] = None,
    hfov_deg: float = 60.0,
) -> List[Object3D]:
    """Compute 3D positions for all detected objects.

    Parameters
    ----------
    bboxes : list of [x1, y1, x2, y2]
    labels : list of str
    depths : list of float  (from monocular depth estimation, higher = farther)
    img_width, img_height : int
    camera : CameraIntrinsics or None  (estimated from image size if None)
    hfov_deg : float  (assumed horizontal field of view in degrees)

    Returns
    -------
    objects_3d : list of Object3D
    """
    if camera is None:
        camera = CameraIntrinsics.from_image_size(img_width, img_height, hfov_deg)

    objects_3d = []
    for box, label, Z in zip(bboxes, labels, depths):
        x1, y1, x2, y2 = box
        u = (x1 + x2) / 2.0  # pixel centre x
        v = (y1 + y2) / 2.0  # pixel centre y

        # Pinhole backprojection
        X = (u - camera.cx) * Z / camera.fx
        Y = (v - camera.cy) * Z / camera.fy
        # Z already from depth estimation

        # Euclidean distance
        dist = math.sqrt(X**2 + Y**2 + Z**2)

        # Azimuth: angle in the XZ plane (+ = right of camera)
        azimuth = math.atan2(X, Z)

        # Elevation: angle above/below horizon
        # In image coords Y goes down, so negate for "up = positive"
        ground_dist = math.sqrt(X**2 + Z**2)
        elevation = math.atan2(-Y, ground_dist) if ground_dist > 1e-6 else 0.0

        obj = Object3D(
            label=label,
            bbox=box,
            pixel_centre=(u, v),
            depth_value=Z,
            X=X, Y=Y, Z=Z,
            distance=dist,
            azimuth=azimuth,
            elevation=elevation,
            azimuth_deg=math.degrees(azimuth),
            elevation_deg=math.degrees(elevation),
        )
        objects_3d.append(obj)

    return objects_3d


# ═══════════════════════════════════════════════════════════════════════════
# 3b. Listener rotation: yaw + pitch
# ═══════════════════════════════════════════════════════════════════════════

def apply_listener_rotation(
    objects_3d: List[Object3D],
    yaw_deg: float = 0.0,
    pitch_deg: float = 0.0,
) -> List[Object3D]:
    """Rotate object 3D positions to account for listener head orientation.

    The rotation transforms object coordinates from the camera frame
    (camera looks along +Z) into the listener frame (listener faces the
    direction given by yaw + pitch).

    Parameters
    ----------
    objects_3d : list of Object3D
        Objects in camera-relative coordinates (from ``backproject_objects``).
    yaw_deg : float
        Listener yaw in degrees.  **+ve = looking right**,  -ve = left.
    pitch_deg : float
        Listener pitch in degrees.  **+ve = looking up**,  -ve = down.

    Returns
    -------
    rotated : list of Object3D
        New objects with rotated X, Y, Z and recomputed spherical coords.
    """
    if abs(yaw_deg) < 0.01 and abs(pitch_deg) < 0.01:
        return list(objects_3d)  # no rotation needed

    # Negate angles: we rotate the *world* opposite to head direction
    yaw   = math.radians(-yaw_deg)
    pitch = math.radians(-pitch_deg)

    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)

    rotated: List[Object3D] = []
    for obj in objects_3d:
        # Yaw rotation around Y axis: Ry(-yaw)
        x1 =  cy * obj.X + sy * obj.Z
        y1 =  obj.Y
        z1 = -sy * obj.X + cy * obj.Z

        # Pitch rotation around X axis: Rx(-pitch)
        x2 = x1
        y2 =  cp * y1 - sp * z1
        z2 =  sp * y1 + cp * z1

        # Recompute spherical coordinates
        dist = math.sqrt(x2**2 + y2**2 + z2**2)
        azimuth = math.atan2(x2, z2)
        ground_dist = math.sqrt(x2**2 + z2**2)
        elevation = math.atan2(-y2, ground_dist) if ground_dist > 1e-6 else 0.0

        rotated.append(Object3D(
            label=obj.label,
            bbox=obj.bbox,
            pixel_centre=obj.pixel_centre,
            depth_value=obj.depth_value,
            X=x2, Y=y2, Z=z2,
            distance=dist,
            azimuth=azimuth,
            elevation=elevation,
            azimuth_deg=math.degrees(azimuth),
            elevation_deg=math.degrees(elevation),
        ))

    return rotated


# ═══════════════════════════════════════════════════════════════════════════
# 4.  3D → Spatial Audio Parameters
# ═══════════════════════════════════════════════════════════════════════════

def compute_3d_spatial_params(
    objects_3d: List[Object3D],
    attenuation_exp: float = 1.5,
    min_gain_db: float = -30.0,
    max_reverb_wet: float = 0.75,
    enable_delay: bool = True,
    max_delay_s: float = 0.25,
    reference_distance: float = 3.0,
) -> List[Dict]:
    """Compute audio spatial parameters from 3D object positions.

    Enhanced version with HRTF-inspired binaural cues:
      - Aggressive distance attenuation (exp=1.5) so moving closer/farther
        produces clearly audible gain changes.
      - Interaural Time Difference (ITD) from azimuth for binaural realism.
      - Head shadow (ILD) — high-frequency attenuation for the far ear.
      - Stronger reverb gradient to distinguish near vs far objects.

    Parameters
    ----------
    objects_3d : list of Object3D
    attenuation_exp : float
        Distance attenuation exponent. 1.5 = strong perceptual difference.
    min_gain_db : float
        Floor gain in dB.
    max_reverb_wet : float
        Maximum wet reverb ratio.
    enable_delay : bool
        Whether to apply propagation delay.
    max_delay_s : float
        Cap on propagation delay.
    reference_distance : float
        Distance at which gain = 0 dB.

    Returns
    -------
    params : list[dict]
        Keys: pan, gain, reverb, delay_s, distance, azimuth_deg, elevation_deg,
              itd_s, ild_db, elevation_lp_hz, X, Y, Z, label
    """
    if not objects_3d:
        return []

    # Normalise distances so nearest is at reference_distance
    distances = [o.distance for o in objects_3d]
    d_min = min(distances)
    scale = reference_distance / d_min if d_min > 0 else 1.0

    # Head radius for HRTF approximation (~8.75 cm)
    HEAD_RADIUS = 0.0875

    params = []
    for obj in objects_3d:
        d_scaled = obj.distance * scale

        # ── Pan from azimuth (stronger mapping) ──
        # Use azimuth directly for full stereo sweep
        # Clamp to [-1, +1] but scale more aggressively
        pan = math.sin(obj.azimuth)
        # Amplify small azimuth differences (objects near center become
        # more distinguishable left/right)
        pan = math.copysign(abs(pan) ** 0.6, pan)
        pan = max(-1.0, min(1.0, pan))

        # ── Interaural Time Difference (ITD) ──
        # Woodworth formula: ITD = (r/c)(θ + sin(θ))  for |θ| ≤ π/2
        theta = max(-math.pi / 2, min(math.pi / 2, obj.azimuth))
        itd_s = (HEAD_RADIUS / SPEED_OF_SOUND) * (theta + math.sin(theta))
        # Positive ITD → sound arrives at left ear first (source on left)

        # ── Interaural Level Difference (ILD) — head shadow ──
        # Frequency-dependent, but we approximate as a broadband dB diff.
        # ~0 dB in front, up to ~10 dB at 90°
        ild_db = 10.0 * abs(math.sin(obj.azimuth))

        # ── Distance attenuation (aggressive) ──
        if d_scaled > 0:
            gain_linear = (reference_distance / d_scaled) ** attenuation_exp
        else:
            gain_linear = 1.0
        gain_linear = min(gain_linear, 2.0)

        gain_floor = 10.0 ** (min_gain_db / 20.0)
        gain_linear = max(gain_linear, gain_floor)

        # ── Elevation cue ──
        # Objects above → brighter (higher low-pass), below → duller
        # Base cutoff 4000 Hz, range ±2000 Hz from elevation
        elev_norm = obj.elevation / (math.pi / 2)  # [-1, +1]
        elevation_lp_hz = 4000.0 + 2000.0 * elev_norm
        elevation_lp_hz = max(1000.0, min(8000.0, elevation_lp_hz))

        # Also small gain modifier for elevation
        elev_factor = 1.0 + 0.15 * elev_norm
        gain_linear *= elev_factor

        # ── Reverb (stronger gradient) ──
        d_norm = (d_scaled - reference_distance) / reference_distance
        reverb = max_reverb_wet * _sigmoid(d_norm * 3.0)

        # ── Propagation delay ──
        delay_s = 0.0
        if enable_delay:
            delay_s = d_scaled / SPEED_OF_SOUND
            delay_s = min(delay_s, max_delay_s)

        params.append({
            "label": obj.label,
            "pan": pan,
            "gain": float(gain_linear),
            "reverb": float(reverb),
            "delay_s": float(delay_s),
            "distance": float(d_scaled),
            "azimuth_deg": float(obj.azimuth_deg),
            "elevation_deg": float(obj.elevation_deg),
            "itd_s": float(itd_s),
            "ild_db": float(ild_db),
            "elevation_lp_hz": float(elevation_lp_hz),
            "X": float(obj.X * scale),
            "Y": float(obj.Y * scale),
            "Z": float(obj.Z * scale),
        })

    return params


def reposition_objects_from_listener(
    objects_3d: List[Object3D],
    listener_pos: Tuple[float, float, float],
    listener_yaw_deg: float = 0.0,
    listener_pitch_deg: float = 0.0,
) -> List[Object3D]:
    """Recompute object positions relative to a new listener position.

    This is the key function for camera-dependent spatial audio:
    when the listener moves closer to one object, that object's
    distance decreases → it becomes louder, drier, and more centred.

    Parameters
    ----------
    objects_3d : list of Object3D
        Objects in world (camera-origin) coordinates.
    listener_pos : (x, y, z)
        Listener position in the same world coordinate system.
    listener_yaw_deg : float
        Direction listener is facing (+ = right).
    listener_pitch_deg : float
        Up/down tilt.

    Returns
    -------
    repositioned : list of Object3D
        Objects with X, Y, Z, distance, azimuth, elevation recomputed
        relative to the listener position and orientation.
    """
    lx, ly, lz = listener_pos

    # Listener rotation matrix (yaw + pitch) — same as apply_listener_rotation
    yaw = math.radians(-listener_yaw_deg)
    pitch = math.radians(-listener_pitch_deg)
    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)

    repositioned: List[Object3D] = []
    for obj in objects_3d:
        # Vector from listener to object (world coords)
        dx = obj.X - lx
        dy = obj.Y - ly
        dz = obj.Z - lz

        # Rotate into listener's local frame
        # Yaw around Y
        x1 = cy * dx + sy * dz
        y1 = dy
        z1 = -sy * dx + cy * dz
        # Pitch around X
        x2 = x1
        y2 = cp * y1 - sp * z1
        z2 = sp * y1 + cp * z1

        # Recompute spherical coords
        dist = math.sqrt(x2**2 + y2**2 + z2**2)
        if dist < 1e-6:
            dist = 1e-6  # avoid division by zero
        azimuth = math.atan2(x2, z2)
        ground_dist = math.sqrt(x2**2 + z2**2)
        elevation = math.atan2(-y2, ground_dist) if ground_dist > 1e-6 else 0.0

        repositioned.append(Object3D(
            label=obj.label,
            bbox=obj.bbox,
            pixel_centre=obj.pixel_centre,
            depth_value=obj.depth_value,
            X=x2, Y=y2, Z=z2,
            distance=dist,
            azimuth=azimuth,
            elevation=elevation,
            azimuth_deg=math.degrees(azimuth),
            elevation_deg=math.degrees(elevation),
        ))

    return repositioned


def _sigmoid(x: float) -> float:
    """Numerically stable sigmoid."""
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    else:
        z = math.exp(x)
        return z / (1.0 + z)


# ═══════════════════════════════════════════════════════════════════════════
# 5.  3D Spatial Stereo Mix
# ═══════════════════════════════════════════════════════════════════════════

def _synthetic_room_ir(
    distance: float,
    reverb_wet: float,
    sr: int = SAMPLE_RATE,
    max_duration_s: float = 0.8,
    n_taps: int = 8,
    base_decay: float = 0.50,
) -> np.ndarray:
    """Generate a distance-dependent synthetic impulse response.

    Farther objects get longer IRs with more late reflections.
    """
    rng = np.random.default_rng(42)
    # IR duration scales with reverb amount
    dur = 0.2 + max_duration_s * reverb_wet
    length = int(sr * dur)
    ir = np.zeros(length, dtype=np.float32)
    ir[0] = 1.0

    # Early reflections
    for t in range(1, n_taps + 1):
        delay_ms = 12 * t + rng.uniform(0, 8)  # 12-100 ms range
        delay_samples = int(sr * delay_ms / 1000.0)
        if delay_samples < length:
            amplitude = base_decay ** t * (0.6 + 0.4 * rng.random())
            ir[delay_samples] += amplitude

    # Late diffuse tail (exponential decay noise)
    if reverb_wet > 0.2:
        tail_start = int(sr * 0.08)
        tail = rng.normal(0, 0.1, length - tail_start).astype(np.float32)
        decay_env = np.exp(-np.arange(length - tail_start) / (sr * 0.15 * reverb_wet))
        ir[tail_start:] += tail * decay_env * reverb_wet * 0.3

    # Low-pass for naturalness
    ir = np.convolve(ir, np.ones(5) / 5, mode="same")
    return ir / (np.max(np.abs(ir)) + 1e-8)


def _constant_power_pan(pan: float) -> Tuple[float, float]:
    """Constant-power pan law. pan ∈ [-1, +1] → (left_gain, right_gain)."""
    theta = (pan + 1.0) / 2.0 * (math.pi / 2.0)
    return math.cos(theta), math.sin(theta)


def _hrtf_filters(azimuth_deg: float, elevation_lp_hz: float,
                   ild_db: float, sr: int = SAMPLE_RATE,
                   n_taps: int = 65) -> Tuple[np.ndarray, np.ndarray]:
    """Synthesize approximate HRTF FIR filters for left and right ears.

    This is a simplified analytical HRTF that captures the three main cues:
    1) ITD (handled separately via delay)
    2) ILD — level difference between ears from head shadow
    3) Spectral shaping — pinna filtering approximation

    Returns two FIR filters (left_ir, right_ir) of length n_taps.
    """
    # Base filter: slight low-pass shaped by pinna (~3-6 kHz notch)
    t = np.arange(n_taps)
    centre = n_taps // 2

    # Gaussian envelope
    sigma = n_taps / 6.0
    envelope = np.exp(-0.5 * ((t - centre) / sigma) ** 2)

    # Sinc low-pass at elevation_lp_hz
    fc = elevation_lp_hz / sr
    sinc_filter = np.sinc(2 * fc * (t - centre)) * envelope
    sinc_filter /= (np.sum(np.abs(sinc_filter)) + 1e-12)

    # ILD: attenuate the far ear
    # positive azimuth (right) → right ear gets full signal,
    # left ear gets attenuated (and vice versa)
    ild_linear = 10.0 ** (-ild_db / 20.0)

    if azimuth_deg >= 0:
        # Source on the right
        right_ir = sinc_filter.copy()
        left_ir = sinc_filter * ild_linear
        # Also apply extra low-pass to shadowed ear (head blocks high freqs)
        shadow_fc = max(1500.0, elevation_lp_hz * 0.5) / sr
        shadow_sinc = np.sinc(2 * shadow_fc * (t - centre)) * envelope
        shadow_sinc /= (np.sum(np.abs(shadow_sinc)) + 1e-12)
        left_ir = np.convolve(left_ir, shadow_sinc, mode='same')
    else:
        # Source on the left
        left_ir = sinc_filter.copy()
        right_ir = sinc_filter * ild_linear
        shadow_fc = max(1500.0, elevation_lp_hz * 0.5) / sr
        shadow_sinc = np.sinc(2 * shadow_fc * (t - centre)) * envelope
        shadow_sinc /= (np.sum(np.abs(shadow_sinc)) + 1e-12)
        right_ir = np.convolve(right_ir, shadow_sinc, mode='same')

    return left_ir.astype(np.float32), right_ir.astype(np.float32)


def _spectral_gate(
    signal: np.ndarray,
    noise_floor_db: float = -30.0,
    n_fft: int = 1024,
    hop: int = 256,
) -> np.ndarray:
    """Simple spectral gating to suppress low-energy noise in a mono track.

    Frames whose energy falls below *noise_floor_db* (relative to the track
    peak) are attenuated by a soft gate.  This removes hiss / artefacts from
    per-object AudioLDM generations without touching the main content.
    """
    from numpy.lib.stride_tricks import sliding_window_view  # numpy ≥ 1.20

    sig = signal.astype(np.float64)
    peak = np.max(np.abs(sig)) + 1e-12
    threshold = peak * (10.0 ** (noise_floor_db / 20.0))

    # Frame-level RMS
    n_pad = (n_fft - len(sig) % n_fft) % n_fft
    padded = np.pad(sig, (0, n_pad))
    frames = padded.reshape(-1, hop)  # rough frame split
    rms = np.sqrt(np.mean(frames ** 2, axis=1) + 1e-18)

    # Soft gate: smooth mask per frame (0 → muted, 1 → pass)
    gate = np.clip((rms - threshold * 0.5) / (threshold * 0.5 + 1e-12), 0.0, 1.0)

    # Expand gate to sample level
    gate_samples = np.repeat(gate, hop)[: len(sig)]
    # Smooth the gate to avoid clicks (simple moving avg)
    kernel_len = min(hop, 64)
    kernel = np.ones(kernel_len) / kernel_len
    gate_samples = np.convolve(gate_samples, kernel, mode="same")
    gate_samples = np.clip(gate_samples, 0.0, 1.0)

    return (sig * gate_samples).astype(np.float32)


def spatial_3d_stereo_mix(
    mono_waves: np.ndarray,
    spatial_params: List[Dict],
    sr: int = SAMPLE_RATE,
    remix_wave: Optional[np.ndarray] = None,
    remix_weight: float = 0.55,
) -> np.ndarray:
    """Hybrid 3D stereo mix: clean remix bed + spatialized per-object tracks.

    Strategy ("game audio" approach):
    1. The **remix** (from cycle_mix → AudioLDM) is high quality but mono/flat.
       We duplicate it to stereo center as a clean "ambient bed".
    2. The **per-object tracks** are noisier but carry spatial information.
       We spectral-gate them, apply full HRTF / distance / reverb, then
       layer them on top at reduced weight.
    3. Final blend:  remix_weight × remix_stereo  +  (1 - remix_weight) × spatial_objects
       When listener is equidistant from all objects → mostly clean remix.
       When close to one object → that object's track swells and becomes
       directional, layered on top of the remix.

    If *remix_wave* is None, falls back to pure per-object spatial mix.

    Parameters
    ----------
    mono_waves : np.ndarray, shape [N, 1, samples]
        Per-object mono tracks from AudioLDM.
    spatial_params : list[dict]
        From ``compute_3d_spatial_params``.
    remix_wave : np.ndarray, shape [1, 1, samples], optional
        The remix mono track from AudioLDM (the "original" output).
    remix_weight : float
        Blend factor for the remix bed (0.0–1.0). Default 0.55 means
        55% remix clarity + 45% spatial detail.

    Returns
    -------
    stereo : np.ndarray, shape [2, samples]
    """
    n_objects = mono_waves.shape[0]
    n_samples = mono_waves.shape[-1]

    # ── Spectral-gate per-object tracks to remove noise artefacts ──
    gated_waves = np.empty_like(mono_waves)
    for i in range(n_objects):
        gated_waves[i, 0] = _spectral_gate(mono_waves[i, 0], noise_floor_db=-28.0)

    # Compute max delay (propagation + ITD)
    max_delay_samples = 0
    for p in spatial_params:
        prop_delay = int(p.get("delay_s", 0) * sr)
        itd_samples = int(abs(p.get("itd_s", 0)) * sr)
        max_delay_samples = max(max_delay_samples, prop_delay + itd_samples)

    total_samples = n_samples + max_delay_samples + 128  # extra for filter tails

    stereo = np.zeros((2, total_samples), dtype=np.float64)

    for i in range(n_objects):
        signal = gated_waves[i, 0].astype(np.float64)
        p = spatial_params[i]

        # ── 1. Distance gain attenuation ──
        signal *= p["gain"]

        # ── 2. Reverb (distance-dependent room IR) ──
        wet = p["reverb"]
        if wet > 0.01:
            ir = _synthetic_room_ir(p["distance"], wet, sr=sr)
            reverbed = fftconvolve(signal, ir, mode="full")[:n_samples]
            signal = (1.0 - wet) * signal + wet * reverbed

        # ── 3. HRTF binaural filtering ──
        azimuth_deg = p.get("azimuth_deg", 0.0)
        elevation_lp = p.get("elevation_lp_hz", 4000.0)
        ild_db = p.get("ild_db", 0.0)

        left_ir, right_ir = _hrtf_filters(azimuth_deg, elevation_lp,
                                           ild_db, sr=sr)

        left_signal = fftconvolve(signal, left_ir, mode="full")[:n_samples]
        right_signal = fftconvolve(signal, right_ir, mode="full")[:n_samples]

        # ── 4. Constant-power pan (on top of HRTF for clarity) ──
        l_gain, r_gain = _constant_power_pan(p["pan"])
        left_signal *= l_gain
        right_signal *= r_gain

        # ── 5. Propagation delay + ITD ──
        prop_delay = int(p.get("delay_s", 0) * sr)
        itd_s = p.get("itd_s", 0.0)
        itd_samples = int(abs(itd_s) * sr)

        if itd_s >= 0:
            # Source on left → left ear hears first
            left_delay = prop_delay
            right_delay = prop_delay + itd_samples
        else:
            # Source on right → right ear hears first
            left_delay = prop_delay + itd_samples
            right_delay = prop_delay

        stereo[0, left_delay:left_delay + n_samples] += left_signal
        stereo[1, right_delay:right_delay + n_samples] += right_signal

    # Trim
    stereo = stereo[:, :n_samples + max_delay_samples]

    # ── Hybrid blend with remix bed ──
    if remix_wave is not None:
        remix_mono = remix_wave.flatten().astype(np.float64)
        # Match length to spatial stereo
        out_len = stereo.shape[1]
        if len(remix_mono) < out_len:
            remix_mono = np.pad(remix_mono, (0, out_len - len(remix_mono)))
        else:
            remix_mono = remix_mono[:out_len]

        # Remix → centre stereo (equal L/R)
        remix_stereo = np.stack([remix_mono, remix_mono])  # [2, samples]

        # Normalise each component independently before blending
        sp_peak = np.max(np.abs(stereo)) + 1e-12
        rx_peak = np.max(np.abs(remix_stereo)) + 1e-12
        stereo_norm = stereo / sp_peak
        remix_norm = remix_stereo / rx_peak

        # Blend: remix provides clarity, spatial provides directionality
        rw = float(np.clip(remix_weight, 0.0, 1.0))
        blended = rw * remix_norm + (1.0 - rw) * stereo_norm

        # Final normalise
        peak = np.max(np.abs(blended)) + 1e-12
        blended = blended / peak * 0.95
        return blended.astype(np.float32)

    # ── Fallback: pure spatial (no remix available) ──
    peak = np.max(np.abs(stereo))
    if peak > 0:
        stereo = stereo / peak * 0.95

    return stereo.astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════════
# 6.  Reporting
# ═══════════════════════════════════════════════════════════════════════════

def save_3d_spatial_report(
    objects_3d: List[Object3D],
    spatial_params: List[Dict],
    save_path: str,
    extra: Optional[Dict] = None,
):
    """Save a JSON report of all 3D positions and audio parameters."""
    report = {
        "objects": [],
    }
    if extra:
        report.update(extra)

    for obj, p in zip(objects_3d, spatial_params):
        report["objects"].append({
            "label": obj.label,
            "bbox": obj.bbox,
            "pixel_centre": list(obj.pixel_centre),
            "depth_raw": float(obj.depth_value),
            "position_3d": {"X": p["X"], "Y": p["Y"], "Z": p["Z"]},
            "distance_m": p["distance"],
            "azimuth_deg": p["azimuth_deg"],
            "elevation_deg": p["elevation_deg"],
            "audio": {
                "pan": p["pan"],
                "gain": p["gain"],
                "gain_dB": float(20 * math.log10(max(p["gain"], 1e-8))),
                "reverb_wet": p["reverb"],
                "delay_ms": p["delay_s"] * 1000,
            },
        })

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    with open(save_path, "w") as fp:
        json.dump(report, fp, indent=2)
    print(f"[3D SPATIAL] Report saved → {save_path}")


def print_3d_spatial_report(objects_3d: List[Object3D], spatial_params: List[Dict]):
    """Pretty-print 3D spatial audio placement."""
    print()
    print("╔═══════════════════════════════════════════════════════════════════════╗")
    print("║                   3D SPATIAL AUDIO — Object Placement               ║")
    print("╠═══════════════════════════════════════════════════════════════════════╣")
    print("║  #  │ Label             │  X      Y      Z     │ Dist  │ Az°   El°  ║")
    print("╟─────┼───────────────────┼───────────────────────┼───────┼────────────╢")
    for i, (obj, p) in enumerate(zip(objects_3d, spatial_params)):
        gain_db = 20 * math.log10(max(p["gain"], 1e-8))
        print(
            "║ %2d  │ %-17s │ %+6.1f %+6.1f %6.1f │ %5.1f │ %+5.1f %+5.1f ║"
            % (i, obj.label[:17], p["X"], p["Y"], p["Z"],
               p["distance"], p["azimuth_deg"], p["elevation_deg"])
        )
    print("╟─────┼───────────────────┼───────────────────────┼───────┼────────────╢")
    print("║  #  │ Label             │ Pan    Gain(dB) Reverb│ Delay │            ║")
    print("╟─────┼───────────────────┼───────────────────────┼───────┼────────────╢")
    for i, (obj, p) in enumerate(zip(objects_3d, spatial_params)):
        gain_db = 20 * math.log10(max(p["gain"], 1e-8))
        dir_str = "L" if p["pan"] < -0.2 else ("R" if p["pan"] > 0.2 else "C")
        print(
            "║ %2d  │ %-17s │ %+5.2f%s %+6.1fdB %4.0f%% │ %4.1fms│            ║"
            % (i, obj.label[:17], p["pan"], dir_str, gain_db,
               p["reverb"] * 100, p["delay_s"] * 1000)
        )
    print("╚═══════════════════════════════════════════════════════════════════════╝")
    print()
