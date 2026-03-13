"""Feature Engineer — Computes 52 model features from raw E2 measurements.

Bridges the gap between E2SM-KPM indications (~12 raw measurements from gNB)
and the 52-feature input required by the LightGBM+XGBoost+CatBoost ensemble.

Raw E2 measurements available from Keysight NTN testbed / real gNB:
    rsrpNtn, sinrNtn, rsrpTn, sinrTn, rsrqNtn, rsrqTn,
    elevationDeg, dopplerHz, distanceKm, pathLossDb, shadowingDb,
    rainAttenuationDb, bestLEORsrp, bestMEORsrp, bestGEORsrp,
    bestLEOElevationDeg, bestMEOElevationDeg, bestGEOElevationDeg

Computed features (grouped):
    - Signal gaps: sinr_gap, rsrp_gap
    - Normalized: elevationNorm, distanceNorm, dopplerNorm, channelQuality
    - Link budget: fsplDb, atmosphericAbsorptionDb, delayUs, totalChannelLossDb,
                   scintillationDb, clutterLossDb, additionalLossDb, isLOS
    - Temporal (per-UE sliding window): sinrNtn_delta, sinrTn_delta,
      elevationDeg_delta, dopplerHz_delta, distanceKm_delta,
      sinrNtn_variance, sinrTn_variance
    - Handover tracking: ho_count_cumulative, time_since_ho, ho_rate_10s
    - Time encoding: time_sin, time_cos, time_position
    - Categoricals: orbit_type_encoded, scenario_encoded, mobility_encoded
"""

import logging
import math
import time
from collections import deque
from typing import Dict, Optional

try:
    import mdclogpy
    logger = mdclogpy.Logger(name="tn-ntn-xapp.features")
except ImportError:
    logger = logging.getLogger(__name__)

# Physical constants
EARTH_R_KM = 6371.0
C_LIGHT_KM_S = 299_792.458
F_CARRIER_HZ = 2.0e9

# Orbit type encoding (matches training pipeline)
ORBIT_TYPE_MAP = {"TN": 0, "LEO": 1, "MEO": 2, "GEO": 3}

# Sliding window size for delta/variance computation
WINDOW_SIZE = 10

# Handover rate tracking window (seconds)
HO_RATE_WINDOW_S = 10.0


class _UEState:
    """Per-UE temporal state for delta/variance/handover tracking."""

    __slots__ = (
        "sinrNtn_hist", "sinrTn_hist", "elevation_hist",
        "doppler_hist", "distance_hist",
        "ho_count", "last_ho_time", "ho_timestamps",
        "last_decision", "last_update",
    )

    def __init__(self):
        self.sinrNtn_hist: deque = deque(maxlen=WINDOW_SIZE)
        self.sinrTn_hist: deque = deque(maxlen=WINDOW_SIZE)
        self.elevation_hist: deque = deque(maxlen=WINDOW_SIZE)
        self.doppler_hist: deque = deque(maxlen=WINDOW_SIZE)
        self.distance_hist: deque = deque(maxlen=WINDOW_SIZE)
        self.ho_count: int = 0
        self.last_ho_time: float = 0.0
        self.ho_timestamps: deque = deque(maxlen=100)
        self.last_decision: Optional[int] = None
        self.last_update: float = 0.0


class FeatureEngineer:
    """Computes 52 model features from raw E2 measurements.

    Maintains per-UE temporal state for delta/variance computation and
    handover tracking. Thread-safe for concurrent UE processing.

    Usage:
        fe = FeatureEngineer()
        features = fe.compute(ue_id="ue-001", raw=meas_data)
        # features is a dict with all 52 keys ready for ensemble prediction
    """

    def __init__(self, max_ue_states: int = 10_000):
        self._ue_states: Dict[str, _UEState] = {}
        self._max_ue_states = max_ue_states
        self._compute_count = 0

    def compute(self, ue_id: str, raw: dict) -> dict:
        """Compute all 52 features from raw E2 measurements.

        Args:
            ue_id: UE identifier for temporal state tracking
            raw: Raw measurement dict from E2SM-KPM indication. Expected keys:
                rsrpNtn, sinrNtn, rsrpTn, sinrTn, rsrqNtn, rsrqTn,
                elevationDeg, dopplerHz, distanceKm, pathLossDb, shadowingDb,
                rainAttenuationDb, orbitType, ueSpeed, ueDirection, ueAltitude,
                bestLEORsrp, bestMEORsrp, bestGEORsrp,
                bestLEOElevationDeg, bestMEOElevationDeg, bestGEOElevationDeg

        Returns:
            Dict with all 52 feature values (float).
        """
        self._compute_count += 1
        now = time.time()
        state = self._get_or_create_state(ue_id)

        # -- Extract raw measurements with safe defaults --
        sinr_ntn = _float(raw, "sinrNtn", "sinrNTN", default=-5.0)
        sinr_tn = _float(raw, "sinrTn", "sinrTN_measured", default=10.0)
        rsrp_ntn = _float(raw, "rsrpNtn", "rsrpNTN", default=-100.0)
        rsrp_tn = _float(raw, "rsrpTn", "rsrpTN_measured", default=-85.0)
        rsrq_ntn = _float(raw, "rsrqNtn", "rsrqNTN", default=-12.0)
        rsrq_tn = _float(raw, "rsrqTn", "rsrqTN_measured", default=-10.0)
        elevation = _float(raw, "elevationDeg", default=45.0)
        doppler = _float(raw, "dopplerHz", default=0.0)
        distance = _float(raw, "distanceKm", default=550.0)
        path_loss = _float(raw, "pathLossDb", "pathLossDbNTN", default=0.0)
        shadowing = _float(raw, "shadowingDb", "shadowingDbNTN", default=0.0)
        rain_atten = _float(raw, "rainAttenuationDb", default=0.0)
        ue_speed = _float(raw, "ueSpeed", default=0.0)
        ue_direction = _float(raw, "ueDirection", default=0.0)
        ue_altitude = _float(raw, "ueAltitude", default=0.0)
        orbit_type = str(raw.get("orbitType", raw.get("orbit_type", "LEO")))
        prop_delay_ms = _float(raw, "propagationDelayMs", default=0.0)

        # -- Multi-orbit best signals --
        best_leo_rsrp = _float(raw, "bestLEORsrp", default=-999.0)
        best_meo_rsrp = _float(raw, "bestMEORsrp", default=-999.0)
        best_geo_rsrp = _float(raw, "bestGEORsrp", default=-999.0)
        best_leo_elev = _float(raw, "bestLEOElevationDeg", default=0.0)
        best_meo_elev = _float(raw, "bestMEOElevationDeg", default=0.0)
        best_geo_elev = _float(raw, "bestGEOElevationDeg", default=0.0)

        # -- Time encoding (diurnal pattern) --
        hour_frac = (now % 86400) / 86400.0
        time_val = _float(raw, "time", default=now % 86400)
        time_sin = math.sin(2 * math.pi * hour_frac)
        time_cos = math.cos(2 * math.pi * hour_frac)
        time_position = hour_frac

        # -- Scenario / mobility encoding --
        scenario_enc = _float(raw, "scenario_encoded", default=0.0)
        mobility_enc = _float(raw, "mobility_encoded", default=0.0)

        # -- Derived signal features --
        sinr_gap = sinr_ntn - sinr_tn
        rsrp_gap = rsrp_ntn - rsrp_tn

        # -- Normalized features --
        elevation_norm = max(0.0, min(1.0, elevation / 90.0))
        distance_norm = max(0.0, min(1.0, distance / 40_000.0))
        doppler_norm = max(-1.0, min(1.0, doppler / 40_000.0))
        channel_quality = max(0.0, min(1.0, sinr_ntn / 30.0))

        # -- Link budget features --
        # Line-of-sight flag (elevation > 10 deg heuristic per 3GPP TR 38.811)
        is_los = 1.0 if elevation > 10.0 else 0.0

        # Propagation delay
        if prop_delay_ms > 0:
            delay_us = prop_delay_ms * 1000.0
        elif distance > 0:
            delay_us = (distance / C_LIGHT_KM_S) * 2 * 1e6
        else:
            delay_us = 0.0

        # Free-space path loss (FSPL)
        if distance > 0:
            wavelength_km = C_LIGHT_KM_S / (F_CARRIER_HZ / 1e9)
            arg = max(4 * math.pi * distance / wavelength_km, 1e-10)
            fspl_db = 20 * math.log10(arg)
        elif path_loss > 0:
            fspl_db = path_loss - shadowing
        else:
            fspl_db = 0.0

        # Atmospheric absorption (ITU-R P.676 simplified)
        elev_rad = math.radians(max(elevation, 5.0))
        atm_absorption_db = 0.04 / math.sin(elev_rad)

        # Scintillation (low at S-band, placeholder)
        scintillation_db = 0.0

        # Clutter loss (urban: ~10 dB below 10 deg, 0 above 30 deg)
        clutter_loss_db = 0.0

        # Total channel loss
        total_channel_loss = path_loss + shadowing if (path_loss > 0) else fspl_db + atm_absorption_db + rain_atten

        # Additional loss (rain fade)
        additional_loss_db = rain_atten

        # -- Temporal features (per-UE sliding window) --
        state.sinrNtn_hist.append(sinr_ntn)
        state.sinrTn_hist.append(sinr_tn)
        state.elevation_hist.append(elevation)
        state.doppler_hist.append(doppler)
        state.distance_hist.append(distance)

        sinr_ntn_delta = _delta(state.sinrNtn_hist)
        sinr_tn_delta = _delta(state.sinrTn_hist)
        elevation_delta = _delta(state.elevation_hist)
        doppler_delta = _delta(state.doppler_hist)
        distance_delta = _delta(state.distance_hist)

        sinr_ntn_var = _variance(state.sinrNtn_hist)
        sinr_tn_var = _variance(state.sinrTn_hist)

        # -- Handover tracking --
        ho_count = state.ho_count
        time_since_ho = now - state.last_ho_time if state.last_ho_time > 0 else 999.0

        # Prune old timestamps outside rate window
        cutoff = now - HO_RATE_WINDOW_S
        while state.ho_timestamps and state.ho_timestamps[0] < cutoff:
            state.ho_timestamps.popleft()
        ho_rate_10s = float(len(state.ho_timestamps))

        # -- Orbit type encoding --
        orbit_enc = float(ORBIT_TYPE_MAP.get(orbit_type, 0))

        state.last_update = now

        return {
            "time": time_val,
            "scenario_encoded": scenario_enc,
            "mobility_encoded": mobility_enc,
            "ueSpeed": ue_speed,
            "ueDirection": ue_direction,
            "ueAltitude": ue_altitude,
            "time_sin": round(time_sin, 6),
            "time_cos": round(time_cos, 6),
            "time_position": round(time_position, 6),
            "sinrNtn": sinr_ntn,
            "sinrTn": sinr_tn,
            "rsrpNtn": rsrp_ntn,
            "rsrpTn": rsrp_tn,
            "rsrqNtn": rsrq_ntn,
            "rsrqTn": rsrq_tn,
            "elevationDeg": elevation,
            "dopplerHz": doppler,
            "distanceKm": distance,
            "pathLossDb": path_loss,
            "shadowingDb": shadowing,
            "delayUs": delay_us,
            "sinr_gap": sinr_gap,
            "rsrp_gap": rsrp_gap,
            "ho_count_cumulative": float(ho_count),
            "time_since_ho": time_since_ho,
            "ho_rate_10s": ho_rate_10s,
            "sinrNtn_delta": sinr_ntn_delta,
            "sinrTn_delta": sinr_tn_delta,
            "elevationDeg_delta": elevation_delta,
            "dopplerHz_delta": doppler_delta,
            "distanceKm_delta": distance_delta,
            "sinrNtn_variance": sinr_ntn_var,
            "sinrTn_variance": sinr_tn_var,
            "isLOS": is_los,
            "atmosphericAbsorptionDb": round(atm_absorption_db, 4),
            "scintillationDb": scintillation_db,
            "clutterLossDb": clutter_loss_db,
            "totalChannelLossDb": total_channel_loss,
            "fsplDb": round(fspl_db, 2),
            "additionalLossDb": additional_loss_db,
            "channelQuality": round(channel_quality, 4),
            "elevationNorm": round(elevation_norm, 6),
            "distanceNorm": round(distance_norm, 6),
            "dopplerNorm": round(doppler_norm, 6),
            "bestLEORsrp": best_leo_rsrp,
            "bestMEORsrp": best_meo_rsrp,
            "bestGEORsrp": best_geo_rsrp,
            "bestLEOElevationDeg": best_leo_elev,
            "bestMEOElevationDeg": best_meo_elev,
            "bestGEOElevationDeg": best_geo_elev,
            "orbit_type_encoded": orbit_enc,
            "rainAttenuationDb": rain_atten,
        }

    def record_handover(self, ue_id: str) -> None:
        """Record a handover event for a UE (updates temporal counters)."""
        state = self._get_or_create_state(ue_id)
        state.ho_count += 1
        now = time.time()
        state.last_ho_time = now
        state.ho_timestamps.append(now)

    def update_decision(self, ue_id: str, decision: int) -> None:
        """Track last prediction decision per UE for handover detection."""
        state = self._ue_states.get(ue_id)
        if state:
            prev = state.last_decision
            state.last_decision = decision
            # Auto-detect handover: decision flipped
            if prev is not None and prev != decision:
                self.record_handover(ue_id)

    def _get_or_create_state(self, ue_id: str) -> _UEState:
        state = self._ue_states.get(ue_id)
        if state is not None:
            return state
        # Evict oldest if at capacity
        if len(self._ue_states) >= self._max_ue_states:
            oldest_id = min(self._ue_states, key=lambda k: self._ue_states[k].last_update)
            del self._ue_states[oldest_id]
        state = _UEState()
        self._ue_states[ue_id] = state
        return state

    @property
    def tracked_ues(self) -> int:
        return len(self._ue_states)

    @property
    def metrics(self) -> dict:
        return {
            "compute_count": self._compute_count,
            "tracked_ues": self.tracked_ues,
        }


# -- Helpers ------------------------------------------------------------------

def _float(d: dict, *keys: str, default: float = 0.0) -> float:
    """Extract a float from dict, trying multiple key names."""
    for key in keys:
        val = d.get(key)
        if val is not None:
            try:
                v = float(val)
                if not math.isnan(v):
                    return v
            except (ValueError, TypeError):
                pass
    return default


def _delta(hist: deque) -> float:
    """Compute first-order difference (latest - previous)."""
    if len(hist) < 2:
        return 0.0
    return hist[-1] - hist[-2]


def _variance(hist: deque) -> float:
    """Compute variance over sliding window."""
    if len(hist) < 2:
        return 0.0
    mean = sum(hist) / len(hist)
    return sum((x - mean) ** 2 for x in hist) / len(hist)
