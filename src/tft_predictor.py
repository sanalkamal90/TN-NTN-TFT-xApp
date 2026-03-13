"""TFT Model Wrapper for proactive TN-NTN broadband handover prediction.

Loads a Temporal Fusion Transformer checkpoint (pytorch_forecasting) and
provides per-UE sliding-window inference with 12-step lookahead and
7 uncertainty quantiles per target.

Targets: rsrp_ntn, elevation_norm
Known future reals: distance_norm, time_sin, time_cos
Unknown reals: sinr_ntn, rsrq_ntn, doppler_hz, path_loss_ntn,
               sinr_tn, rsrp_tn, ue_speed, sinr_gap, filtered_sinr
Static categoricals: ue_id
Static reals: scenario_enc, mobility_enc
"""

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import mdclogpy
    logger = mdclogpy.Logger(name="tft_predictor")
except ImportError:
    logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# TFT configuration constants (must match training pipeline exactly)
# ---------------------------------------------------------------------------

TFT_TARGETS = ["rsrp_ntn", "elevation_norm"]
TFT_QUANTILES = [0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98]

TFT_COLUMN_MAP = {
    "ueId": "ue_id",
    "sinrNTN": "sinr_ntn",
    "rsrpNTN": "rsrp_ntn",
    "rsrqNTN": "rsrq_ntn",
    "dopplerHz": "doppler_hz",
    "pathLossDbNTN": "path_loss_ntn",
    "sinrTN_measured": "sinr_tn",
    "rsrpTN_measured": "rsrp_tn",
    "ueSpeed": "ue_speed",
    "sinr_gap_measured": "sinr_gap",
    "filteredNtnSinr": "filtered_sinr",
    "elevationDeg": "elevation_deg",
    "distanceKm": "distance_km",
    "scenario_encoded": "scenario_enc",
    "mobility_encoded": "mobility_enc",
    "time_tick": "time_tick",
}

TFT_KNOWN_FUTURE_REALS = ["distance_norm", "time_sin", "time_cos"]
TFT_UNKNOWN_REALS = [
    "sinr_ntn", "rsrq_ntn", "doppler_hz", "path_loss_ntn",
    "sinr_tn", "rsrp_tn", "ue_speed", "sinr_gap", "filtered_sinr",
]
TFT_STATIC_CATEGORICALS = ["ue_id"]
TFT_STATIC_REALS = ["scenario_enc", "mobility_enc"]
TFT_ENCODER_LENGTH = 60
TFT_PREDICTION_LENGTH = 12

# Default alert thresholds (overridable via A1 policy)
DEFAULT_RSRP_ALERT_THRESHOLD = -110.0       # dBm
DEFAULT_ELEVATION_ALERT_THRESHOLD = 0.15     # normalized (~13.5 deg)

# Default operational parameters
DEFAULT_TFT_EVERY_N = 5       # Run TFT every N indications
DEFAULT_MIN_WINDOW_DEPTH = 30  # Minimum rows before first inference


# ---------------------------------------------------------------------------
# TFTPrediction dataclass
# ---------------------------------------------------------------------------

@dataclass
class TFTPrediction:
    """Result of a single TFT inference for one UE."""

    ue_id: str
    timestamp: float
    latency_ms: float
    rsrp_ntn_quantiles: np.ndarray       # shape (12, 7)
    elevation_norm_quantiles: np.ndarray  # shape (12, 7)
    rsrp_alert_step: Optional[int] = None
    elevation_alert_step: Optional[int] = None
    urgency: str = "normal"              # "normal", "prepare", "imminent"
    recommended_action: str = "NONE"     # "NONE", "PREPARE_HANDOVER", "EXECUTE_HANDOVER"

    def to_dict(self) -> dict:
        """Serialize prediction to JSON-safe dict."""
        median_idx = 3
        return {
            "ue_id": self.ue_id,
            "timestamp": self.timestamp,
            "latency_ms": round(self.latency_ms, 2),
            "horizon_steps": TFT_PREDICTION_LENGTH,
            "rsrp_ntn_forecast": {
                "median": self.rsrp_ntn_quantiles[:, median_idx].tolist(),
                "q02": self.rsrp_ntn_quantiles[:, 0].tolist(),
                "q10": self.rsrp_ntn_quantiles[:, 1].tolist(),
                "q25": self.rsrp_ntn_quantiles[:, 2].tolist(),
                "q75": self.rsrp_ntn_quantiles[:, 4].tolist(),
                "q90": self.rsrp_ntn_quantiles[:, 5].tolist(),
                "q98": self.rsrp_ntn_quantiles[:, 6].tolist(),
                "unit": "dBm",
            },
            "elevation_norm_forecast": {
                "median": self.elevation_norm_quantiles[:, median_idx].tolist(),
                "q02": self.elevation_norm_quantiles[:, 0].tolist(),
                "q10": self.elevation_norm_quantiles[:, 1].tolist(),
                "q25": self.elevation_norm_quantiles[:, 2].tolist(),
                "q75": self.elevation_norm_quantiles[:, 4].tolist(),
                "q90": self.elevation_norm_quantiles[:, 5].tolist(),
                "q98": self.elevation_norm_quantiles[:, 6].tolist(),
            },
            "alert": {
                "rsrp_alert_step": self.rsrp_alert_step,
                "elevation_alert_step": self.elevation_alert_step,
                "urgency": self.urgency,
                "recommended_action": self.recommended_action,
            },
        }


# ---------------------------------------------------------------------------
# TFT Inference Wrapper
# ---------------------------------------------------------------------------

class TFTPredictor:
    """Loads TFT checkpoint and provides real-time per-UE inference.

    Manages per-UE sliding windows (target 60 rows, minimum 30).
    Runs 12-step forecast with 7 quantiles per target.
    Detects RSRP/elevation degradation and assigns urgency levels.
    """

    def __init__(
        self,
        checkpoint_path: str,
        device: str = "cpu",
        rsrp_alert_threshold: float = DEFAULT_RSRP_ALERT_THRESHOLD,
        elevation_alert_threshold: float = DEFAULT_ELEVATION_ALERT_THRESHOLD,
        tft_every_n: int = DEFAULT_TFT_EVERY_N,
        min_window_depth: int = DEFAULT_MIN_WINDOW_DEPTH,
    ):
        self.checkpoint_path = checkpoint_path
        self.device = device
        self.model = None
        self.ready = False

        self.rsrp_alert_threshold = rsrp_alert_threshold
        self.elevation_alert_threshold = elevation_alert_threshold
        self.tft_every_n = tft_every_n
        self.min_window_depth = min_window_depth

        self._windows: Dict[str, deque] = {}
        self._indication_counts: Dict[str, int] = {}
        self._latest_predictions: Dict[str, TFTPrediction] = {}

        self._total_predictions = 0
        self._total_latency_ms = 0.0

        self._load_model()

    def _load_model(self) -> None:
        """Load TFT checkpoint from disk."""
        try:
            import torch
            from pytorch_forecasting import TemporalFusionTransformer

            logger.info("Loading TFT model from %s...", self.checkpoint_path)
            self.model = TemporalFusionTransformer.load_from_checkpoint(
                self.checkpoint_path, map_location=self.device,
            )
            self.model.eval()
            self.model.to(self.device)
            self.ready = True
            total_params = sum(p.numel() for p in self.model.parameters())
            logger.info(
                "TFT loaded: %s parameters, device=%s",
                f"{total_params:,}", self.device,
            )
        except ImportError:
            logger.warning("pytorch-forecasting not installed -- TFT disabled")
        except FileNotFoundError:
            logger.warning("TFT checkpoint not found: %s", self.checkpoint_path)
        except Exception as e:
            logger.error("TFT load failed: %s", e)

    # -----------------------------------------------------------------
    # Sliding window management
    # -----------------------------------------------------------------

    def append_measurement(self, ue_id: str, measurement: dict) -> int:
        """Append a measurement to a UE's sliding window.

        Returns:
            Current window depth for this UE
        """
        if ue_id not in self._windows:
            self._windows[ue_id] = deque(maxlen=TFT_ENCODER_LENGTH)
            self._indication_counts[ue_id] = 0

        self._windows[ue_id].append(measurement)
        self._indication_counts[ue_id] += 1
        return len(self._windows[ue_id])

    def should_run_inference(self, ue_id: str) -> bool:
        """Check whether TFT inference should run for this UE now."""
        if not self.ready:
            return False
        depth = len(self._windows.get(ue_id, []))
        if depth < self.min_window_depth:
            return False
        count = self._indication_counts.get(ue_id, 0)
        return count % self.tft_every_n == 0

    def get_window_depth(self, ue_id: str) -> int:
        """Return current window depth for a UE."""
        return len(self._windows.get(ue_id, []))

    def get_window(self, ue_id: str) -> List[dict]:
        """Return the current sliding window as a list of dicts."""
        return list(self._windows.get(ue_id, []))

    def set_window(self, ue_id: str, window: List[dict]) -> None:
        """Restore a UE's sliding window (for SDL recovery)."""
        self._windows[ue_id] = deque(window, maxlen=TFT_ENCODER_LENGTH)
        self._indication_counts[ue_id] = len(window)

    # -----------------------------------------------------------------
    # TFT inference
    # -----------------------------------------------------------------

    def predict(self, ue_id: str) -> Optional[TFTPrediction]:
        """Run TFT inference on a UE's sliding window.

        Returns:
            TFTPrediction with 12-step forecasts and alert analysis,
            or None if inference cannot run.
        """
        window = self._windows.get(ue_id)
        if not self.ready or window is None or len(window) < self.min_window_depth:
            return None

        import torch
        from pytorch_forecasting import TimeSeriesDataSet
        from pytorch_forecasting.data import GroupNormalizer, MultiNormalizer

        t0 = time.perf_counter()

        try:
            df = pd.DataFrame(list(window))

            rename_map = {k: v for k, v in TFT_COLUMN_MAP.items() if k in df.columns}
            df.rename(columns=rename_map, inplace=True)

            if "elevation_deg" in df.columns:
                df["elevation_norm"] = df["elevation_deg"].astype(float) / 90.0
            elif "elevation_norm" not in df.columns:
                df["elevation_norm"] = 0.0

            if "distance_km" in df.columns:
                df["distance_norm"] = df["distance_km"].astype(float) / 40000.0
            elif "distance_norm" not in df.columns:
                df["distance_norm"] = 0.0

            if "sinr_ntn" in df.columns:
                df["sinr_ntn"] = df["sinr_ntn"].astype(float).clip(-30, 30)

            df["ue_id"] = str(ue_id)
            df["time_idx"] = list(range(len(df)))

            for col in ["time_sin", "time_cos"]:
                if col not in df.columns:
                    df[col] = 0.0

            for col in TFT_STATIC_REALS:
                if col not in df.columns:
                    df[col] = 0.0

            ntn_cols = [
                "sinr_ntn", "rsrp_ntn", "rsrq_ntn", "doppler_hz",
                "path_loss_ntn", "filtered_sinr", "elevation_norm", "distance_norm",
            ]
            tn_cols = ["sinr_tn", "rsrp_tn", "sinr_gap"]
            for col in ntn_cols + tn_cols + TFT_UNKNOWN_REALS:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
            df = df.fillna(0.0)
            df.replace([np.inf, -np.inf], 0.0, inplace=True)

            all_needed = (
                TFT_KNOWN_FUTURE_REALS + TFT_UNKNOWN_REALS +
                TFT_STATIC_REALS + TFT_TARGETS + ["time_idx", "ue_id"]
            )
            for col in all_needed:
                if col not in df.columns:
                    df[col] = 0.0

            encoder_len = len(df)
            last_row = df.iloc[-1]
            future_rows = []
            for step in range(1, TFT_PREDICTION_LENGTH + 1):
                frow = {
                    "ue_id": str(ue_id),
                    "time_idx": encoder_len - 1 + step,
                    "scenario_enc": float(last_row.get("scenario_enc", 0)),
                    "mobility_enc": float(last_row.get("mobility_enc", 0)),
                    "distance_norm": float(last_row.get("distance_norm", 0)),
                    "time_sin": float(last_row.get("time_sin", 0)),
                    "time_cos": float(last_row.get("time_cos", 0)),
                    "rsrp_ntn": 0.0,
                    "elevation_norm": 0.0,
                }
                for col in TFT_UNKNOWN_REALS:
                    frow[col] = 0.0
                future_rows.append(frow)

            df_full = pd.concat(
                [df[all_needed], pd.DataFrame(future_rows)], ignore_index=True
            )
            df_full["time_idx"] = list(range(len(df_full)))

            for col in df_full.columns:
                if col == "ue_id":
                    df_full[col] = df_full[col].astype(str)
                elif col == "time_idx":
                    df_full[col] = df_full[col].astype(int)
                else:
                    df_full[col] = (
                        pd.to_numeric(df_full[col], errors="coerce")
                        .fillna(0.0)
                        .astype("float32")
                    )

            available_known = [f for f in TFT_KNOWN_FUTURE_REALS if f in df_full.columns]
            available_unknown = [f for f in TFT_UNKNOWN_REALS if f in df_full.columns]
            available_static_reals = [f for f in TFT_STATIC_REALS if f in df_full.columns]

            dataset = TimeSeriesDataSet(
                df_full,
                time_idx="time_idx",
                target=TFT_TARGETS,
                group_ids=["ue_id"],
                min_encoder_length=TFT_ENCODER_LENGTH // 2,
                max_encoder_length=TFT_ENCODER_LENGTH,
                min_prediction_length=1,
                max_prediction_length=TFT_PREDICTION_LENGTH,
                static_categoricals=TFT_STATIC_CATEGORICALS,
                static_reals=available_static_reals,
                time_varying_known_reals=available_known,
                time_varying_unknown_reals=available_unknown,
                target_normalizer=MultiNormalizer([
                    GroupNormalizer(groups=["ue_id"], transformation=None, center=True),
                    GroupNormalizer(groups=["ue_id"], transformation=None, center=True),
                ]),
                add_relative_time_idx=True,
                add_target_scales=True,
                add_encoder_length=True,
                allow_missing_timesteps=True,
            )

            dataloader = dataset.to_dataloader(train=False, batch_size=1, num_workers=0)

            with torch.no_grad():
                predictions = self.model.predict(dataloader, return_x=True)

            pred_outputs = predictions.output
            if not isinstance(pred_outputs, (list, tuple)):
                pred_outputs = [pred_outputs]

            rsrp_pred = pred_outputs[0].cpu().numpy()[0]   # (12, 7)
            elev_pred = pred_outputs[1].cpu().numpy()[0]   # (12, 7)

            latency_ms = (time.perf_counter() - t0) * 1000

            rsrp_alert_step, elev_alert_step = self._detect_alerts(rsrp_pred, elev_pred)
            urgency, action = self._classify_urgency(rsrp_alert_step, elev_alert_step)

            prediction = TFTPrediction(
                ue_id=ue_id,
                timestamp=time.time(),
                latency_ms=latency_ms,
                rsrp_ntn_quantiles=rsrp_pred,
                elevation_norm_quantiles=elev_pred,
                rsrp_alert_step=rsrp_alert_step,
                elevation_alert_step=elev_alert_step,
                urgency=urgency,
                recommended_action=action,
            )

            self._total_predictions += 1
            self._total_latency_ms += latency_ms
            self._latest_predictions[ue_id] = prediction

            logger.info(
                "TFT inference: ue=%s, latency=%.1fms, urgency=%s, action=%s",
                ue_id, latency_ms, urgency, action,
            )
            return prediction

        except Exception as e:
            logger.error("TFT inference error for UE %s: %s", ue_id, e, exc_info=True)
            return None

    def _detect_alerts(
        self, rsrp_pred: np.ndarray, elev_pred: np.ndarray
    ) -> Tuple[Optional[int], Optional[int]]:
        """Detect alert steps from quantile forecasts (uses median idx=3)."""
        median_idx = 3
        rsrp_medians = rsrp_pred[:, median_idx]
        elev_medians = elev_pred[:, median_idx]

        rsrp_alert = None
        for step, val in enumerate(rsrp_medians):
            if val < self.rsrp_alert_threshold:
                rsrp_alert = step
                break

        elev_alert = None
        for step, val in enumerate(elev_medians):
            if val < self.elevation_alert_threshold:
                elev_alert = step
                break

        return rsrp_alert, elev_alert

    def _classify_urgency(
        self, rsrp_alert_step: Optional[int], elev_alert_step: Optional[int]
    ) -> Tuple[str, str]:
        """Classify urgency and recommended action from alert steps.

        Urgency levels:
        - imminent (<=2 steps) -> EXECUTE_HANDOVER
        - prepare  (3-7 steps) -> PREPARE_HANDOVER
        - normal   (8-11 or none) -> NONE
        """
        first_alert = min(
            rsrp_alert_step if rsrp_alert_step is not None else 999,
            elev_alert_step if elev_alert_step is not None else 999,
        )
        if first_alert <= 2:
            return "imminent", "EXECUTE_HANDOVER"
        elif first_alert <= 7:
            return "prepare", "PREPARE_HANDOVER"
        elif first_alert <= 11:
            return "normal", "NONE"
        else:
            return "normal", "NONE"

    # -----------------------------------------------------------------
    # Accessors
    # -----------------------------------------------------------------

    def get_latest_prediction(self, ue_id: str) -> Optional[TFTPrediction]:
        """Return the latest TFT prediction for a UE."""
        return self._latest_predictions.get(ue_id)

    def get_all_active_alerts(self) -> List[dict]:
        """Return all UEs with active proactive handover alerts."""
        alerts = []
        for ue_id, pred in self._latest_predictions.items():
            if pred.urgency in ("imminent", "prepare"):
                alerts.append(pred.to_dict())
        return alerts

    @property
    def metrics(self) -> dict:
        """Return predictor metrics."""
        avg_latency = (
            self._total_latency_ms / self._total_predictions
            if self._total_predictions > 0 else 0.0
        )
        return {
            "model_loaded": self.ready,
            "total_predictions": self._total_predictions,
            "avg_latency_ms": round(avg_latency, 2),
            "active_ues": len(self._windows),
            "ues_with_predictions": len(self._latest_predictions),
            "active_alerts": len(self.get_all_active_alerts()),
            "config": {
                "rsrp_alert_threshold_dbm": self.rsrp_alert_threshold,
                "elevation_alert_threshold": self.elevation_alert_threshold,
                "tft_every_n": self.tft_every_n,
                "min_window_depth": self.min_window_depth,
                "encoder_length": TFT_ENCODER_LENGTH,
                "prediction_length": TFT_PREDICTION_LENGTH,
                "quantiles": TFT_QUANTILES,
            },
        }

    def update_config(
        self,
        rsrp_alert_threshold: Optional[float] = None,
        elevation_alert_threshold: Optional[float] = None,
        tft_every_n: Optional[int] = None,
        min_window_depth: Optional[int] = None,
    ) -> dict:
        """Update predictor configuration (via A1 policy)."""
        if rsrp_alert_threshold is not None:
            self.rsrp_alert_threshold = rsrp_alert_threshold
        if elevation_alert_threshold is not None:
            self.elevation_alert_threshold = elevation_alert_threshold
        if tft_every_n is not None:
            self.tft_every_n = max(1, tft_every_n)
        if min_window_depth is not None:
            self.min_window_depth = max(10, min(TFT_ENCODER_LENGTH, min_window_depth))

        logger.info(
            "TFT config updated: rsrp_thresh=%.1f, elev_thresh=%.3f, every_n=%d, min_depth=%d",
            self.rsrp_alert_threshold, self.elevation_alert_threshold,
            self.tft_every_n, self.min_window_depth,
        )
        return {
            "rsrp_alert_threshold_dbm": self.rsrp_alert_threshold,
            "elevation_alert_threshold": self.elevation_alert_threshold,
            "tft_every_n": self.tft_every_n,
            "min_window_depth": self.min_window_depth,
        }
