"""O-RAN xApp Router -- FastAPI endpoints for TN-NTN TFT proactive handover.

Provides:
- E2SM-KPM indication ingestion (POST /xapp/v1/indication)
- A1 policy updates (POST /xapp/v1/a1-policy)
- xApp status & health (GET /xapp/v1/status)
- Per-UE 12-step forecast with quantiles (GET /xapp/v1/forecast/{ue_id})
- Active proactive handover alerts (GET /xapp/v1/alerts)
- Per-UE sliding window depth (GET /xapp/v1/windows/{ue_id})
"""

import logging
from typing import Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

try:
    import mdclogpy
    logger = mdclogpy.Logger(name="xapp_router")
except ImportError:
    logger = logging.getLogger(__name__)

router = APIRouter(prefix="/xapp/v1", tags=["O-RAN TFT xApp"])


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------

class E2Indication(BaseModel):
    """E2SM-KPM indication from Near-RT RIC with broadband NTN measurements."""
    ranFunctionId: int = Field(default=1, description="RAN function ID")
    cellGlobalId: str = Field(default="", description="Cell global identifier")
    ueId: str = Field(description="UE identifier")
    measData: dict = Field(
        description="Measurement data: rsrp_ntn, sinr_ntn, rsrq_ntn, "
                    "doppler_hz, path_loss_ntn, sinr_tn, rsrp_tn, ue_speed, "
                    "sinr_gap, filtered_sinr, elevation_deg, distance_km, "
                    "time_sin, time_cos, scenario_enc, mobility_enc"
    )
    timestamp: float = Field(default=0, description="Unix timestamp (0 = use server time)")


class A1Policy(BaseModel):
    """A1 policy update from Near-RT RIC."""
    policyId: str = Field(description="Policy instance ID")
    policyType: str = Field(
        description="Policy type: FORECAST_CONFIG"
    )
    policyData: dict = Field(
        description="Policy parameters: rsrp_alert_threshold_dbm, "
                    "elevation_alert_threshold, tft_every_n, min_window_depth"
    )


# ---------------------------------------------------------------------------
# Dependency injection
# ---------------------------------------------------------------------------

def _get_xapp(request: Request):
    xapp = getattr(request.app.state, "xapp_adapter", None)
    if xapp is None:
        raise HTTPException(status_code=503, detail="O-RAN xApp adapter not enabled")
    return xapp


def _get_tft(request: Request):
    tft = getattr(request.app.state, "tft_predictor", None)
    if tft is None:
        raise HTTPException(status_code=503, detail="TFT predictor not initialized")
    return tft


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post("/indication")
async def receive_indication(indication: E2Indication, xapp=Depends(_get_xapp)):
    """Receive an E2SM-KPM indication from the Near-RT RIC.

    The measurement is appended to the UE's sliding window. If the
    tft_every_n threshold is reached, TFT inference runs and returns
    a 12-step proactive forecast with urgency classification.
    """
    import time
    data = indication.model_dump()
    if data["timestamp"] == 0:
        data["timestamp"] = time.time()
    result = await xapp.handle_indication(data)
    return result


@router.post("/a1-policy")
async def receive_a1_policy(policy: A1Policy, xapp=Depends(_get_xapp)):
    """Receive an A1 policy update from the Near-RT RIC.

    Supported policy types:
    - FORECAST_CONFIG: Update TFT forecast parameters
      (rsrp_alert_threshold_dbm, elevation_alert_threshold,
       tft_every_n, min_window_depth)
    """
    result = xapp.handle_a1_policy(policy.model_dump())
    return result


@router.get("/status")
async def xapp_status(xapp=Depends(_get_xapp)):
    """Get xApp health, operational status, and TFT metrics.

    Returns RMR connectivity, indication/control/alert counts,
    TFT model status, average inference latency, and active alert count.
    """
    return xapp.status


@router.get("/forecast/{ue_id}")
async def get_forecast(ue_id: str, tft=Depends(_get_tft)):
    """Get the latest 12-step TFT forecast for a UE.

    Returns quantile predictions [0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98]
    for both rsrp_ntn and elevation_norm, plus alert analysis.
    """
    prediction = tft.get_latest_prediction(ue_id)
    if prediction is None:
        raise HTTPException(
            status_code=404,
            detail=f"No forecast available for UE {ue_id}. "
                   f"Window depth: {tft.get_window_depth(ue_id)}/{tft.min_window_depth}",
        )
    return prediction.to_dict()


@router.get("/alerts")
async def get_alerts(tft=Depends(_get_tft)):
    """Get all active proactive handover alerts.

    Returns UEs with urgency 'imminent' or 'prepare' based on
    TFT 12-step forecast analysis.
    """
    alerts = tft.get_all_active_alerts()
    return {
        "alert_count": len(alerts),
        "alerts": alerts,
    }


@router.get("/windows/{ue_id}")
async def get_window_info(ue_id: str, tft=Depends(_get_tft)):
    """Get current sliding window depth and status for a UE.

    Returns window depth, minimum required depth, target depth,
    and whether the window is ready for TFT inference.
    """
    from ..tft_predictor import TFT_ENCODER_LENGTH

    depth = tft.get_window_depth(ue_id)
    return {
        "ue_id": ue_id,
        "window_depth": depth,
        "min_depth": tft.min_window_depth,
        "target_depth": TFT_ENCODER_LENGTH,
        "ready_for_inference": depth >= tft.min_window_depth and tft.ready,
        "model_loaded": tft.ready,
    }
