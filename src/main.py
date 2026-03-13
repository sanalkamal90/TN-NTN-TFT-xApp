"""TN-NTN TFT xApp -- Main entry point.

Starts FastAPI server with O-RAN xApp adapter, TFT predictor,
SDL store, and E2AP decoder.
"""

import asyncio
import logging
import os
import sys
import time

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

try:
    import mdclogpy
    logger = mdclogpy.Logger(name="tft_xapp_main")
except ImportError:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    logger = logging.getLogger(__name__)


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""

    app = FastAPI(
        title="TN-NTN TFT xApp",
        description=(
            "O-RAN Near-RT RIC xApp for proactive TN-NTN broadband handover "
            "prediction using Temporal Fusion Transformer (TFT). "
            "12-step lookahead with 7 uncertainty quantiles."
        ),
        version="1.0.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Health endpoint (liveness probe)
    @app.get("/health")
    async def health():
        return {
            "status": "healthy",
            "xapp": "tn-ntn-tft-xapp",
            "version": "1.0.0",
            "timestamp": time.time(),
        }

    # Include xApp router
    from .adapters.xapp_router import router as xapp_router
    app.include_router(xapp_router)

    @app.on_event("startup")
    async def startup():
        """Initialize TFT predictor, SDL, E2AP decoder, and xApp adapter."""
        logger.info("Starting TN-NTN TFT xApp...")

        # Configuration from environment
        checkpoint_path = os.environ.get(
            "TFT_CHECKPOINT_PATH", "/app/models/tft_best.ckpt"
        )
        device = os.environ.get("TFT_DEVICE", "cpu")
        ric_url = os.environ.get("RIC_URL", "")
        xapp_id = os.environ.get("XAPP_ID", "tn-ntn-tft-xapp")
        tft_every_n = int(os.environ.get("TFT_EVERY_N", "5"))
        min_window_depth = int(os.environ.get("MIN_WINDOW_DEPTH", "30"))
        rsrp_threshold = float(os.environ.get("RSRP_ALERT_THRESHOLD", "-110.0"))
        elev_threshold = float(os.environ.get("ELEVATION_ALERT_THRESHOLD", "0.15"))

        # 1. Initialize TFT predictor
        from .tft_predictor import TFTPredictor
        tft = TFTPredictor(
            checkpoint_path=checkpoint_path,
            device=device,
            rsrp_alert_threshold=rsrp_threshold,
            elevation_alert_threshold=elev_threshold,
            tft_every_n=tft_every_n,
            min_window_depth=min_window_depth,
        )
        app.state.tft_predictor = tft

        # 2. Initialize SDL store
        sdl = None
        try:
            from .adapters.sdl_store import SDLStore
            sdl = SDLStore(namespace="tn-ntn-tft-xapp")
            logger.info("SDL store initialized")
        except Exception as e:
            logger.warning("SDL store initialization failed (non-fatal): %s", e)
            # Provide a no-op SDL for development without Redis
            sdl = _NoOpSDL()

        # 3. Initialize E2AP decoder
        e2ap = None
        try:
            from .adapters.e2ap_decoder import E2APDecoder
            e2ap = E2APDecoder()
            logger.info("E2AP decoder initialized")
        except Exception as e:
            logger.warning("E2AP decoder initialization failed (non-fatal): %s", e)
            e2ap = _NoOpE2APDecoder()

        # 4. Initialize xApp adapter
        try:
            from .adapters.xapp_adapter import XAppAdapter
            xapp = XAppAdapter(
                ric_url=ric_url,
                xapp_id=xapp_id,
                tft_predictor=tft,
                sdl_store=sdl,
                e2ap_decoder=e2ap,
            )
            await xapp.start()
            app.state.xapp_adapter = xapp
            logger.info("xApp adapter started")
        except Exception as e:
            logger.warning(
                "xApp adapter startup failed (running in standalone mode): %s", e
            )
            # Create a standalone adapter that works without RMR
            app.state.xapp_adapter = _StandaloneAdapter(tft, sdl, e2ap, xapp_id)

        logger.info("TN-NTN TFT xApp startup complete")

    @app.on_event("shutdown")
    async def shutdown():
        """Gracefully stop xApp adapter."""
        logger.info("Shutting down TN-NTN TFT xApp...")
        xapp = getattr(app.state, "xapp_adapter", None)
        if xapp and hasattr(xapp, "stop"):
            await xapp.stop()
        logger.info("TN-NTN TFT xApp shutdown complete")

    return app


class _NoOpSDL:
    """No-op SDL for development without Redis."""

    def store_prediction(self, *a, **kw): pass
    def get_prediction(self, *a, **kw): return None
    def store_window(self, *a, **kw): pass
    def get_window(self, *a, **kw): return None
    def get_all_windows(self, **kw): return {}
    def store_forecast(self, *a, **kw): pass
    def get_forecast(self, *a, **kw): return None
    def store_alert(self, *a, **kw): pass
    def get_alert(self, *a, **kw): return None
    def get_all_alerts(self, **kw): return []
    def store_cell_metrics(self, *a, **kw): pass
    def get_cell_metrics(self, *a, **kw): return None
    def store_policy(self, *a, **kw): pass
    def get_policy(self, *a, **kw): return None
    def store_subscription(self, *a, **kw): pass
    def get_subscription(self, *a, **kw): return None
    def get_all_subscriptions(self, **kw): return []
    def delete_subscription(self, *a, **kw): pass

    @property
    def status(self):
        return {"backend": "noop", "namespace": "dev", "key_count": 0}


class _NoOpE2APDecoder:
    """No-op E2AP decoder for development without ASN.1 schemas."""

    @property
    def ready(self):
        return False

    def decode_ric_indication(self, payload):
        return {"raw_hex": payload.hex() if payload else "", "decode_error": "no codec"}

    def encode_ric_control(self, cell_id, action):
        return None

    def encode_subscription_request(self, *a, **kw):
        return None

    def decode_subscription_response(self, payload):
        return {"decode_error": "no codec"}

    def decode_subscription_failure(self, payload):
        return {"decode_error": "no codec"}


class _StandaloneAdapter:
    """Standalone adapter that works without RMR for development/testing."""

    def __init__(self, tft, sdl, e2ap, xapp_id):
        self._tft = tft
        self._sdl = sdl
        self._e2ap = e2ap
        self._xapp_id = xapp_id
        self._indication_count = 0
        self._control_count = 0
        self._alert_count = 0
        self._last_indication_time = None
        self._a1_policies = {}

    async def handle_indication(self, indication: dict) -> dict:
        """Process indication through TFT predictor (standalone mode)."""
        self._indication_count += 1
        self._last_indication_time = time.time()

        ue_id = indication.get("ueId", "unknown")
        cell_id = indication.get("cellGlobalId", "")
        meas_data = indication.get("measData", {})
        timestamp = indication.get("timestamp", time.time())

        measurement = {
            "timestamp": timestamp,
            "rsrp_ntn": meas_data.get("rsrp", meas_data.get("rsrp_ntn", -100.0)),
            "sinr_ntn": meas_data.get("sinr_ntn", 0.0),
            "rsrq_ntn": meas_data.get("rsrq", meas_data.get("rsrq_ntn", 0.0)),
            "doppler_hz": meas_data.get("doppler_hz", 0.0),
            "path_loss_ntn": meas_data.get("path_loss_ntn", 0.0),
            "sinr_tn": meas_data.get("sinr_tn", 0.0),
            "rsrp_tn": meas_data.get("rsrp_tn", 0.0),
            "ue_speed": meas_data.get("ue_speed", 0.0),
            "sinr_gap": meas_data.get("sinr_gap", 0.0),
            "filtered_sinr": meas_data.get("filtered_sinr", 0.0),
            "elevation_deg": meas_data.get("elevation_deg", 0.0),
            "elevation_norm": meas_data.get("elevation_norm", 0.0),
            "distance_km": meas_data.get("distance_km", 0.0),
            "distance_norm": meas_data.get("distance_norm", 0.0),
            "time_sin": meas_data.get("time_sin", 0.0),
            "time_cos": meas_data.get("time_cos", 0.0),
            "scenario_enc": meas_data.get("scenario_enc", 0.0),
            "mobility_enc": meas_data.get("mobility_enc", 0.0),
        }

        depth = self._tft.append_measurement(ue_id, measurement)

        response = {
            "ueId": ue_id,
            "cellGlobalId": cell_id,
            "window_depth": depth,
            "action": "NONE",
            "forecast": None,
        }

        if self._tft.should_run_inference(ue_id):
            prediction = self._tft.predict(ue_id)
            if prediction is not None:
                forecast_dict = prediction.to_dict()
                response["forecast"] = forecast_dict
                response["action"] = prediction.recommended_action
                response["urgency"] = prediction.urgency

                self._sdl.store_forecast(ue_id, forecast_dict)

                if prediction.urgency in ("imminent", "prepare"):
                    self._alert_count += 1
                    self._sdl.store_alert(ue_id, {
                        "ue_id": ue_id,
                        "cell_id": cell_id,
                        "urgency": prediction.urgency,
                        "action": prediction.recommended_action,
                        "timestamp": time.time(),
                    })

        self._sdl.store_prediction(ue_id, response)
        return response

    def handle_a1_policy(self, policy: dict) -> dict:
        policy_id = policy.get("policyId", "unknown")
        policy_type = policy.get("policyType", "")
        policy_data = policy.get("policyData", {})

        self._a1_policies[policy_id] = {
            "type": policy_type, "data": policy_data, "applied_at": time.time()
        }

        if policy_type == "FORECAST_CONFIG":
            updated = self._tft.update_config(
                rsrp_alert_threshold=policy_data.get("rsrp_alert_threshold_dbm"),
                elevation_alert_threshold=policy_data.get("elevation_alert_threshold"),
                tft_every_n=policy_data.get("tft_every_n"),
                min_window_depth=policy_data.get("min_window_depth"),
            )
            result = {"policyId": policy_id, "status": "applied", "config": updated}
        else:
            result = {
                "policyId": policy_id, "status": "rejected",
                "reason": f"Unknown policy type: {policy_type}",
            }

        self._sdl.store_policy(policy_id, result)
        return result

    @property
    def status(self) -> dict:
        return {
            "xapp_id": self._xapp_id,
            "xapp_type": "tn-ntn-tft-proactive",
            "mode": "standalone",
            "rmr_connected": False,
            "indication_count": self._indication_count,
            "control_count": self._control_count,
            "alert_count": self._alert_count,
            "last_indication_time": self._last_indication_time,
            "active_policies": len(self._a1_policies),
            "tft": self._tft.metrics,
            "sdl_backend": self._sdl.status.get("backend", "noop"),
            "e2ap_decoder_ready": self._e2ap.ready,
            "subscriptions": None,
        }

    async def stop(self):
        logger.info("Standalone adapter stopped")


app = create_app()


if __name__ == "__main__":
    host = os.environ.get("TFT_XAPP_HOST", "0.0.0.0")
    port = int(os.environ.get("TFT_XAPP_PORT", "8447"))

    uvicorn.run(
        "src.main:app",
        host=host,
        port=port,
        log_level="info",
        access_log=True,
    )
