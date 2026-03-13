"""O-RAN xApp Adapter for TN-NTN TFT proactive handover prediction.

Provides ricxappframe RMRXapp integration for:
- Receiving E2SM-KPM indications (broadband NTN measurements) via RMR
- Maintaining per-UE sliding windows (60-step history for TFT encoder)
- Running TFT inference every N indications (configurable, default 5)
- Generating proactive handover alerts when predicted RSRP drops below threshold
- Sending preemptive RIC control requests for upcoming handovers
- Handling A1 policy for forecast parameters

Resolves O-RAN certification blockers:
  C1 -- RMR integration (ricxappframe RMRXapp)
  C4 -- SDL integration (SDLStore for shared state)
  C5 -- E2 subscription lifecycle (E2SubscriptionManager)
  C6 -- Health check RMR handler
"""

import json
import logging
import time
from typing import Callable, Dict, Optional

from ricxappframe.xapp_frame import RMRXapp

from ..tft_predictor import TFTPredictor, TFTPrediction
from .e2_subscription_manager import (
    E2SubscriptionManager,
    RIC_SUB_RESP,
    RIC_SUB_FAILURE,
    RIC_SUB_DEL_RESP,
)

try:
    import mdclogpy
    logger = mdclogpy.Logger(name="xapp_adapter")
except ImportError:
    logger = logging.getLogger(__name__)

# RMR message type constants (O-RAN E2AP)
RIC_INDICATION = 12050
RIC_CONTROL_REQ = 12040
RIC_CONTROL_ACK = 12041
A1_POLICY_REQ = 20010
A1_POLICY_RESP = 20011
A1_POLICY_QUERY = 20012
RIC_HEALTH_CHECK_REQ = 100
RIC_HEALTH_CHECK_RESP = 101


class XAppAdapter:
    """O-RAN xApp adapter for TFT-based proactive handover prediction.

    RMRXapp runs in a background thread alongside FastAPI.
    Handlers registered for RIC_INDICATION, A1_POLICY_REQ,
    RIC_HEALTH_CHECK_REQ, and E2 subscription responses.

    Each RIC_INDICATION is appended to the UE's sliding window.
    Every tft_every_n indications, TFT inference runs and produces
    a 12-step proactive forecast with urgency classification.
    """

    def __init__(
        self,
        ric_url: str,
        xapp_id: str,
        tft_predictor: TFTPredictor,
        sdl_store,
        e2ap_decoder,
    ):
        self._ric_url = ric_url.rstrip("/") if ric_url else ""
        self._xapp_id = xapp_id
        self._tft = tft_predictor
        self._sdl = sdl_store
        self._e2ap_decoder = e2ap_decoder

        # RMR state
        self._rmr_xapp: Optional[RMRXapp] = None
        self._rmr_ready = False

        # E2 subscription manager
        self._sub_manager: Optional[E2SubscriptionManager] = None

        # Shared state
        self._a1_policies: Dict[str, dict] = {}
        self._indication_count = 0
        self._control_count = 0
        self._alert_count = 0
        self._last_indication_time: Optional[float] = None

        # Event loop reference for RMR thread -> async bridge
        self._event_loop = None

    async def start(self) -> None:
        """Initialize RMRXapp and register handlers."""
        import asyncio
        try:
            self._event_loop = asyncio.get_running_loop()
        except RuntimeError:
            self._event_loop = None

        self._start_rmr()

        # Restore sliding windows from SDL
        self._restore_windows()

        logger.info(
            "xApp adapter started (mode=rmr, RIC=%s, xApp-ID=%s)",
            self._ric_url, self._xapp_id,
        )

    def _start_rmr(self) -> None:
        """Initialize ricxappframe RMRXapp with message handlers."""
        self._rmr_xapp = RMRXapp(
            default_handler=self._rmr_default_handler,
            rmr_port=4560,
            rmr_wait_for_ready=False,
            use_fake_sdl=False,
        )

        # Register typed handlers
        self._rmr_xapp.register_callback(self._rmr_indication_handler, RIC_INDICATION)
        self._rmr_xapp.register_callback(self._rmr_a1_policy_handler, A1_POLICY_REQ)
        self._rmr_xapp.register_callback(self._rmr_health_check_handler, RIC_HEALTH_CHECK_REQ)
        self._rmr_xapp.register_callback(self._rmr_sub_response_handler, RIC_SUB_RESP)
        self._rmr_xapp.register_callback(self._rmr_sub_failure_handler, RIC_SUB_FAILURE)
        self._rmr_xapp.register_callback(self._rmr_sub_del_response_handler, RIC_SUB_DEL_RESP)

        # Run in background thread (non-blocking)
        self._rmr_xapp.run(thread=True)
        self._rmr_ready = True

        # Initialize E2 subscription manager
        self._sub_manager = E2SubscriptionManager(
            rmr_xapp=self._rmr_xapp,
            e2ap_decoder=self._e2ap_decoder,
            sdl_store=self._sdl,
        )
        self._sub_manager.restore_from_sdl()

        logger.info(
            "RMRXapp started (handlers: RIC_INDICATION=%d, A1_POLICY_REQ=%d, "
            "RIC_HEALTH_CHECK_REQ=%d)",
            RIC_INDICATION, A1_POLICY_REQ, RIC_HEALTH_CHECK_REQ,
        )

    def _restore_windows(self) -> None:
        """Restore per-UE sliding windows from SDL after restart."""
        try:
            windows = self._sdl.get_all_windows()
            restored = 0
            for ue_id, window_data in windows.items():
                self._tft.set_window(ue_id, window_data)
                restored += 1
            if restored:
                logger.info("Restored %d UE sliding windows from SDL", restored)
        except Exception as e:
            logger.warning("Failed to restore windows from SDL: %s", e)

    async def stop(self) -> None:
        """Stop RMR and unsubscribe from E2 nodes."""
        # Persist current windows to SDL before shutdown
        self._persist_windows()

        if self._sub_manager:
            self._sub_manager.unsubscribe_all()

        if self._rmr_xapp and self._rmr_ready:
            try:
                self._rmr_xapp.stop()
                self._rmr_ready = False
                logger.info("RMRXapp stopped")
            except Exception as e:
                logger.warning("RMRXapp stop failed: %s", e)

        logger.info("xApp adapter stopped")

    def _persist_windows(self) -> None:
        """Persist all sliding windows to SDL for recovery."""
        try:
            for ue_id in list(self._tft._windows.keys()):
                window = self._tft.get_window(ue_id)
                if window:
                    self._sdl.store_window(ue_id, window)
            logger.info("Persisted %d UE windows to SDL", len(self._tft._windows))
        except Exception as e:
            logger.warning("Failed to persist windows to SDL: %s", e)

    @property
    def subscription_manager(self) -> Optional[E2SubscriptionManager]:
        return self._sub_manager

    # -----------------------------------------------------------------
    # RMR message handlers (called by ricxappframe in background thread)
    # -----------------------------------------------------------------

    def _rmr_indication_handler(self, rmr_xapp, summary, sbuf) -> None:
        """Handle RIC_INDICATION (12050) message via RMR."""
        try:
            payload = summary.get("payload", b"")
            logger.debug("RMR indication received (size=%d)", len(payload))

            # Decode E2SM-KPM indication
            try:
                decoded = self._e2ap_decoder.decode_ric_indication(payload)
            except Exception as e:
                logger.warning("E2AP decode failed, trying JSON: %s", e)
                decoded = self._try_json_decode(payload)

            if decoded:
                import asyncio
                try:
                    loop = self._event_loop
                    if loop is not None and loop.is_running():
                        future = asyncio.run_coroutine_threadsafe(
                            self.handle_indication(decoded), loop
                        )
                        future.result(timeout=30)
                    else:
                        asyncio.run(self.handle_indication(decoded))
                except Exception as e:
                    logger.error("Indication processing failed: %s", e)

        except Exception as e:
            logger.error("RMR indication handler error: %s", e)
        finally:
            rmr_xapp.rmr_free(sbuf)

    def _rmr_a1_policy_handler(self, rmr_xapp, summary, sbuf) -> None:
        """Handle A1_POLICY_REQ (20010) message via RMR."""
        try:
            payload = summary.get("payload", b"")
            try:
                policy = json.loads(payload.decode("utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError) as e:
                logger.warning("A1 policy decode failed: %s", e)
                return

            result = self.handle_a1_policy(policy)
            self._sdl.store_policy(policy.get("policyId", "unknown"), result)

            try:
                resp_payload = json.dumps(result).encode("utf-8")
                success = rmr_xapp.rmr_rts(
                    sbuf, new_mtype=A1_POLICY_RESP,
                    new_payload=resp_payload, retries=3,
                )
                if success:
                    logger.info("A1 policy response sent")
                else:
                    logger.warning("A1 policy response send failed")
            except Exception as e:
                logger.warning("Failed to send A1 policy response: %s", e)

        except Exception as e:
            logger.error("RMR A1 policy handler error: %s", e)
        finally:
            rmr_xapp.rmr_free(sbuf)

    def _rmr_health_check_handler(self, rmr_xapp, summary, sbuf) -> None:
        """Handle RIC_HEALTH_CHECK_REQ (100) via RMR."""
        try:
            health = {
                "xapp_id": self._xapp_id,
                "status": "healthy",
                "rmr_connected": self._rmr_ready,
                "indication_count": self._indication_count,
                "tft_predictions": self._tft._total_predictions,
                "active_alerts": len(self._tft.get_all_active_alerts()),
                "active_subscriptions": (
                    len(self._sub_manager.active_subscriptions)
                    if self._sub_manager else 0
                ),
                "timestamp": time.time(),
            }
            resp_payload = json.dumps(health).encode("utf-8")
            success = rmr_xapp.rmr_rts(
                sbuf, new_mtype=RIC_HEALTH_CHECK_RESP,
                new_payload=resp_payload, retries=3,
            )
            if not success:
                logger.warning("Health check response send failed")
        except Exception as e:
            logger.error("RMR health check handler error: %s", e)
        finally:
            rmr_xapp.rmr_free(sbuf)

    def _rmr_sub_response_handler(self, rmr_xapp, summary, sbuf) -> None:
        try:
            payload = summary.get("payload", b"")
            if self._sub_manager:
                self._sub_manager.handle_sub_response(payload)
        except Exception as e:
            logger.error("RMR subscription response handler error: %s", e)
        finally:
            rmr_xapp.rmr_free(sbuf)

    def _rmr_sub_failure_handler(self, rmr_xapp, summary, sbuf) -> None:
        try:
            payload = summary.get("payload", b"")
            if self._sub_manager:
                self._sub_manager.handle_sub_failure(payload)
        except Exception as e:
            logger.error("RMR subscription failure handler error: %s", e)
        finally:
            rmr_xapp.rmr_free(sbuf)

    def _rmr_sub_del_response_handler(self, rmr_xapp, summary, sbuf) -> None:
        try:
            payload = summary.get("payload", b"")
            if self._sub_manager:
                self._sub_manager.handle_sub_del_response(payload)
        except Exception as e:
            logger.error("RMR subscription delete response handler error: %s", e)
        finally:
            rmr_xapp.rmr_free(sbuf)

    def _rmr_default_handler(self, rmr_xapp, summary, sbuf) -> None:
        mtype = summary.get("message type", -1)
        logger.debug("Unhandled RMR message type: %d", mtype)
        rmr_xapp.rmr_free(sbuf)

    def _try_json_decode(self, payload: bytes) -> Optional[dict]:
        try:
            return json.loads(payload.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            return None

    # -----------------------------------------------------------------
    # Core indication handler
    # -----------------------------------------------------------------

    async def handle_indication(self, indication: dict) -> dict:
        """Handle an E2SM-KPM indication.

        1. Extract measurement data from the indication
        2. Append to the UE's sliding window
        3. If tft_every_n threshold reached, run TFT inference
        4. If forecast shows degradation, generate proactive alert
        5. If urgency is imminent, send preemptive control request
        """
        self._indication_count += 1
        self._last_indication_time = time.time()

        ue_id = indication.get("ueId", "unknown")
        cell_id = indication.get("cellGlobalId")
        meas_data = indication.get("measData", {})
        timestamp = indication.get("timestamp", time.time())

        # Build measurement dict for sliding window
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

        # Append to sliding window
        depth = self._tft.append_measurement(ue_id, measurement)

        response = {
            "ueId": ue_id,
            "cellGlobalId": cell_id,
            "window_depth": depth,
            "action": "NONE",
            "forecast": None,
        }

        # Check if TFT inference should run
        if self._tft.should_run_inference(ue_id):
            prediction = self._tft.predict(ue_id)
            if prediction is not None:
                forecast_dict = prediction.to_dict()
                response["forecast"] = forecast_dict
                response["action"] = prediction.recommended_action
                response["urgency"] = prediction.urgency

                # Store forecast in SDL
                self._sdl.store_forecast(ue_id, forecast_dict)

                # Handle proactive alerts
                if prediction.urgency in ("imminent", "prepare"):
                    self._alert_count += 1
                    alert_data = {
                        "ue_id": ue_id,
                        "cell_id": cell_id,
                        "urgency": prediction.urgency,
                        "action": prediction.recommended_action,
                        "rsrp_alert_step": prediction.rsrp_alert_step,
                        "elevation_alert_step": prediction.elevation_alert_step,
                        "timestamp": time.time(),
                    }
                    self._sdl.store_alert(ue_id, alert_data)

                    logger.info(
                        "Proactive alert: ue=%s, urgency=%s, action=%s",
                        ue_id, prediction.urgency, prediction.recommended_action,
                    )

                    # Send preemptive control request for imminent handovers
                    if prediction.urgency == "imminent" and cell_id:
                        self.send_control_request(cell_id, {
                            "type": "PROACTIVE_HANDOVER",
                            "ueId": ue_id,
                            "urgency": prediction.urgency,
                            "rsrp_alert_step": prediction.rsrp_alert_step,
                            "elevation_alert_step": prediction.elevation_alert_step,
                            "reason": "tft_proactive_prediction",
                        })

                # Persist window periodically
                if self._indication_count % 50 == 0:
                    window = self._tft.get_window(ue_id)
                    if window:
                        self._sdl.store_window(ue_id, window)

        # Store prediction result in SDL
        self._sdl.store_prediction(ue_id, response)

        return response

    def handle_a1_policy(self, policy: dict) -> dict:
        """Handle an A1 policy update from the RIC.

        Supported policy types:
        - FORECAST_CONFIG: Update TFT forecast parameters
          (rsrp_alert_threshold_dbm, elevation_alert_threshold,
           tft_every_n, min_window_depth)
        """
        policy_id = policy.get("policyId", "unknown")
        policy_type = policy.get("policyType", "")
        policy_data = policy.get("policyData", {})

        self._a1_policies[policy_id] = {
            "type": policy_type,
            "data": policy_data,
            "applied_at": time.time(),
        }

        if policy_type == "FORECAST_CONFIG":
            updated = self._tft.update_config(
                rsrp_alert_threshold=policy_data.get("rsrp_alert_threshold_dbm"),
                elevation_alert_threshold=policy_data.get("elevation_alert_threshold"),
                tft_every_n=policy_data.get("tft_every_n"),
                min_window_depth=policy_data.get("min_window_depth"),
            )
            logger.info("A1 policy %s: forecast config updated: %s", policy_id, updated)
            result = {
                "policyId": policy_id,
                "status": "applied",
                "config": updated,
            }
        else:
            logger.info("A1 policy %s: unhandled type %s", policy_id, policy_type)
            result = {
                "policyId": policy_id,
                "status": "rejected",
                "reason": f"Unknown policy type: {policy_type}. Supported: FORECAST_CONFIG",
            }

        self._sdl.store_policy(policy_id, result)
        return result

    def send_control_request(self, cell_id: str, action: dict) -> dict:
        """Send E2SM-RC control request via RMR for proactive handover."""
        self._control_count += 1

        payload = self._e2ap_decoder.encode_ric_control(cell_id, action)

        if payload is None:
            control_body = {
                "xappInstanceId": self._xapp_id,
                "cellGlobalId": cell_id,
                "controlAction": action,
                "timestamp": time.time(),
            }
            payload = json.dumps(control_body).encode("utf-8")

        try:
            success = self._rmr_xapp.rmr_send(
                payload, mtype=RIC_CONTROL_REQ, retries=3,
            )
            if success:
                logger.info(
                    "Proactive control request sent: cell=%s, action=%s",
                    cell_id, action.get("type", "unknown"),
                )
                return {"status": "sent", "mode": "rmr", "cell_id": cell_id}
            else:
                logger.warning("RMR control request send failed for cell %s", cell_id)
                return {"status": "error", "detail": "RMR send failed"}
        except Exception as e:
            logger.error("RMR control request error: %s", e)
            return {"status": "error", "detail": str(e)}

    # -----------------------------------------------------------------
    # Status
    # -----------------------------------------------------------------

    @property
    def status(self) -> dict:
        return {
            "xapp_id": self._xapp_id,
            "xapp_type": "tn-ntn-tft-proactive",
            "ric_url": self._ric_url,
            "mode": "rmr",
            "rmr_connected": self._rmr_ready,
            "indication_count": self._indication_count,
            "control_count": self._control_count,
            "alert_count": self._alert_count,
            "last_indication_time": self._last_indication_time,
            "active_policies": len(self._a1_policies),
            "tft": self._tft.metrics,
            "sdl_backend": self._sdl.status["backend"],
            "e2ap_decoder_ready": self._e2ap_decoder.ready,
            "subscriptions": self._sub_manager.status if self._sub_manager else None,
        }
