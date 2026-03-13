"""E2 Subscription Lifecycle Manager for TN-NTN TFT xApp.

Manages the E2 subscription state machine per O-RAN.WG3.E2AP-v02.03:

    IDLE --subscribe()--> PENDING --handle_sub_response()--> ACTIVE
      ^                      |                                  |
      |                      | handle_sub_failure()              | unsubscribe()
      |                      v                                  v
      +-------------------- IDLE <-- handle_sub_del_response() DELETING

Supports:
- Periodic KPM reports (higher frequency for TFT window building)
- Event-triggered subscriptions for RSRP crossing alerts

Persists subscription state to SDL for recovery after xApp restart.
"""

import enum
import json
import logging
import time
from typing import Dict, List, Optional

try:
    import mdclogpy
    logger = mdclogpy.Logger(name="e2_sub_mgr")
except ImportError:
    logger = logging.getLogger(__name__)

# RMR message types for E2 subscription lifecycle
RIC_SUB_REQ = 12010
RIC_SUB_RESP = 12011
RIC_SUB_FAILURE = 12012
RIC_SUB_DEL_REQ = 12020
RIC_SUB_DEL_RESP = 12021


class SubState(str, enum.Enum):
    """E2 subscription states."""
    IDLE = "IDLE"
    PENDING = "PENDING"
    ACTIVE = "ACTIVE"
    DELETING = "DELETING"


class E2SubscriptionManager:
    """Manages E2 subscription lifecycle with SDL persistence.

    Each subscription is identified by (requestor_id, instance_id) and
    tracks its state through the E2AP subscription flow.

    Supports two subscription modes for TFT xApp:
    1. Periodic KPM: High-frequency reports for sliding window building
    2. Event-triggered: RSRP threshold crossing alerts
    """

    def __init__(self, rmr_xapp, e2ap_decoder, sdl_store):
        self._rmr_xapp = rmr_xapp
        self._decoder = e2ap_decoder
        self._sdl = sdl_store

        self._subscriptions: Dict[str, dict] = {}
        self._next_requestor_id = 1

    @staticmethod
    def _sub_id(requestor_id: int, instance_id: int) -> str:
        return f"{requestor_id}:{instance_id}"

    def restore_from_sdl(self) -> int:
        """Restore subscription state from SDL after restart."""
        stored = self._sdl.get_all_subscriptions()
        restored = 0
        for sub in stored:
            sub_id = sub.get("sub_id")
            state = sub.get("state")
            if sub_id and state == SubState.ACTIVE:
                self._subscriptions[sub_id] = sub
                restored += 1
                req_id = sub.get("requestor_id", 0)
                if req_id >= self._next_requestor_id:
                    self._next_requestor_id = req_id + 1

        if restored:
            logger.info("Restored %d active E2 subscriptions from SDL", restored)
        return restored

    def subscribe_periodic_kpm(
        self,
        ran_function_id: int = 2,
        reporting_period_ms: int = 100,
        instance_id: int = 0,
    ) -> Optional[str]:
        """Subscribe to periodic KPM reports for TFT window building.

        Higher frequency than typical xApps -- TFT needs dense time-series
        data to build 60-step encoder windows quickly.

        Args:
            ran_function_id: RAN function ID for E2SM-KPM
            reporting_period_ms: Reporting period in ms (default 100ms)
            instance_id: RIC instance ID

        Returns:
            Subscription ID string, or None if failed
        """
        # Encode periodic event trigger
        event_trigger = json.dumps({
            "eventDefinition-Format1": {
                "reportingPeriod": reporting_period_ms,
            }
        }).encode("utf-8")

        actions = [
            {
                "action_id": 0,
                "action_type": "report",
                "action_definition": None,
            }
        ]

        return self.subscribe(ran_function_id, event_trigger, actions, instance_id)

    def subscribe_event_triggered(
        self,
        ran_function_id: int = 2,
        rsrp_threshold_dbm: float = -110.0,
        instance_id: int = 1,
    ) -> Optional[str]:
        """Subscribe to event-triggered RSRP crossing alerts.

        Args:
            ran_function_id: RAN function ID for E2SM-KPM
            rsrp_threshold_dbm: RSRP threshold for crossing detection
            instance_id: RIC instance ID

        Returns:
            Subscription ID string, or None if failed
        """
        event_trigger = json.dumps({
            "eventDefinition-Format1": {
                "triggerType": "upon-change",
                "measurementType": "SS-RSRP",
                "threshold": rsrp_threshold_dbm,
            }
        }).encode("utf-8")

        actions = [
            {
                "action_id": 1,
                "action_type": "report",
                "action_definition": None,
            }
        ]

        return self.subscribe(ran_function_id, event_trigger, actions, instance_id)

    def subscribe(
        self,
        ran_function_id: int,
        event_trigger: bytes,
        actions: List[dict],
        instance_id: int = 0,
    ) -> Optional[str]:
        """Send a RICsubscriptionRequest."""
        requestor_id = self._next_requestor_id
        self._next_requestor_id += 1
        sub_id = self._sub_id(requestor_id, instance_id)

        encoded = self._decoder.encode_subscription_request(
            requestor_id=requestor_id,
            instance_id=instance_id,
            ran_function_id=ran_function_id,
            event_trigger=event_trigger,
            actions=actions,
        )

        if encoded is None:
            logger.error("Failed to encode RICsubscriptionRequest for sub %s", sub_id)
            return None

        try:
            success = self._rmr_xapp.rmr_send(encoded, mtype=RIC_SUB_REQ, retries=3)
            if not success:
                logger.error("RMR send failed for RIC_SUB_REQ (sub %s)", sub_id)
                return None
        except Exception as e:
            logger.error("RMR send error for RIC_SUB_REQ: %s", e)
            return None

        sub_record = {
            "sub_id": sub_id,
            "requestor_id": requestor_id,
            "instance_id": instance_id,
            "ran_function_id": ran_function_id,
            "state": SubState.PENDING,
            "actions_requested": [a["action_id"] for a in actions],
            "actions_admitted": [],
            "created_at": time.time(),
        }
        self._subscriptions[sub_id] = sub_record
        self._sdl.store_subscription(sub_id, sub_record)

        logger.info(
            "RIC_SUB_REQ sent (sub=%s, ran_func=%d, actions=%d) -> PENDING",
            sub_id, ran_function_id, len(actions),
        )
        return sub_id

    def handle_sub_response(self, payload: bytes) -> Optional[str]:
        """Handle RICsubscriptionResponse (12011)."""
        decoded = self._decoder.decode_subscription_response(payload)
        if "decode_error" in decoded:
            logger.warning("Failed to decode RIC_SUB_RESP: %s", decoded.get("decode_error"))
            return None

        req_id = decoded.get("ricRequestID", {})
        requestor_id = req_id.get("requestorID", 0)
        instance_id = req_id.get("instanceID", 0)
        sub_id = self._sub_id(requestor_id, instance_id)

        sub = self._subscriptions.get(sub_id)
        if not sub:
            logger.warning("RIC_SUB_RESP for unknown subscription %s", sub_id)
            return None

        admitted = decoded.get("admittedActions", [])
        not_admitted = decoded.get("notAdmittedActions", [])

        sub["state"] = SubState.ACTIVE
        sub["actions_admitted"] = admitted
        sub["activated_at"] = time.time()
        self._sdl.store_subscription(sub_id, sub)

        logger.info(
            "RIC_SUB_RESP received (sub=%s) -> ACTIVE (admitted=%d, not_admitted=%d)",
            sub_id, len(admitted), len(not_admitted),
        )
        return sub_id

    def handle_sub_failure(self, payload: bytes) -> Optional[str]:
        """Handle RICsubscriptionFailure (12012)."""
        decoded = self._decoder.decode_subscription_failure(payload)
        if "decode_error" in decoded:
            logger.warning("Failed to decode RIC_SUB_FAILURE: %s", decoded.get("decode_error"))
            return None

        req_id = decoded.get("ricRequestID", {})
        requestor_id = req_id.get("requestorID", 0)
        instance_id = req_id.get("instanceID", 0)
        sub_id = self._sub_id(requestor_id, instance_id)

        sub = self._subscriptions.get(sub_id)
        if not sub:
            logger.warning("RIC_SUB_FAILURE for unknown subscription %s", sub_id)
            return None

        cause = decoded.get("cause", "unknown")
        sub["state"] = SubState.IDLE
        sub["failure_cause"] = cause
        sub["failed_at"] = time.time()
        self._sdl.delete_subscription(sub_id)
        del self._subscriptions[sub_id]

        logger.error("RIC_SUB_FAILURE received (sub=%s, cause=%s) -> IDLE", sub_id, cause)
        return sub_id

    def unsubscribe(self, sub_id: str) -> bool:
        """Send a RICsubscriptionDeleteRequest."""
        sub = self._subscriptions.get(sub_id)
        if not sub:
            logger.warning("Cannot unsubscribe: unknown subscription %s", sub_id)
            return False

        if sub["state"] not in (SubState.ACTIVE, SubState.PENDING):
            logger.warning("Cannot unsubscribe %s: invalid state %s", sub_id, sub["state"])
            return False

        del_payload = json.dumps({
            "ricRequestID": {
                "ricRequestorID": sub["requestor_id"],
                "ricInstanceID": sub["instance_id"],
            },
            "ranFunctionID": sub["ran_function_id"],
        }).encode("utf-8")

        try:
            success = self._rmr_xapp.rmr_send(del_payload, mtype=RIC_SUB_DEL_REQ, retries=3)
            if not success:
                logger.error("RMR send failed for RIC_SUB_DEL_REQ (sub %s)", sub_id)
                return False
        except Exception as e:
            logger.error("RMR send error for RIC_SUB_DEL_REQ: %s", e)
            return False

        sub["state"] = SubState.DELETING
        self._sdl.store_subscription(sub_id, sub)
        logger.info("RIC_SUB_DEL_REQ sent (sub=%s) -> DELETING", sub_id)
        return True

    def handle_sub_del_response(self, payload: bytes) -> Optional[str]:
        """Handle RICsubscriptionDeleteResponse (12021)."""
        try:
            data = json.loads(payload.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            if self._decoder._e2ap_codec:
                try:
                    decoded = self._decoder._e2ap_codec.decode(
                        "RICsubscriptionDeleteResponse", payload
                    )
                    ies = decoded.get("protocolIEs", [])
                    data = {}
                    for ie in ies:
                        value = ie.get("value")
                        if isinstance(value, tuple) and ie.get("id") == 29:
                            data["ricRequestID"] = {
                                "ricRequestorID": value[1].get("ricRequestorID"),
                                "ricInstanceID": value[1].get("ricInstanceID"),
                            }
                except Exception:
                    logger.warning("Failed to decode RIC_SUB_DEL_RESP")
                    return None
            else:
                logger.warning("Failed to decode RIC_SUB_DEL_RESP payload")
                return None

        req_id = data.get("ricRequestID", {})
        requestor_id = req_id.get("ricRequestorID", 0)
        instance_id = req_id.get("ricInstanceID", 0)
        sub_id = self._sub_id(requestor_id, instance_id)

        sub = self._subscriptions.get(sub_id)
        if not sub:
            logger.warning("RIC_SUB_DEL_RESP for unknown subscription %s", sub_id)
            return None

        self._sdl.delete_subscription(sub_id)
        del self._subscriptions[sub_id]
        logger.info("RIC_SUB_DEL_RESP received (sub=%s) -> IDLE (deleted)", sub_id)
        return sub_id

    def unsubscribe_all(self) -> int:
        """Unsubscribe from all active subscriptions."""
        active_subs = [
            sub_id for sub_id, sub in self._subscriptions.items()
            if sub["state"] == SubState.ACTIVE
        ]
        count = 0
        for sub_id in active_subs:
            if self.unsubscribe(sub_id):
                count += 1
        if count:
            logger.info("Sent %d RIC_SUB_DEL_REQ (shutting down)", count)
        return count

    @property
    def active_subscriptions(self) -> List[dict]:
        return [sub for sub in self._subscriptions.values() if sub["state"] == SubState.ACTIVE]

    @property
    def status(self) -> dict:
        states = {}
        for sub in self._subscriptions.values():
            state = sub["state"]
            states[state] = states.get(state, 0) + 1
        return {
            "total_subscriptions": len(self._subscriptions),
            "states": states,
            "next_requestor_id": self._next_requestor_id,
        }
