"""SDL (Shared Data Layer) Store for TN-NTN TFT xApp state persistence.

Wraps ricsdl for Redis-backed shared state in the O-RAN Near-RT RIC.
Requires DBAAS_SERVICE_HOST environment variable pointing to Redis.

Extended for TFT xApp:
- Per-UE sliding windows (last 60 measurements)
- Latest forecasts with quantiles
- Active alerts with TTL
- Window recovery after restart

Serialization: msgpack (compact, fast -- standard in O-RAN SC xApps).
"""

import logging
import time
from typing import Any, Dict, List, Optional

import msgpack
from ricsdl.syncstorage import SyncStorage

try:
    import mdclogpy
    logger = mdclogpy.Logger(name="sdl_store")
except ImportError:
    logger = logging.getLogger(__name__)

# Alert TTL: 5 minutes (alerts auto-expire if not refreshed)
ALERT_TTL_SECONDS = 300


def _serialize(data) -> bytes:
    return msgpack.packb(data, use_bin_type=True)


def _deserialize(raw: bytes):
    return msgpack.unpackb(raw, raw=False)


class SDLStore:
    """Shared Data Layer store backed by Redis via ricsdl.

    Requires DBAAS_SERVICE_HOST to be set (O-RAN DBAAS Redis service).
    """

    def __init__(self, namespace: str = "tn-ntn-tft-xapp"):
        import os
        if not os.environ.get("DBAAS_SERVICE_HOST"):
            raise RuntimeError(
                "DBAAS_SERVICE_HOST must be set for SDL (Redis). "
                "Set it to the DBAAS service address (e.g., 'dbaas' in RIC, "
                "'localhost' for local dev with Redis)."
            )

        self._namespace = namespace
        self._sdl = SyncStorage()
        self._cache: Dict[str, bytes] = {}

        logger.info("SDL store initialized (backend=redis, namespace=%s)", self._namespace)

    def _set(self, key: str, value) -> None:
        raw = _serialize(value)
        self._cache[key] = raw
        self._sdl.set(self._namespace, {key: raw})

    def _get(self, key: str) -> Optional[Any]:
        try:
            result = self._sdl.get(self._namespace, {key})
            raw = result.get(key)
        except Exception as e:
            logger.warning("SDL get failed for key=%s, using cache: %s", key, e)
            raw = self._cache.get(key)
        if raw is None:
            return None
        return _deserialize(raw)

    def _delete(self, key: str) -> None:
        self._sdl.remove(self._namespace, {key})
        self._cache.pop(key, None)

    # -----------------------------------------------------------------
    # Prediction storage (per-UE)
    # -----------------------------------------------------------------

    def store_prediction(self, ue_id: str, prediction: dict) -> None:
        data = {**prediction, "_stored_at": time.time()}
        self._set(f"pred:{ue_id}", data)

    def get_prediction(self, ue_id: str) -> Optional[dict]:
        return self._get(f"pred:{ue_id}")

    # -----------------------------------------------------------------
    # Per-UE sliding windows (for TFT encoder)
    # -----------------------------------------------------------------

    def store_window(self, ue_id: str, window: List[dict]) -> None:
        """Store a UE's sliding window (list of measurement dicts)."""
        self._set(f"win:{ue_id}", {"window": window, "_stored_at": time.time()})

    def get_window(self, ue_id: str) -> Optional[List[dict]]:
        """Retrieve a UE's sliding window."""
        data = self._get(f"win:{ue_id}")
        if data is None:
            return None
        return data.get("window", [])

    def get_all_windows(self) -> Dict[str, List[dict]]:
        """Retrieve all stored sliding windows (for restart recovery)."""
        try:
            keys = self._sdl.find_keys(self._namespace, "win:*")
            if not keys:
                return {}
            result = self._sdl.get(self._namespace, set(keys))
            windows = {}
            for key, raw in result.items():
                if raw is not None:
                    data = _deserialize(raw)
                    ue_id = key.replace("win:", "", 1)
                    windows[ue_id] = data.get("window", [])
            return windows
        except Exception as e:
            logger.warning("Failed to list windows: %s", e)
            return {}

    # -----------------------------------------------------------------
    # TFT forecast storage (per-UE)
    # -----------------------------------------------------------------

    def store_forecast(self, ue_id: str, forecast: dict) -> None:
        """Store the latest TFT forecast for a UE."""
        data = {**forecast, "_stored_at": time.time()}
        self._set(f"fc:{ue_id}", data)

    def get_forecast(self, ue_id: str) -> Optional[dict]:
        """Retrieve the latest TFT forecast for a UE."""
        return self._get(f"fc:{ue_id}")

    # -----------------------------------------------------------------
    # Proactive alert storage (per-UE, with effective TTL)
    # -----------------------------------------------------------------

    def store_alert(self, ue_id: str, alert: dict) -> None:
        """Store an active proactive handover alert for a UE."""
        data = {**alert, "_stored_at": time.time(), "_ttl": ALERT_TTL_SECONDS}
        self._set(f"alert:{ue_id}", data)

    def get_alert(self, ue_id: str) -> Optional[dict]:
        """Retrieve an active alert, respecting TTL."""
        data = self._get(f"alert:{ue_id}")
        if data is None:
            return None
        stored_at = data.get("_stored_at", 0)
        ttl = data.get("_ttl", ALERT_TTL_SECONDS)
        if time.time() - stored_at > ttl:
            self._delete(f"alert:{ue_id}")
            return None
        return data

    def get_all_alerts(self) -> List[dict]:
        """Retrieve all active alerts (expired ones filtered out)."""
        try:
            keys = self._sdl.find_keys(self._namespace, "alert:*")
            if not keys:
                return []
            result = self._sdl.get(self._namespace, set(keys))
            alerts = []
            now = time.time()
            for key, raw in result.items():
                if raw is not None:
                    data = _deserialize(raw)
                    stored_at = data.get("_stored_at", 0)
                    ttl = data.get("_ttl", ALERT_TTL_SECONDS)
                    if now - stored_at <= ttl:
                        alerts.append(data)
                    else:
                        self._delete(key)
            return alerts
        except Exception as e:
            logger.warning("Failed to list alerts: %s", e)
            return []

    # -----------------------------------------------------------------
    # Cell metrics
    # -----------------------------------------------------------------

    def store_cell_metrics(self, cell_id: str, metrics: dict) -> None:
        data = {**metrics, "_stored_at": time.time()}
        self._set(f"cell:{cell_id}", data)

    def get_cell_metrics(self, cell_id: str) -> Optional[dict]:
        return self._get(f"cell:{cell_id}")

    # -----------------------------------------------------------------
    # A1 policy cache
    # -----------------------------------------------------------------

    def store_policy(self, policy_id: str, policy: dict) -> None:
        data = {**policy, "_stored_at": time.time()}
        self._set(f"a1:{policy_id}", data)

    def get_policy(self, policy_id: str) -> Optional[dict]:
        return self._get(f"a1:{policy_id}")

    # -----------------------------------------------------------------
    # E2 subscription persistence
    # -----------------------------------------------------------------

    def store_subscription(self, subscription_id: str, subscription: dict) -> None:
        data = {**subscription, "_stored_at": time.time()}
        self._set(f"sub:{subscription_id}", data)

    def get_subscription(self, subscription_id: str) -> Optional[dict]:
        return self._get(f"sub:{subscription_id}")

    def get_all_subscriptions(self) -> List[dict]:
        try:
            keys = self._sdl.find_keys(self._namespace, "sub:*")
            if not keys:
                return []
            result = self._sdl.get(self._namespace, set(keys))
            return [_deserialize(v) for v in result.values() if v is not None]
        except Exception as e:
            logger.warning("Failed to list subscriptions: %s", e)
            return []

    def delete_subscription(self, subscription_id: str) -> None:
        self._delete(f"sub:{subscription_id}")

    # -----------------------------------------------------------------
    # Status
    # -----------------------------------------------------------------

    @property
    def status(self) -> dict:
        key_count = 0
        try:
            keys = self._sdl.find_keys(self._namespace, "*")
            key_count = len(keys)
        except Exception:
            key_count = len(self._cache)
        return {
            "backend": "redis",
            "namespace": self._namespace,
            "key_count": key_count,
        }
