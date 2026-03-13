"""E2AP / E2SM-KPM / E2SM-RC Decoder and Encoder for TN-NTN TFT xApp.

Decodes APER-encoded E2AP RICindication messages containing E2SM-KPM v3
measurement data, and encodes E2SM-RC control requests for proactive
handover. Encodes/decodes E2 subscription lifecycle messages.

Uses asn1tools with APER codec (not UPER -- E2AP uses APER).
"""

import logging
import os
import time
from typing import Dict, List, Optional, Tuple

import asn1tools

try:
    import mdclogpy
    logger = mdclogpy.Logger(name="e2ap_decoder")
except ImportError:
    logger = logging.getLogger(__name__)

# Well-known E2AP IE IDs
_IE_RIC_REQUEST_ID = 29
_IE_RAN_FUNCTION_ID = 5
_IE_RIC_ACTION_ID = 15
_IE_RIC_INDICATION_SN = 27
_IE_RIC_INDICATION_TYPE = 28
_IE_RIC_INDICATION_HEADER = 25
_IE_RIC_INDICATION_MESSAGE = 26
_IE_RIC_CONTROL_HEADER = 22
_IE_RIC_CONTROL_MESSAGE = 23
_IE_RIC_SUBSCRIPTION_DETAILS = 30
_IE_RIC_ACTION_ADMITTED_LIST = 17
_IE_RIC_ACTION_NOT_ADMITTED_LIST = 18
_IE_CAUSE = 1

# Well-known E2SM-KPM measurement names (3GPP TS 28.552)
_KPM_MEAS_NAMES = {
    "DRB.UEThpDl": "throughput_dl",
    "DRB.UEThpUl": "throughput_ul",
    "RRU.PrbUsedDl": "prb_used_dl",
    "RRU.PrbUsedUl": "prb_used_ul",
    "L3 serving SS-RSRP": "rsrp",
    "SS-RSRP": "rsrp",
    "L3 serving SS-RSRQ": "rsrq",
    "SS-RSRQ": "rsrq",
    "L3 serving SS-SINR": "sinr",
    "SS-SINR": "sinr",
}

# E2SM-RC Style Type 3 = Connected Mode Mobility Control
_RC_STYLE_TYPE_HANDOVER = 3
_RC_ACTION_ID_HANDOVER = 1
_RANP_TARGET_CELL_ID = 1
_RANP_TARGET_CELL_RSRP = 2
_RANP_HANDOVER_CAUSE = 3

# 3GPP TS 38.133 RSRP/RSRQ/SINR index conversion
def _rsrp_index_to_dbm(index: int) -> float:
    return -156.0 + index

def _rsrq_index_to_db(index: int) -> float:
    return -43.5 + 0.5 * index

def _sinr_index_to_db(index: int) -> float:
    return -23.0 + 0.5 * index


class E2APDecoder:
    """Decodes E2AP/E2SM-KPM indications and encodes E2SM-RC control requests."""

    def __init__(
        self,
        kpm_schema_path: Optional[str] = None,
        rc_schema_path: Optional[str] = None,
        e2ap_schema_path: Optional[str] = None,
    ):
        schemas_dir = os.path.join(os.path.dirname(__file__), "..", "..", "schemas")
        schemas_dir = os.path.abspath(schemas_dir)

        if kpm_schema_path is None:
            kpm_schema_path = os.path.join(schemas_dir, "e2sm_kpm_v3.asn")
        if rc_schema_path is None:
            rc_schema_path = os.path.join(schemas_dir, "e2sm_rc_v1.asn")
        if e2ap_schema_path is None:
            e2ap_schema_path = os.path.join(schemas_dir, "e2ap_v2.asn")

        for path, name in [
            (kpm_schema_path, "E2SM-KPM"),
            (rc_schema_path, "E2SM-RC"),
            (e2ap_schema_path, "E2AP"),
        ]:
            if not os.path.exists(path):
                raise FileNotFoundError(f"{name} schema not found: {path}")

        logger.info("Compiling E2AP/E2SM schemas (APER via 'per' codec)...")

        self._kpm_codec = None
        self._rc_codec = None
        self._e2ap_codec = None

        try:
            self._kpm_codec = asn1tools.compile_files(kpm_schema_path, "per")
            logger.info("E2SM-KPM v3 schema compiled")
        except Exception as e:
            logger.warning("E2SM-KPM schema compilation failed: %s", e)

        try:
            self._rc_codec = asn1tools.compile_files(rc_schema_path, "per")
            logger.info("E2SM-RC v1 schema compiled")
        except Exception as e:
            logger.warning("E2SM-RC schema compilation failed: %s", e)

        try:
            self._e2ap_codec = asn1tools.compile_files(e2ap_schema_path, "per")
            logger.info("E2AP v2 schema compiled")
        except Exception as e:
            logger.warning("E2AP schema compilation failed: %s", e)

        self._ready = any([self._kpm_codec, self._rc_codec, self._e2ap_codec])
        if self._ready:
            logger.info("E2AP decoder ready")
        else:
            logger.warning("No E2AP/E2SM schemas compiled -- decoder in passthrough mode")

    @property
    def ready(self) -> bool:
        return self._ready

    def decode_kpm_indication_header(self, header_bytes: bytes) -> dict:
        """Decode E2SM-KPM IndicationHeader."""
        if not self._kpm_codec:
            return {"raw_hex": header_bytes.hex(), "decode_error": "KPM codec not loaded"}
        try:
            decoded = self._kpm_codec.decode("E2SM-KPM-IndicationHeader", header_bytes)
            if isinstance(decoded, tuple) and decoded[0] == "indicationHeader-Format1":
                fmt1 = decoded[1]
                result = {}
                if "colletStartTime" in fmt1:
                    ts_bytes = fmt1["colletStartTime"]
                    result["collectStartTime"] = int.from_bytes(ts_bytes, "big")
                if "senderName" in fmt1:
                    result["senderName"] = fmt1["senderName"]
                if "vendorName" in fmt1:
                    result["vendorName"] = fmt1["vendorName"]
                return result
            return {"decoded": str(decoded)}
        except Exception as e:
            logger.warning("KPM indication header decode failed: %s", e)
            return {"raw_hex": header_bytes.hex(), "decode_error": str(e)}

    def decode_kpm_indication_message(self, message_bytes: bytes) -> dict:
        """Decode E2SM-KPM IndicationMessage."""
        if not self._kpm_codec:
            return {"raw_hex": message_bytes.hex(), "decode_error": "KPM codec not loaded"}
        try:
            decoded = self._kpm_codec.decode("E2SM-KPM-IndicationMessage", message_bytes)
            if isinstance(decoded, tuple) and decoded[0] == "indicationMessage-Format1":
                return self._parse_kpm_format1(decoded[1])
            elif isinstance(decoded, tuple) and decoded[0] == "indicationMessage-Format3":
                return self._parse_kpm_format3(decoded[1])
            return {"decoded": str(decoded)}
        except Exception as e:
            logger.warning("KPM indication message decode failed: %s", e)
            return {"raw_hex": message_bytes.hex(), "decode_error": str(e)}

    def _parse_kpm_format1(self, fmt1: dict) -> dict:
        result = {"format": 1, "measurements": []}
        meas_names = []
        if "measInfoList" in fmt1 and fmt1["measInfoList"]:
            for info_item in fmt1["measInfoList"]:
                meas_type = info_item.get("measType")
                if isinstance(meas_type, tuple):
                    if meas_type[0] == "measName":
                        meas_names.append(_KPM_MEAS_NAMES.get(meas_type[1], meas_type[1]))
                    elif meas_type[0] == "measID":
                        meas_names.append(f"meas_{meas_type[1]}")

        meas_data = fmt1.get("measData", [])
        for data_item in meas_data:
            record = data_item.get("measRecord", [])
            meas_values = {}
            for idx, record_item in enumerate(record):
                if isinstance(record_item, tuple):
                    choice_type, value = record_item
                    name = meas_names[idx] if idx < len(meas_names) else f"metric_{idx}"
                    if choice_type == "integer":
                        meas_values[name] = int(value)
                    elif choice_type == "real":
                        meas_values[name] = float(value)
                    elif choice_type == "noValue":
                        meas_values[name] = None
            if meas_values:
                result["measurements"].append(meas_values)

        if "granulPeriod" in fmt1 and fmt1["granulPeriod"]:
            result["granularity_period_ms"] = fmt1["granulPeriod"]
        return result

    def _parse_kpm_format3(self, fmt3: dict) -> dict:
        result = {"format": 3, "ue_reports": []}
        ue_list = fmt3.get("ueMeasReportList", [])
        for ue_item in ue_list:
            ue_id = self._extract_ue_id(ue_item.get("ueID"))
            meas_report = ue_item.get("measReport")
            if meas_report:
                parsed = self._parse_kpm_format1(meas_report)
                parsed["ueId"] = ue_id
                result["ue_reports"].append(parsed)
        return result

    def _extract_ue_id(self, ue_id) -> str:
        if ue_id is None:
            return "unknown"
        if isinstance(ue_id, tuple):
            choice_type, value = ue_id
            if choice_type == "gNB-UEID" and isinstance(value, dict):
                amf_id = value.get("amf-UE-NGAP-ID")
                if amf_id is not None:
                    return str(amf_id)
            return f"{choice_type}:{value}"
        return str(ue_id)

    def decode_ric_indication(self, payload: bytes) -> dict:
        """Decode a full RICindication payload from RMR message."""
        result = {"timestamp": time.time(), "payload_size": len(payload)}

        if self._e2ap_codec:
            try:
                e2ap_decoded = self._e2ap_codec.decode("RICindication", payload)
                ies = e2ap_decoded.get("protocolIEs", [])
                has_e2sm = any(
                    ie.get("id") in (_IE_RIC_INDICATION_HEADER, _IE_RIC_INDICATION_MESSAGE)
                    for ie in ies
                )
                if has_e2sm:
                    return self._extract_from_e2ap_indication(e2ap_decoded)
            except Exception:
                pass

        if self._kpm_codec and len(payload) > 0:
            try:
                msg_data = self.decode_kpm_indication_message(payload)
                result["measData"] = msg_data
                return result
            except Exception as e:
                logger.debug("Direct KPM decode failed: %s", e)

        result["raw_hex"] = payload.hex()
        result["decode_note"] = "Could not decode as E2AP or E2SM-KPM"
        return result

    def _extract_from_e2ap_indication(self, e2ap_decoded: dict) -> dict:
        result = {"timestamp": time.time()}
        ies = e2ap_decoded.get("protocolIEs", [])
        header_bytes = None
        message_bytes = None

        for ie in ies:
            ie_id = ie.get("id")
            value = ie.get("value")
            if isinstance(value, tuple):
                value_type, value_data = value
            else:
                continue
            if ie_id == _IE_RAN_FUNCTION_ID:
                result["ranFunctionId"] = value_data
            elif ie_id == _IE_RIC_INDICATION_HEADER:
                header_bytes = value_data
            elif ie_id == _IE_RIC_INDICATION_MESSAGE:
                message_bytes = value_data
            elif ie_id == _IE_RIC_INDICATION_TYPE:
                result["indicationType"] = str(value_data)

        if header_bytes:
            result["header"] = self.decode_kpm_indication_header(header_bytes)
        if message_bytes:
            result["measData"] = self.decode_kpm_indication_message(message_bytes)
        return result

    def encode_ric_control(self, cell_id: str, action: dict) -> Optional[bytes]:
        """Encode an E2SM-RC control request for proactive handover."""
        if not self._rc_codec:
            logger.warning("E2SM-RC codec not available -- cannot encode control")
            return None

        target_cell = action.get("targetCell", cell_id)

        try:
            header = (
                "controlHeader-Format1",
                {
                    "ric-Style-Type": _RC_STYLE_TYPE_HANDOVER,
                    "ric-ControlAction-ID": _RC_ACTION_ID_HANDOVER,
                },
            )
            header_bytes = self._rc_codec.encode("E2SM-RC-ControlHeader", header)

            message = (
                "controlMessage-Format1",
                {
                    "ranP-List": [
                        {
                            "ranParameter-ID": _RANP_TARGET_CELL_ID,
                            "ranParameter-valueType": (
                                "ranP-Choice-ElementTrue",
                                {
                                    "ranParameter-value": (
                                        "valueOctS",
                                        target_cell.encode("utf-8"),
                                    ),
                                },
                            ),
                        },
                    ],
                },
            )
            message_bytes = self._rc_codec.encode("E2SM-RC-ControlMessage", message)
            return header_bytes + message_bytes

        except Exception as e:
            logger.error("E2SM-RC control encoding failed: %s", e)
            return None

    def encode_subscription_request(
        self, requestor_id: int, instance_id: int,
        ran_function_id: int, event_trigger: bytes, actions: List[dict],
    ) -> Optional[bytes]:
        """Encode a RICsubscriptionRequest message."""
        if not self._e2ap_codec:
            logger.warning("E2AP codec not available -- cannot encode subscription request")
            return None

        try:
            action_list = []
            for act in actions:
                item = {
                    "ricActionID": act["action_id"],
                    "ricActionType": act.get("action_type", "report"),
                }
                if act.get("action_definition"):
                    item["ricActionDefinition"] = act["action_definition"]
                action_list.append(item)

            ies = [
                {
                    "id": _IE_RIC_REQUEST_ID,
                    "criticality": "reject",
                    "value": ("ricRequestID", {
                        "ricRequestorID": requestor_id,
                        "ricInstanceID": instance_id,
                    }),
                },
                {
                    "id": _IE_RAN_FUNCTION_ID,
                    "criticality": "reject",
                    "value": ("ranFunctionID", ran_function_id),
                },
                {
                    "id": _IE_RIC_SUBSCRIPTION_DETAILS,
                    "criticality": "reject",
                    "value": ("ricSubscriptionDetails", {
                        "ricEventTriggerDefinition": event_trigger,
                        "ricAction-ToBeSetup-List": action_list,
                    }),
                },
            ]

            encoded = self._e2ap_codec.encode(
                "RICsubscriptionRequest", {"protocolIEs": ies}
            )
            logger.info(
                "Encoded RICsubscriptionRequest (requestor=%d, instance=%d, ran_func=%d)",
                requestor_id, instance_id, ran_function_id,
            )
            return encoded
        except Exception as e:
            logger.error("RICsubscriptionRequest encoding failed: %s", e)
            return None

    def decode_subscription_response(self, payload: bytes) -> dict:
        if not self._e2ap_codec:
            return {"raw_hex": payload.hex(), "decode_error": "E2AP codec not loaded"}
        try:
            decoded = self._e2ap_codec.decode("RICsubscriptionResponse", payload)
            result = {}
            for ie in decoded.get("protocolIEs", []):
                ie_id = ie.get("id")
                value = ie.get("value")
                if not isinstance(value, tuple):
                    continue
                _, value_data = value
                if ie_id == _IE_RIC_REQUEST_ID:
                    result["ricRequestID"] = {
                        "requestorID": value_data.get("ricRequestorID"),
                        "instanceID": value_data.get("ricInstanceID"),
                    }
                elif ie_id == _IE_RAN_FUNCTION_ID:
                    result["ranFunctionID"] = value_data
                elif ie_id == _IE_RIC_ACTION_ADMITTED_LIST:
                    result["admittedActions"] = [item.get("ricActionID") for item in value_data]
                elif ie_id == _IE_RIC_ACTION_NOT_ADMITTED_LIST:
                    result["notAdmittedActions"] = [
                        {"actionID": item.get("ricActionID"), "cause": str(item.get("cause", "unknown"))}
                        for item in value_data
                    ]
            return result
        except Exception as e:
            logger.warning("RICsubscriptionResponse decode failed: %s", e)
            return {"raw_hex": payload.hex(), "decode_error": str(e)}

    def decode_subscription_failure(self, payload: bytes) -> dict:
        if not self._e2ap_codec:
            return {"raw_hex": payload.hex(), "decode_error": "E2AP codec not loaded"}
        try:
            decoded = self._e2ap_codec.decode("RICsubscriptionFailure", payload)
            result = {}
            for ie in decoded.get("protocolIEs", []):
                ie_id = ie.get("id")
                value = ie.get("value")
                if not isinstance(value, tuple):
                    continue
                _, value_data = value
                if ie_id == _IE_RIC_REQUEST_ID:
                    result["ricRequestID"] = {
                        "requestorID": value_data.get("ricRequestorID"),
                        "instanceID": value_data.get("ricInstanceID"),
                    }
                elif ie_id == _IE_RAN_FUNCTION_ID:
                    result["ranFunctionID"] = value_data
                elif ie_id == _IE_CAUSE:
                    result["cause"] = str(value_data)
            return result
        except Exception as e:
            logger.warning("RICsubscriptionFailure decode failed: %s", e)
            return {"raw_hex": payload.hex(), "decode_error": str(e)}

    @property
    def status(self) -> dict:
        return {
            "ready": self._ready,
            "kpm_codec": self._kpm_codec is not None,
            "rc_codec": self._rc_codec is not None,
            "e2ap_codec": self._e2ap_codec is not None,
        }
