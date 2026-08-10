"""Outbound calling: AI-calls-human via Tata Smartflo's Click-to-Call Support API.

Tata rings customer_number first; once they answer, it bridges to whatever
destination the api_key's DID is configured to route to on Tata's dashboard
(Voice Streaming -> this bot).
"""

import asyncio
import os

import requests
from loguru import logger

OUTBOUND_CALLER_ID = os.getenv("OUTBOUND_CALLER_ID", "")
OUTBOUND_TEST_DESTINATION_NUMBER = os.getenv("OUTBOUND_TEST_DESTINATION_NUMBER", "")

# Tata Smartflo endpoint: customer-first flow. Tata rings customer_number
# first; once they answer, it bridges to whatever destination this api_key
# was assigned to when it was generated on Tata's dashboard.
# Docs: https://docs.smartflo.tatatelebusiness.com/reference/v1click_to_call_support
TATA_CLICK_TO_CALL_SUPPORT_URL = "https://api-smartflo.tatateleservices.com/v1/click_to_call_support"
TATA_CLICK_TO_CALL_SUPPORT_API_KEY = os.getenv("TATA_CLICK_TO_CALL_SUPPORT_API_KEY", "")


async def trigger_customer_first_call(
    customer_number: str = "",
    api_key: str = "",
    caller_id: str = "",
    customer_ring_timeout: int | None = None,
    call_timeout: int | None = None,
) -> dict:
    """Places an outbound call via /v1/click_to_call_support: rings customer_number
    first, then bridges to whatever destination api_key was assigned to on Tata's
    dashboard.

    Returns the parsed JSON response from Tata (or {"success": False, ...} on
    a request-level failure, e.g. network error or missing api_key).
    """
    customer_number = customer_number or OUTBOUND_TEST_DESTINATION_NUMBER
    caller_id = caller_id or OUTBOUND_CALLER_ID
    api_key = api_key or TATA_CLICK_TO_CALL_SUPPORT_API_KEY

    if not api_key:
        logger.warning("⚠️ TATA_CLICK_TO_CALL_SUPPORT_API_KEY not set — call not placed.")
        return {"success": False, "message": "TATA_CLICK_TO_CALL_SUPPORT_API_KEY not set"}
    if not customer_number:
        logger.warning("⚠️ Missing customer_number for outbound call_support.")
        return {"success": False, "message": "customer_number is required"}

    body = {
        "customer_number": customer_number,
        "api_key": api_key,
        "async": 1,
    }
    if caller_id:
        body["caller_id"] = caller_id
    if customer_ring_timeout is not None:
        body["customer_ring_timeout"] = customer_ring_timeout
    if call_timeout is not None:
        body["call_timeout"] = call_timeout

    try:
        resp = await asyncio.to_thread(
            requests.post,
            TATA_CLICK_TO_CALL_SUPPORT_URL,
            json=body,
            timeout=10,
            verify=False,  # local TLS-interception proxy — see Bot.py's SSL patch block
        )
        logger.info(f"📞 Outbound click-to-call-support → HTTP {resp.status_code}: {resp.text[:200]}")
        return resp.json()
    except Exception as e:
        logger.error(f"❌ Outbound click-to-call-support request failed: {e}")
        return {"success": False, "message": str(e)}
