import os

from dotenv import load_dotenv

load_dotenv()

# -------------------------------
# CA bundle (Avast's local TLS-scanning proxy re-signs all HTTPS traffic with
# its own root, which isn't in Python's certifi bundle even though Windows
# trusts it — so grpc/httpx-based clients fail cert verification unless
# pointed at a bundle that includes it).
# -------------------------------

_CA_BUNDLE = os.path.join(os.path.dirname(__file__), ".ca_bundle.pem")
if os.path.exists(_CA_BUNDLE):
    os.environ.setdefault("SSL_CERT_FILE", _CA_BUNDLE)
    os.environ.setdefault("REQUESTS_CA_BUNDLE", _CA_BUNDLE)
    os.environ.setdefault("GRPC_DEFAULT_SSL_ROOTS_FILE_PATH", _CA_BUNDLE)

# -------------------------------
# Google Vertex AI
# -------------------------------

PROJECT_ID = "vertex-ai-project-494805"

LOCATION = "asia-south1"

MODEL_NAME = "gemini-2.5-flash"

# -------------------------------
# Qdrant Cloud (region: us-east-1-1)
# -------------------------------

QDRANT_URL = os.environ["QDRANT_URL"]
QDRANT_API_KEY = os.environ["QDRANT_API_KEY"]

# Which Qdrant collection is the active KB. Both search.py and ingest.py read
# this so switching KBs is a single .env edit + restart, no code change.
# All known collections (edit RAG_COLLECTION in .env to any of these):
#   aps_knowledge           -> APS knowledge base
#   kvk_knowledge           -> KVK knowledge base
#   grow_it_knowledge       -> Grow It KB
#   salam_kisan_knowledge   -> Salam Kisan KB
#   nacro_new_knowledge     -> Nacro New KB
#   rythu_nestham_knowledge -> Rythu Nestham KB (not yet ingested)
# telugu_knowledge was the old pre-separation collection that merged every
# KB together (see the cross-contamination bugs that caused) — it's been
# deleted, so the fallback below must never point to it again.
RAG_COLLECTION = os.getenv("RAG_COLLECTION", "aps_knowledge")
