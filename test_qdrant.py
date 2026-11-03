
import httpx
from qdrant_client import QdrantClient

url = "http://127.0.0.1:6333"

print("1. Testing direct HTTP connection...")

try:
    response = httpx.get(
        url,
        timeout=10,
        trust_env=False
    )
    print("HTTP status:", response.status_code)
    print("Response:", response.text)
except Exception as e:
    print("HTTP failed:", repr(e))

print("\n2. Testing Qdrant Python client...")

client = QdrantClient(
    url=url,
    timeout=15
)

try:
    collections = client.get_collections()
    print("SUCCESS! Qdrant client connected.")
    print("Collections:", collections)
except Exception as e:
    print("Qdrant client failed:", repr(e))
finally:
    client.close()
