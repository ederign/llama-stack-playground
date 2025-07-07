# mcp_server.py
from fastapi import FastAPI
from sse_starlette.sse import EventSourceResponse
import json

app = FastAPI()

@app.get("/sse")
async def mcp_context():
    # Example: serve a list of toolgroups (this could be per-session in a real app)
    data = {
        "toolgroups": [
            {
                "name": "rag-dogs",
                "args": {
                    "vector_db_ids": ["toy_faiss_db"]
                }
            }
        ]
    }
    return EventSourceResponse((json.dumps(data) for _ in range(1)))


# Run with: uvicorn mcp_server:app --port 8421
