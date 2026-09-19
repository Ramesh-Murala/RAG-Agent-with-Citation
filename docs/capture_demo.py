from pathlib import Path
import json
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
DEST = Path(__file__).parent / "assets"
DEST.mkdir(exist_ok=True)
from types import SimpleNamespace
from agent import RAGCitationAgent
from vectorstore import Chunk, TfidfRetriever
# Scripted provider; retrieval and validation execute the real application code.
class ScriptedClient:
    def __init__(self):
        self.messages = self
    def create(self, **kwargs):
        data = {"answer": "Employees receive 20 days of paid leave each year.", "citations": [{"source": "handbook", "chunk_id": "leave", "supporting_quote": "20 days of paid leave"}], "self_reported_confidence": 0.9, "grounded": True}
        return SimpleNamespace(content=[SimpleNamespace(type="tool_use", input=data)])
r = TfidfRetriever()
r.add_documents([Chunk(id="leave", source="handbook", text="Employees receive 20 days of paid leave each year.")])
request = {"question": "How many days of paid leave?"}
result = RAGCitationAgent(retriever=r, client=ScriptedClient()).query(request["question"])
response = {"answer": result.answer.model_dump(), "final_confidence": result.final_confidence, "is_low_confidence": result.is_low_confidence, "used_fallback": result.used_fallback, "attempt_count": len(result.attempts)}
assert result.answer.grounded

(DEST / "request.json").write_text(json.dumps(request, indent=2) + "\n")
(DEST / "response.json").write_text(json.dumps(response, indent=2) + "\n")
print(json.dumps(response, indent=2))
