"""Zero-configuration browser demo for inspecting retrieval evidence."""

from __future__ import annotations

import argparse
import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from vectorstore import Chunk, build_retriever

ROOT = Path(__file__).parent
FIXTURE = json.loads((ROOT / "evaluation" / "cases.json").read_text())
CHUNKS = [Chunk(**document) for document in FIXTURE["documents"]]
RETRIEVER_NAMES = ("tfidf", "bm25", "dense_lsa", "hybrid", "hybrid_rerank")
RETRIEVERS = {name: build_retriever(name) for name in RETRIEVER_NAMES}
for retriever in RETRIEVERS.values():
    retriever.add_documents(CHUNKS)

HTML = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>Grounded Retrieval Lab</title>
<style>
:root{color-scheme:dark;--bg:#07111f;--panel:#101d2f;--line:#26384e;--text:#eef4fb;--muted:#a9b8ca;--accent:#6ee7d8}
*{box-sizing:border-box}body{margin:0;background:radial-gradient(circle at top,#132b42,var(--bg) 45%);color:var(--text);font:16px/1.55 system-ui,sans-serif}
main{max-width:980px;margin:auto;padding:56px 24px}.eyebrow{color:var(--accent);font-weight:700;letter-spacing:.12em;text-transform:uppercase}
h1{font-size:clamp(2.2rem,6vw,4.6rem);line-height:1;margin:.25em 0}.lede{color:var(--muted);max-width:720px;font-size:1.1rem}
form{display:grid;grid-template-columns:1fr 190px 110px;gap:10px;margin:32px 0}input,select,button{border:1px solid var(--line);border-radius:10px;padding:13px;background:var(--panel);color:var(--text);font:inherit}
button{background:var(--accent);color:#06221e;border:0;font-weight:800;cursor:pointer}.meta{color:var(--muted);margin:8px 0 20px}
.card{background:rgba(16,29,47,.86);border:1px solid var(--line);border-radius:14px;padding:20px;margin:12px 0}.top{display:flex;justify-content:space-between;gap:12px}
.source{color:var(--accent);font-weight:700}.score{font-variant-numeric:tabular-nums;color:var(--muted)}mark{background:#244f52;color:#dffffb;padding:1px 3px;border-radius:3px}
.note{border-left:3px solid var(--accent);padding-left:14px;color:var(--muted)}@media(max-width:700px){form{grid-template-columns:1fr}main{padding-top:32px}}
</style></head><body><main>
<div class="eyebrow">RAG reliability experiment</div><h1>Grounded Retrieval Lab</h1>
<p class="lede">Compare five local retrieval strategies over 20 synthetic company-policy documents. Every result exposes its source, chunk ID, and raw score.</p>
<form id="search"><input id="q" value="How soon must missing equipment be reported?" aria-label="Question">
<select id="method"><option value="hybrid_rerank">Hybrid + rerank</option><option value="hybrid">Hybrid</option><option value="bm25">BM25</option><option value="dense_lsa">Dense LSA</option><option value="tfidf">TF-IDF</option></select>
<button>Retrieve</button></form><div id="out"></div>
<p class="note">This demo shows retrieval evidence without calling an LLM. Dense LSA is truncated SVD trained on this corpus; it is not a pretrained embedding model.</p>
</main><script>
const form=document.querySelector('#search'),out=document.querySelector('#out');
function esc(s){return s.replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]))}
form.addEventListener('submit',async e=>{e.preventDefault();out.innerHTML='<p class="meta">Searching…</p>';
 const q=document.querySelector('#q').value,m=document.querySelector('#method').value;
 const r=await fetch('/api/search?q='+encodeURIComponent(q)+'&retriever='+encodeURIComponent(m)),d=await r.json();
 if(!r.ok){out.innerHTML='<p class="meta">'+esc(d.error)+'</p>';return}
 out.innerHTML='<p class="meta">'+d.results.length+' chunks · '+esc(d.retriever)+' · '+d.elapsed_ms.toFixed(3)+' ms</p>'+
 d.results.map((x,i)=>'<article class="card"><div class="top"><span class="source">#'+(i+1)+' '+esc(x.source)+'</span><span class="score">score '+x.score.toFixed(5)+'</span></div><p>'+esc(x.text)+'</p><small>chunk_id='+esc(x.chunk_id)+'</small></article>').join('');
});form.requestSubmit();
</script></body></html>"""


class DemoHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self._send(HTTPStatus.OK, "text/html; charset=utf-8", HTML.encode())
            return
        if parsed.path != "/api/search":
            self._json(HTTPStatus.NOT_FOUND, {"error": "Not found"})
            return
        params = parse_qs(parsed.query)
        question = params.get("q", [""])[0].strip()
        name = params.get("retriever", ["hybrid_rerank"])[0]
        if not question:
            self._json(HTTPStatus.BAD_REQUEST, {"error": "Question is required"})
            return
        if name not in RETRIEVERS:
            self._json(HTTPStatus.BAD_REQUEST, {"error": f"Unknown retriever: {name}"})
            return
        import time

        started = time.perf_counter()
        results = RETRIEVERS[name].retrieve(question, k=3)
        elapsed_ms = (time.perf_counter() - started) * 1000
        self._json(
            HTTPStatus.OK,
            {
                "query": question,
                "retriever": name,
                "elapsed_ms": elapsed_ms,
                "results": [
                    {
                        "chunk_id": item.chunk.id,
                        "source": item.chunk.source,
                        "text": item.chunk.text,
                        "score": item.score,
                    }
                    for item in results
                ],
            },
        )

    def log_message(self, format: str, *args: object) -> None:
        return

    def _json(self, status: HTTPStatus, payload: dict) -> None:
        self._send(status, "application/json", json.dumps(payload).encode())

    def _send(self, status: HTTPStatus, content_type: str, body: bytes) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    print(f"Open http://{args.host}:{args.port}")
    ThreadingHTTPServer((args.host, args.port), DemoHandler).serve_forever()
