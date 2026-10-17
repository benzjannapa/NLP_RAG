"""Step 9: วินิจฉัยระบบครูทีละขั้น (chunk -> analyzer -> search) ให้ออกมาเป็นตัวเลข

เรียกโค้ดของครู (authenticRAG.py) ตรง ๆ ไม่เขียนเลียนแบบ: load_documents, _create_or_update_indices,
add_documents_with_context, sparse_search, dense_search, hybrid_search
แต่ใช้ชื่อ index ของตัวเอง (diag-*) จึงไม่ทับ index ที่ครูสร้างไว้

    python steps/step09_diagnose_teacher.py                 # ไม่เรียก LLM: context ว่าง
    python steps/step09_diagnose_teacher.py --with-context  # ให้ LLM สร้าง context แบบที่โค้ดครูทำจริง (ช้า)
    python steps/step09_diagnose_teacher.py --chunk-size=512  # ลองขนาด chunk อื่น (ต้องเป็นโค้ดครูที่มี build_chunker)

ตั้ง TEACHER_DIR ถ้า repo ครูไม่ได้อยู่ที่ ./teacher หรือ ~/2025-authenticRAG
วัดโค้ดชุดที่แก้แล้ว:  TEACHER_DIR=authenticRAG_fixed python steps/step09_diagnose_teacher.py
ผลบันทึกที่ results/diagnosis_<ชื่อโฟลเดอร์>.md
"""
import contextlib
import inspect
import io
import json
import os
import re
import statistics
import sys
from pathlib import Path

from common import ROOT, embedder, llm, opensearch

TEACHER = next(p.resolve() for p in [Path(os.environ.get("TEACHER_DIR", "")), ROOT / "teacher",
                           Path.home() / "2025-authenticRAG"] if (p / "authenticRAG.py").exists())
CORPUS = next(p for p in (TEACHER / "corpus_input", TEACHER.parent / "corpus_input") if p.exists())
sys.path.insert(0, str(TEACHER))
import authenticRAG as T  # noqa: E402
from llama_index.core.node_parser import MarkdownNodeParser  # noqa: E402

# ค่าเริ่มต้น = chunk_size ที่ตั้งในโค้ดครู (build_chunker) หรือ 256 ตามโค้ดต้นฉบับ
_default = (inspect.signature(T.AnthropicStyleContextualRAG.build_chunker).parameters["chunk_size"].default
            if hasattr(T.AnthropicStyleContextualRAG, "build_chunker") else 256)
CHUNK_SIZE = next((int(a.split("=")[1]) for a in sys.argv if a.startswith("--chunk-size=")), _default)

FILES = ["1.md", "2.md", "44.md", "5555.md"]           # ตาม main() ของครู
DISEASE = {"1.md": "หัดเยอรมัน", "2.md": "อหิวาต", "44.md": "ต้อกระจก", "5555.md": "กรดไหลย้อน"}
QUESTIONS = [q for q in json.loads((Path(__file__).with_name("questions.json")).read_text(encoding="utf-8"))
             if q["evidence"]]
K = 5
THAI = re.compile(r"[฀-๿]")
VEC, BM25, BM25_THAI = "diag-teacher-vector", "diag-teacher-bm25", "diag-thai-bm25"
quiet = lambda: contextlib.redirect_stdout(io.StringIO())  # noqa: E731  โค้ดครู print ทุกครั้งที่ค้น


def pct(x):
    return f"{100 * x:.0f}%"


def teacher_rag(chunk_size=CHUNK_SIZE):
    """object ของคลาสครูโดยไม่ผ่าน __init__ (ซึ่งบังคับ API key) แล้วใส่เฉพาะของที่ method ต้องใช้"""
    rag = object.__new__(T.AnthropicStyleContextualRAG)
    if hasattr(rag, "build_chunker"):   # โค้ดครูที่แก้ chunking แล้ว
        rag.build_chunker(chunk_size=chunk_size)
    else:
        rag.md_parser = MarkdownNodeParser(chunk_size=chunk_size)
    return rag


def load_chunks(rag):
    with quiet():
        return rag.load_documents([str(CORPUS / f) for f in FILES])


# ---------------------------------------------------------------- 1) chunking
def diagnose_chunks(chunks, report):
    tok = embedder().tokenizer
    n_tok = [len(tok(c.page_content, add_special_tokens=False)["input_ids"]) for c in chunks]
    src = [Path(c.metadata["source"]).name for c in chunks]
    same = [c.page_content for c in chunks] == [c.page_content for c in load_chunks(teacher_rag(100_000))]
    docs = {f: (CORPUS / f).read_text(encoding="utf-8") for f in FILES}

    split_ev = [e for q in QUESTIONS for e in q["evidence"]
                if e in docs[q["file"]] and not any(e in c.page_content for c in chunks)]
    n_ev = sum(len(q["evidence"]) for q in QUESTIONS)
    no_name = sum(DISEASE[s] not in c.page_content for s, c in zip(src, chunks))
    q = statistics.quantiles(n_tok, n=10)
    rows = [
        ("จำนวน chunk", f"{len(chunks)}", ""),
        ("ขนาด chunk (token ของ bge-m3) min / median / p90 / max",
         f"{min(n_tok)} / {statistics.median(n_tok):.0f} / {q[-1]:.0f} / {max(n_tok)}", ""),
        (f"chunk ที่เกิน chunk_size={CHUNK_SIZE} ที่ตั้งไว้", f"{sum(t > CHUNK_SIZE for t in n_tok)}/{len(chunks)} "
         f"({pct(sum(t > CHUNK_SIZE for t in n_tok) / len(chunks))})", "❌" if any(t > CHUNK_SIZE for t in n_tok) else "✅"),
        (f"เปลี่ยน chunk_size {CHUNK_SIZE} → 100,000 แล้วได้ chunk เหมือนเดิมทุกตัว", str(same),
         "❌ chunk_size ไม่มีผล" if same else "✅"),
        ("chunk เล็กเกินไป (< 30 token)", f"{sum(t < 30 for t in n_tok)}", ""),
        ("chunk ที่ไม่มีชื่อโรคของตัวเองอยู่ในเนื้อหา", f"{no_name}/{len(chunks)} ({pct(no_name / len(chunks))})",
         "❌ ค้นแล้วแยกไม่ออกว่าเป็นโรคไหน" if no_name else "✅"),
        ("evidence ที่ถูกตัดขาดข้าม chunk", f"{len(split_ev)}/{n_ev}", "✅" if not split_ev else "❌"),
    ]
    report.append(("1) Chunking", rows))
    return n_tok


# ---------------------------------------------------------------- 2) analyzer
def tokens(client, analyzer, text):
    return [t["token"] for t in client.indices.analyze(body={"analyzer": analyzer, "text": text})["tokens"]]


def diagnose_analyzer(client, chunks, report):
    gold = {q["id"]: next((c.page_content for c in chunks if any(e in c.page_content for e in q["evidence"])), "")
            for q in QUESTIONS}
    rows = []
    for an in ("standard", "thai"):
        th = [t for c in chunks for t in tokens(client, an, c.page_content) if THAI.search(t)]
        cover, any_hit = [], 0
        for q in QUESTIONS:
            qt = {t for t in tokens(client, an, q["question"]) if THAI.search(t)}
            gt = set(tokens(client, an, gold[q["id"]]))
            cover.append(len(qt & gt) / max(len(qt), 1))
            any_hit += bool(qt & gt)
        rows += [
            (f"[{an}] ความยาว token ไทยเฉลี่ย (ตัวอักษร)", f"{statistics.mean(map(len, th)):.1f}",
             "❌ ไม่ได้ตัดคำ" if an == "standard" else ""),
            (f"[{an}] token ไทยที่ยาวเกิน 15 ตัวอักษร", pct(sum(len(t) > 15 for t in th) / len(th)), ""),
            (f"[{an}] คำในคำถามที่พบใน chunk ที่ถูก (mean)", pct(statistics.mean(cover)), ""),
            (f"[{an}] คำถามที่มีคำตรงกับ chunk ที่ถูกอย่างน้อย 1 คำ", f"{any_hit}/{len(QUESTIONS)}", ""),
        ]
    ex = QUESTIONS[1]["question"]
    rows.append(("ตัวอย่าง: " + ex, " | ".join(tokens(client, "standard", ex)) + "  →  "
                 + " | ".join(tokens(client, "thai", ex)), "standard → thai"))
    report.append(("2) Analyzer (BM25)", rows))


# ---------------------------------------------------------------- 3) search
def build_indexes(client, rag, chunks, with_context):
    rag.opensearch_client, rag.vector_index_name, rag.bm25_index_name = client, VEC, BM25
    rag.embedding_dim = len(embedder().encode("x"))
    rag.encoder = type("Enc", (), {"embed_query": staticmethod(lambda t: embedder().encode(t).tolist())})()
    if with_context:  # โค้ดครูเรียก Qwen ตรง ๆ: ใช้ llm() ของเวิร์กช็อปแทน ส่วน prompt คงเดิม
        # ponytail: max_tokens=100 ของครูพอสำหรับ Qwen แต่โมเดลที่ "คิด" ก่อนตอบ (เช่น gemma) จะได้ข้อความว่าง
        rag.call_qwen_api = lambda p, max_tokens=500, temperature=0.2: llm(p, max(max_tokens, 1000), temperature)
    else:
        rag.call_qwen_api = lambda p, max_tokens=500, temperature=0.2: ""
    docs = {f: (CORPUS / f).read_text(encoding="utf-8") for f in FILES}
    generate, rag.seen = rag.generate_context, []

    def recording(document, chunk):  # จดว่า LLM เห็นเอกสารกี่ % ของไฟล์ที่ chunk นี้มาจาก
        full = next((d for f, d in docs.items() if chunk[:200] in d), None) or \
            next(d for f, d in docs.items() if DISEASE[f] in chunk)
        rag.seen.append(len(document) / len(full))
        return generate(document, chunk)
    rag.generate_context = recording
    for name in (VEC, BM25, BM25_THAI):
        client.indices.delete(index=name, ignore_unavailable=True)
    with quiet():
        rag._create_or_update_indices()         # settings/mapping ของครูทุกตัว
        rag.add_documents_with_context(chunks)
    body = client.indices.get(index=BM25)[BM25]
    body = {"mappings": body["mappings"], "settings": {"analysis": {"analyzer": {"default": {"type": "thai"}}}}}
    for f in ("content", "contextualized_content"):
        body["mappings"]["properties"][f]["analyzer"] = "thai"
    client.indices.create(index=BM25_THAI, body=body)
    client.reindex(body={"source": {"index": BM25}, "dest": {"index": BM25_THAI}}, refresh=True)


def knn_space_check(client, rag):
    """index ตั้ง cosinesimil ไว้ที่ settings: ดูว่า score ที่ได้จริงตรงกับสูตร cosine หรือ L2"""
    import numpy as np
    qv = np.array(rag.encoder.embed_query(QUESTIONS[0]["question"]))
    hit = client.search(index=VEC, body={"size": 1, "query": {"knn": {"embedding": {"vector": qv.tolist(), "k": 1}}}})
    h = hit["hits"]["hits"][0]
    dv = np.array(h["_source"]["embedding"])
    cos = qv @ dv / np.linalg.norm(qv) / np.linalg.norm(dv)
    l2 = 1 / (1 + np.sum((qv - dv) ** 2))
    space = "cosinesimil" if abs(h["_score"] - (1 + cos) / 2) < abs(h["_score"] - l2) else "l2"
    return space, h["_score"], (1 + cos) / 2, l2


def diagnose_search(client, rag, chunks, report):
    src = {f"doc_{i}": Path(c.metadata["source"]).name for i, c in enumerate(chunks)}

    def thai_sparse(q, k=10):
        rag.bm25_index_name = BM25_THAI
        try:
            return T.AnthropicStyleContextualRAG.sparse_search(rag, q, k)
        finally:
            rag.bm25_index_name = BM25

    def thai_hybrid(q, k=5):
        sparse = rag.sparse_search
        rag.sparse_search = thai_sparse
        try:
            return T.AnthropicStyleContextualRAG.hybrid_search(rag, q, k)
        finally:
            rag.sparse_search = sparse

    an = client.indices.get_mapping(index=BM25)[BM25]["mappings"]["properties"]["content"].get("analyzer", "standard")
    systems = {f"ครู: BM25 ({an})": lambda q: rag.sparse_search(q, K),
               "ครู: dense": lambda q: rag.dense_search(q, K),
               "ครู: hybrid RRF (ระบบเดิม)": lambda q: rag.hybrid_search(q, K),
               "แก้ analyzer: BM25 (thai)": lambda q: thai_sparse(q, K),
               "แก้ analyzer: hybrid RRF": lambda q: thai_hybrid(q, K)}
    if an == "thai":  # โค้ดครูแก้ analyzer แล้ว สองแถวนี้จะซ้ำกับแถว "ครู"
        systems = {n: f for n, f in systems.items() if not n.startswith("แก้ analyzer")}
    rows, per_q, prompts = [], {}, []
    for name, run in systems.items():
        ranks, empty, wrong, total = [], 0, 0, 0
        for q in QUESTIONS:
            with quiet():
                hits = run(q["question"])[:K]
            if name == "ครู: hybrid RRF (ระบบเดิม)":  # context แบบที่ search_for_question ประกอบส่งให้ LLM
                prompts.append("".join(f"Content: {s.get('content', '')}\nContext: {s.get('contextualized_content', '')}\n\n"
                                       for _, _, s in hits))
            empty += not hits
            wrong += sum(src[d] != q["file"] for d, _, _ in hits)
            total += len(hits)
            ranks.append(next((r for r, (_, _, s) in enumerate(hits, 1)
                               if any(e in s["content"] for e in q["evidence"])), None))
        per_q[name] = ranks
        n = len(QUESTIONS)
        rows.append((name, f"Hit@1={sum(r == 1 for r in ranks) / n:.2f}  Hit@5={sum(r is not None for r in ranks) / n:.2f}"
                     f"  MRR@5={sum(1 / r for r in ranks if r) / n:.2f}",
                     f"ไม่เจออะไรเลย {empty}/{n} ข้อ · chunk ผิดโรค {pct(wrong / max(total, 1))}"))
    ctx = [h["_source"]["contextualized_content"] for h in client.search(
        index=BM25, body={"size": 1000, "query": {"match_all": {}}})["hits"]["hits"]]
    seen = statistics.mean(rag.seen)
    rows.append(("สัดส่วนเอกสารที่ LLM เห็นตอนสร้าง context (mean)", pct(seen),
                 "✅" if seen > 0.99 else "❌ ส่ง chunk ไปแทนเอกสารเต็ม (full_doc_content = doc.page_content)"))
    if any(ctx):
        rows.append(("context ที่ LLM สร้าง: ว่าง / เป็นภาษาไทย",
                     f"{pct(sum(not c for c in ctx) / len(ctx))} / {pct(sum(bool(THAI.search(c)) for c in ctx) / len(ctx))}",
                     "context ภาษาอังกฤษไม่ช่วย BM25 ตอนค้นด้วยคำถามภาษาไทย"))
    # truncate_context(max_tokens=4000): สูตรเดิมประมาณ token = จำนวนคำที่คั่นด้วยช่องว่าง x 1.3
    tok = embedder().tokenizer
    rag.tokenizer = tok
    count = lambda t: len(tok.encode(t, add_special_tokens=False))  # noqa: E731
    real = [count(t) for t in prompts]
    est = [len(t.split()) * 1.3 for t in prompts]
    after = [count(rag.truncate_context(t, max_tokens=1000)) for t in prompts]
    rows.append(("context ที่ส่งให้ LLM ตอบ: token จริง / ที่ truncate_context ประมาณ (mean)",
                 f"{statistics.mean(real):,.0f} / {statistics.mean(est):,.0f}",
                 f"ประมาณต่ำไป {statistics.mean(real) / statistics.mean(est):.1f} เท่า"))
    rows.append(("ทดสอบ truncate_context(max_tokens=1000): ข้อที่ยังเกิน 1000 token หลังย่อ",
                 f"{sum(a > 1010 for a in after)}/{len(after)} (max {max(after):,})",
                 "✅" if max(after) <= 1010 else "❌ ย่อไม่ได้จริงกับภาษาไทย"))
    space, score, cos_s, l2_s = knn_space_check(client, rag)
    rows.append(("k-NN space ที่ใช้จริง", space,
                 f"score={score:.4f} สูตร cosine={cos_s:.4f} สูตร L2={l2_s:.4f}"
                 + ("" if space == "cosinesimil" else " ❌ ไม่ใช่ cosine ตามที่ตั้งไว้")))
    report.append((f"3) Search ({len(QUESTIONS)} คำถาม, top-{K})", rows))
    return per_q


def main(argv):
    with_context = "--with-context" in argv
    client, rag = opensearch(), teacher_rag()
    chunks = load_chunks(rag)
    report = []
    print("1) chunking ..."); diagnose_chunks(chunks, report)
    print("2) analyzer ..."); diagnose_analyzer(client, chunks, report)
    print("3) สร้าง index ด้วยโค้ดครู + ค้น ..."); build_indexes(client, rag, chunks, with_context)
    diagnose_search(client, rag, chunks, report)

    md = [f"# วินิจฉัยระบบครู (authenticRAG.py) ทีละขั้น\n",
          f"โค้ดที่วัด: `{TEACHER}`\n\n",
          f"context จาก LLM: {'สร้างจริงด้วยโค้ดครู' if with_context else 'ไม่ได้สร้าง (contextualized_content ว่าง)'}"
          f" · chunk_size={CHUNK_SIZE}\n"]
    for title, rows in report:
        md += [f"\n## {title}\n", "| ตัวชี้วัด | ค่า | หมายเหตุ |", "|---|---|---|"]
        md += [f"| {a} | {b} | {c} |" for a, b, c in rows]
    text = "\n".join(md) + "\n"
    print("\n" + text)
    (ROOT / "results").mkdir(exist_ok=True)
    out = ROOT / "results" / f"diagnosis_{TEACHER.name}{'_context' if with_context else ''}.md"
    out.write_text(text, encoding="utf-8")
    print(f"บันทึกที่ {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main(sys.argv[1:])
