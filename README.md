# บันทึกการแก้ไข Authentic RAG (โค้ดอาจารย์ → ฉบับแก้)

**งาน KDAI NLP Assignment: repo นี้ตอบโจทย์แต่ละข้อที่ไหน**
- **ข้อ 1 ระบบ RAG ตามสไลด์:** ใช้โค้ดของอาจารย์ ([aekanun2020/2025-authenticRAG](https://github.com/aekanun2020/2025-authenticRAG)) เป็นฐาน ผลวัดของระบบต้นฉบับอยู่ที่ [results/diagnosis_2025-authenticRAG.md](results/diagnosis_2025-authenticRAG.md)
- **ข้อ 2 ปรับปรุงโดยไม่เปลี่ยน LLM:** โค้ดที่แก้อยู่ใน [authenticRAG_fixed/](authenticRAG_fixed/)
  - chunking: [F2] [F3]
  - search / analyzer: [F1]
  - scoring (RRF): [F8]
  - context ที่ใช้ทำ embedding: [F4]
  - LLM ค่าเริ่มต้นยังเป็น `qwen/qwen-2.5-72b-instruct` และ embedding model ยังเป็น `BAAI/bge-m3` ตามต้นฉบับ
  - ไม่ได้เพิ่ม reranking (ดูหัวข้อ "สิ่งที่ยังวัดไม่ได้ / ข้อจำกัด")
- **ข้อ 3 อธิบายว่าต่างกันตรงไหน ดีขึ้นอย่างไร เพราะอะไร:** หัวข้อ "ผลรวม" และ "ผลทีละขั้น" ให้ตัวเลขก่อนและหลังแก้ หัวข้อ "รายการแก้ไข" อธิบายทีละจุดว่าปัญหาคืออะไร หลักฐานคืออะไร แก้อย่างไร และได้ผลเท่าไร

**ที่มาของชุดวัดผล:**
- `steps/questions.json` และ `steps/common.py` มาจากเอกสารเวิร์กช็อป [Patchanon04/kdai-rag-workshop](https://github.com/Patchanon04/kdai-rag-workshop)
- ไฟล์ `questions.json` มี 29 ข้อ ใช้วัด 28 ข้อที่มี evidence
- `steps/step09_diagnose_teacher.py` เขียนเพิ่มสำหรับงานนี้ และใช้ `common.py` ของเวิร์กช็อป
- `corpus_input/` คัดลอกจาก repo ของอาจารย์

> ฐาน: `authenticRAG.py`, `onlysearchAuthenticRAG.py` จาก [aekanun2020/2025-authenticRAG](https://github.com/aekanun2020/2025-authenticRAG) (commit `4567d3d`)
> วัดผล: ต้นฉบับ 14 ต.ค. 2026, ฉบับแก้ 17 ต.ค. 2026 · embedding `BAAI/bge-m3` · OpenSearch 2.19.1 · ชุดคำถาม `steps/questions.json` 28 ข้อ · top-5
> แก้เฉพาะจุดที่ผิด โครงสร้างคลาส ชื่อ method และขั้นตอนเดิมยังอยู่ครบ ทุกจุดที่แก้มีป้าย `[F1]`–`[F11]` ในโค้ด
> ดูบรรทัดที่เปลี่ยนทั้งหมดได้ที่ [authenticRAG_fixed/changes_authenticRAG.diff](authenticRAG_fixed/changes_authenticRAG.diff) และ [authenticRAG_fixed/changes_onlysearchAuthenticRAG.diff](authenticRAG_fixed/changes_onlysearchAuthenticRAG.diff)

## โครงสร้าง repo

```
README.md                     ← เอกสารนี้
requirements.txt
authenticRAG_fixed/           ← โค้ดอาจารย์ที่แก้แล้ว + diff เทียบต้นฉบับ
steps/                        ← ชุดคำถาม + สคริปต์วัดผล
results/                      ← ผลวัดต้นฉบับ / ฉบับแก้
corpus_input/                 ← เอกสาร 4 ไฟล์จาก repo อาจารย์
```

## ติดตั้ง

ทดสอบกับ Python 3.11 และต้องมี Docker

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
docker run -d --name opensearch -p 9200:9200 -e "discovery.type=single-node" -e "DISABLE_SECURITY_PLUGIN=true" opensearchproject/opensearch:2.19.1
export OPENROUTER_API_KEY='your_api_key_here'          # ต้องใช้กับขั้นที่ 1-2 เท่านั้น
```

ครั้งแรกจะโหลด `BAAI/bge-m3` อัตโนมัติ (ประมาณ 2 GB)

## วิธีรัน

รันทุกคำสั่งจากโฟลเดอร์บนสุดของ repo

```bash
# 1) ingest: สร้าง index ใหม่ทุกครั้ง + เรียก LLM สร้าง context ทีละ chunk
python authenticRAG_fixed/authenticRAG.py
# 2) ถามคำถาม
python authenticRAG_fixed/onlysearchAuthenticRAG.py
# 3) วัดผล (ไม่ต้องใช้ API key, ใช้ index แยกชื่อ diag-*)
TEACHER_DIR=authenticRAG_fixed python steps/step09_diagnose_teacher.py
# 4) วัดโค้ดต้นฉบับของอาจารย์ด้วยสคริปต์เดียวกัน (ได้ results/diagnosis_2025-authenticRAG.md)
git clone https://github.com/aekanun2020/2025-authenticRAG
TEACHER_DIR=2025-authenticRAG python steps/step09_diagnose_teacher.py
```

ตั้ง `OPENROUTER_API_KEY` แบบเดิมได้ หรือจะใช้ผู้ให้บริการอื่นผ่าน `LLM_API_KEY`, `LLM_BASE_URL`, `LLM_MODEL` ก็ได้

---

## ผลรวม: ต้นฉบับเทียบฉบับแก้

วัดด้วยสคริปต์เดียวกัน (`steps/step09_diagnose_teacher.py`) ซึ่งเรียกโค้ดของแต่ละชุดตรง ๆ ทั้งสองแถวไม่ได้สร้าง context (ดูหัวข้อ "สิ่งที่ยังวัดไม่ได้")
ผลละเอียดอยู่ที่ `results/diagnosis_2025-authenticRAG.md` (ต้นฉบับ) และ `results/diagnosis_authenticRAG_fixed.md` (ฉบับแก้)

| ระบบ (hybrid RRF ที่ใช้ตอบจริง) | Hit@1 | Hit@5 | MRR@5 | token ที่ส่งให้ LLM ตอบ (mean) |
|---|---|---|---|---|
| ต้นฉบับ | 0.68 | 0.96 | 0.80 | 3,082 |
| **ฉบับแก้** | **0.82** | **0.96** | **0.88** | **1,970** |
| เปลี่ยนแปลง | **+0.14** | 0 | **+0.08** | **−36%** |

- **Hit@1**: สัดส่วนคำถามที่ chunk อันดับ 1 มีข้อความ evidence
- **Hit@5**: สัดส่วนคำถามที่เจอ evidence ใน 5 อันดับแรก
- **MRR@5**: ค่าเฉลี่ยของ 1/อันดับแรกที่เจอ (ไม่เจอใน top-5 นับเป็น 0)

chunk นับว่า "ถูก" เมื่อมีข้อความ evidence ที่คัดลอกตรงจากเอกสารอยู่ข้างใน จึงไม่ผูกกับวิธีตัด chunk และเทียบสองระบบที่ตัดต่างกันได้ ชุดคำถามมี 28 ข้อ ความต่าง 1 ข้อเท่ากับ 0.036

### ผลทีละขั้น (hybrid, ไม่มี context)

| ขั้น | BM25 Hit@1 / Hit@5 / MRR | dense Hit@1 / Hit@5 / MRR | **hybrid Hit@1 / Hit@5 / MRR** |
|---|---|---|---|
| 0. ต้นฉบับ | 0.00 / 0.07 / 0.04 | 0.71 / 0.96 / 0.81 | 0.68 / 0.96 / 0.80 |
| 1. + [F1] analyzer thai | 0.79 / 1.00 / 0.87 | 0.71 / 0.96 / 0.81 | 0.75 / 1.00 / 0.86 |
| 2. + [F2][F3] chunk 256 token + ชื่อเอกสาร | 0.64 / 0.89 / 0.76 | 0.71 / 1.00 / 0.84 | 0.82 / 0.96 / 0.87 |
| 3. + chunk 512 token + [F8] RRF (ฉบับแก้) | 0.61 / 0.93 / 0.76 | 0.71 / 0.96 / 0.81 | **0.82 / 0.96 / 0.88** |

การแก้ RRF ให้อันดับเริ่มที่ 1 [F8] ไม่ได้ทำให้ตัวเลขเปลี่ยน (เทียบที่ chunk 256 และ 512 แล้วได้ค่าเท่ากันทุกตัว) แก้เพื่อให้ตรงกับสูตรมาตรฐานเท่านั้น

### เลือกขนาด chunk (hybrid, analyzer thai, ไม่มี context)

| chunk_size | จำนวน chunk | Hit@1 | Hit@5 | MRR@5 |
|---|---|---|---|---|
| 256 | 140 | 0.82 | 0.96 | 0.87 |
| 384 | 95 | 0.79 | 1.00 | 0.87 |
| **512 (เลือก)** | **81** | **0.82** | **0.96** | **0.88** |
| 768 | 64 | 0.68 | 1.00 | 0.82 |
| 1024 | 58 | 0.82 | 1.00 | 0.90 |

เลือก 512 เพราะ Hit@1 เท่ากับค่าสูงสุด (0.82) และส่ง token ให้ LLM ตอบน้อยกว่า 1024 (เฉลี่ย 1,970 เทียบกับ 2,957) ส่วน 1024 ได้ Hit@5 และ MRR สูงสุด (1.00 / 0.90) ต่างจาก 512 ที่ Hit@5 เพียง 1 ข้อ ถ้าเน้นหาให้ครบมากกว่าประหยัด token ให้ใช้ 1024 (เปลี่ยนได้ที่ `build_chunker(chunk_size=...)`)

---

## รายการแก้ไข

### [F1] BM25 analyzer: `standard` → `thai`
- **ปัญหา:** `standard` ตัดคำด้วยช่องว่าง ภาษาไทยไม่เว้นวรรคระหว่างคำ ทั้งวลีจึงกลายเป็น term เดียว เช่น `ระยะฟักตัวของโรคหัดเยอรมันนานกี่วัน` → 1 term ส่วน `thai` ตัดได้เป็น `ระยะ | ฟักตัว | โรค | หัดเยอรมัน | นาน | กี่`
- **หลักฐาน:**

  | | standard | thai |
  |---|---|---|
  | ความยาว token ไทยเฉลี่ย | 22.9 ตัวอักษร | 4.5 ตัวอักษร |
  | คำในคำถามที่พบใน chunk ที่ถูก | 4% | 74% |
  | คำถามที่มีคำตรงกับ chunk ที่ถูกอย่างน้อย 1 คำ | 4/28 | 28/28 |
  | BM25 ค้นแล้วไม่เจออะไรเลย | 23/28 ข้อ | 0/28 ข้อ |

- **แก้:** เปลี่ยน `analysis.analyzer.default` และ analyzer ของ field `content`, `contextualized_content` เป็น `thai`
- **ผล:** BM25 Hit@1 0.00 → 0.79 และ hybrid MRR 0.80 → 0.86 ต้นฉบับมี hybrid แย่กว่า dense อย่างเดียว เพราะ BM25 ที่พังดึงคะแนนลง หลังแก้ hybrid ดีกว่า dense แล้ว

### [F2] chunk_size ไม่มีผล
- **ปัญหา:** `MarkdownNodeParser(chunk_size=256)` ตัดตามหัวข้อ markdown อย่างเดียวและไม่ใช้ `chunk_size` เลย
- **หลักฐาน:** เปลี่ยน `chunk_size` จาก 256 เป็น 100,000 แล้วได้ chunk เหมือนเดิมทุกตัว 66% ของ chunk (35/53) ใหญ่เกิน 256 token และ chunk ใหญ่สุดยาว 2,431 token
- **แก้:** ใช้ `MarkdownNodeParser` ตัดตามหัวข้อก่อน แล้วซอยต่อด้วย `SentenceSplitter(chunk_size=512, chunk_overlap=32)` อยู่ในเมธอด `build_chunker()`
  - ไม่นับ metadata (path ไฟล์, header_path) ตอนคิดขนาด chunk เพราะ `SentenceSplitter` หักความยาว metadata ออกจาก `chunk_size` ก่อนตัด ทำให้จำนวน chunk ขึ้นกับว่าเก็บไฟล์ไว้ที่ path ไหน (ทดสอบไฟล์ชุดเดียวกันที่ path ต่างกันได้ 82–90 chunk)
  - กันที่ไว้ให้ชื่อเอกสารที่ใส่หัว chunk [F3] ขนาดรวมจึงไม่เกิน `chunk_size`
  - นับ token ด้วย tokenizer ของ bge-m3 ตัวเดียวกับที่ใช้ทำ embedding
  - ตัดประโยคที่ช่องว่างและขึ้นบรรทัดใหม่ เพราะตัวตัดประโยคค่าเริ่มต้น (NLTK punkt) ใช้กับภาษาอังกฤษ
- **ผล:** chunk เกินขนาด 66% → 0% ขนาดใหญ่สุด 2,431 → 512 token และไม่มี evidence ถูกตัดขาดข้าม chunk (0/49)

### [F3] chunk ไม่รู้ว่าเป็นของโรคไหน
- **ปัญหา:** chunk ย่อย เช่น "อาการ" หรือ "การรักษา" ไม่มีชื่อโรคอยู่ในเนื้อหา เมื่อ chunk เล็กลงตาม [F2] ปัญหานี้จะยิ่งมากขึ้น
- **หลักฐาน:** ต้นฉบับมี chunk ที่ไม่มีชื่อโรค 9% (5/53)
- **แก้:** ใส่ชื่อเอกสาร (หัวข้อ `#` บรรทัดแรก) ไว้หัว chunk เช่น `[หัดเยอรมัน อาการ สาเหตุ ...]`
- **ผล:** 0/81

### [F4] Contextual Retrieval ไม่ได้ส่งเอกสารเต็มให้ LLM
- **ปัญหา:** บรรทัด `full_doc_content = doc.page_content` ส่ง chunk ไปแทนเอกสาร LLM จึงไม่รู้ว่า chunk อยู่ส่วนไหนของเอกสาร ซึ่งขัดกับหลักของวิธี Contextual Retrieval
- **หลักฐาน:**
  - LLM เห็นเอกสารเฉลี่ย 8% ของไฟล์
  - ทดลองรันต้นฉบับแบบมี context: context เป็นภาษาไทยแค่ 9% และ hybrid MRR ลดจาก 0.80 เป็น 0.77 ส่วน dense ลดจาก 0.81 เป็น 0.78 แปลว่า context ทำให้แย่ลง
- **แก้:**
  - เก็บเอกสารเต็มไว้ใน `self.full_docs` ตอนโหลด แล้วส่งเอกสารเต็มให้ LLM
  - เพิ่มคำสั่ง "Answer in the same language as the document" ใน prompt
- **ผล:** LLM เห็นเอกสาร 100% ส่วนผลต่อการค้นหายังวัดไม่ได้ (ดูหัวข้อ "สิ่งที่ยังวัดไม่ได้")

### [F5] index เก่าถูกใช้ต่อเงียบ ๆ
- **ปัญหา:** `_create_or_update_indices()` สร้าง index เฉพาะตอนที่ยังไม่มี ถ้าเคยรันต้นฉบับไว้ แล้วรันโค้ดที่แก้ analyzer ระบบจะใช้ analyzer `standard` ตัวเดิมต่อไป นอกจากนี้ถ้าจำนวน chunk ลดลง chunk ชุดเก่าที่ `_id` ไม่ถูกเขียนทับจะค้างอยู่ใน index
- **แก้:** เพิ่ม `reset_indices()` ลบแล้วสร้าง index ใหม่ทุกครั้งที่ ingest และ `__init__` ไม่สร้าง index อีกต่อไป `onlysearchAuthenticRAG.py` จึงไม่ไปสร้าง index เปล่าโดยไม่ตั้งใจ

### [F6] LLM ตั้งค่าได้
- ค่าเริ่มต้นยังเป็น `qwen/qwen-2.5-72b-instruct` ผ่าน OpenRouter ตามต้นฉบับ
- เพิ่ม `LLM_API_KEY`, `LLM_BASE_URL`, `LLM_MODEL` ให้ใช้ผู้ให้บริการอื่นที่รองรับ OpenAI SDK ได้ เช่น Gemini

### [F7] context ว่างเมื่อใช้โมเดลที่คิดก่อนตอบ
- **ปัญหา:** `generate_context` ตั้ง `max_tokens=100` โมเดลที่คิดก่อนตอบ (เช่น gemma) ใช้ token หมดก่อนได้คำตอบ
- **หลักฐาน:** ทดสอบ 3 chunk ได้ context ว่าง 2 อัน
- **แก้:** เพิ่มเป็น 1000 ซึ่งเป็นแค่เพดาน prompt ยังขอคำตอบสั้นอยู่ โมเดลที่ไม่คิดก่อนตอบ (เช่น Qwen) จึงตอบยาวเท่าเดิม

### [F8] สูตร RRF
- **แก้:** `enumerate(results)` → `enumerate(results, 1)` ให้อันดับเริ่มที่ 1 ตามสูตร 1/(k + rank)
- **ผล:** ตัวเลขไม่เปลี่ยน

### [F9] `truncate_context` ใช้กับภาษาไทยไม่ได้
- **ปัญหา:** ประมาณจำนวน token จากจำนวนคำที่คั่นด้วยช่องว่าง × 1.3 แต่ภาษาไทยมีช่องว่างน้อย จึงประมาณต่ำไปมากและแทบไม่เคยย่อจริง
- **หลักฐาน:** context ที่ส่งให้ LLM ตอบมี token จริงเฉลี่ย 3,082 แต่สูตรเดิมประมาณได้ 720 (ต่ำไป 4.3 เท่า) ทดสอบสั่งย่อเหลือ 1,000 token แล้วยังเกินทั้ง 28/28 ข้อ (ยาวสุด 4,323 token)
- **แก้:** นับและตัดด้วย tokenizer ของ bge-m3
- **ผล:** ย่อได้ตามสั่ง เกิน 0/28 ข้อ

### [F10] ข้อความ error ถูกเก็บเป็น context
- **ปัญหา:** `call_qwen_api` คืนข้อความ `"Qwen API error: ..."` แทนการ raise ข้อความนี้จึงถูกบันทึกลง index เป็น context และถูกนำไปทำ embedding เงียบ ๆ เกิดได้ง่ายเพราะ LLM โดน rate limit (429) บ่อย ระหว่างทดสอบเจอ 429 หลายครั้ง
- **แก้:** ลองใหม่เมื่อเจอ 429 หรือ 5xx (รอ 5, 10, 20, 40 วินาที) ถ้ายังไม่สำเร็จให้ raise

### [F11] `onlysearchAuthenticRAG.py` คัดลอกโค้ดค้นหาซ้ำ
- **ปัญหา:** ไฟล์คัดลอก sparse, dense, RRF, truncate และ LLM มาเองทั้งหมด แก้ที่ `authenticRAG.py` แล้วฝั่งนี้จะไม่ได้แก้ตาม
- **แก้:** สืบทอดจาก `AnthropicStyleContextualRAG` และเก็บเฉพาะส่วนที่ต่างจริง (`search_for_question` ที่คืน dict, ตอบหลายคำถาม, export JSON)

### เล็กน้อย
- **หา corpus:** ต้นฉบับใช้ `./corpus_input` ต้องรันจากโฟลเดอร์ของไฟล์เท่านั้น ฉบับแก้หา `corpus_input` ข้างไฟล์หรือที่โฟลเดอร์แม่
- **เอา `download_corpus()` ออก:** ฟังก์ชันนี้ไม่ได้ดาวน์โหลดอะไร แค่สร้างโฟลเดอร์ `./corpus_input` เปล่า ซึ่งจะบังโฟลเดอร์ corpus จริง

---

## สิ่งที่ตรวจแล้วว่าถูกต้อง (ไม่ได้แก้)
- **k-NN ใช้ cosine จริง:** score ที่ OpenSearch คืนมาเท่ากับสูตร (1+cos)/2 = 0.8776 ไม่ใช่สูตร L2 (0.6714)
- **ตัด chunk ตามหัวข้อไม่ทำให้ evidence ขาด:** 0/49 ทั้งต้นฉบับและฉบับแก้

## สิ่งที่ยังวัดไม่ได้ / ข้อจำกัด
- **ผลของ context ที่สร้างจากเอกสารเต็ม [F4]:** ผลต่อการค้นหายังไม่ได้วัด เพราะต้องส่งเอกสาร 18,000–24,000 ตัวอักษรทุกครั้ง รวม 81 ครั้ง ซึ่งเกินโควตาฟรีของ Gemini (gemma-4-26b รับ input 16,000 token/นาที, โดน 429 ทุกครั้ง) ถ้าจะวัด ให้ใช้ OpenRouter (Qwen ตามต้นฉบับ) หรือ API แบบจ่ายเงิน แล้วรัน:

  ```bash
  TEACHER_DIR=authenticRAG_fixed python steps/step09_diagnose_teacher.py --with-context
  ```

- **chunk ผิดโรคใน top-5 ยังสูง:** BM25 19%, hybrid 9% เพราะคำถามแบบ paraphrase ไม่ได้ระบุชื่อโรค การจะแก้ส่วนนี้ต้องเพิ่ม reranker หรือกรองด้วย LLM ซึ่งเกินขอบเขตการแก้ให้ถูกต้อง
- **ทดสอบแบบรันจริงทั้งระบบ:** โหลดเอกสาร → `reset_indices` → index → hybrid search → LLM ตอบ "ระยะฟักตัวของโรคหัดเยอรมัน ... ประมาณ 12-24 วัน โดยเฉลี่ยประมาณ 16-18 วัน" ตรงกับคำตอบอ้างอิง รอบนี้ข้ามการสร้าง context เพราะติดโควตา
