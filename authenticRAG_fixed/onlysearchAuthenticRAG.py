import os
import json

# [F11] ของเดิมคัดลอกโค้ดค้นหา (sparse/dense/RRF/truncate/LLM) มาซ้ำทั้งหมด -> แก้ที่ authenticRAG.py แล้วฝั่งนี้ไม่ได้แก้ตาม
#       ตอนนี้ใช้คลาสเดียวกับ authenticRAG.py จึงได้ทุกการแก้ [F6]-[F10] ไปด้วย
from authenticRAG import AnthropicStyleContextualRAG


class AuthenticSearchRAG(AnthropicStyleContextualRAG):
    """ค้นหาจาก index ที่ authenticRAG.py สร้างไว้ (ไม่สร้าง/ลบ index)"""

    def search_for_question(self, question, k=5):
        """ค้นหาเอกสารที่เกี่ยวข้องกับคำถามและสร้างคำตอบ"""
        results = self.hybrid_search(question, k=k)

        if not results:
            return {"question": question, "answer": "ไม่พบข้อมูลที่เกี่ยวข้อง", "results": []}

        # แสดงเอกสารที่พบและใช้เนื้อหาเป็น context
        context = ""
        print(f"\nFound {len(results)} relevant documents for question: '{question}'")

        search_results = []
        for i, (doc_id, score, doc) in enumerate(results, 1):
            print(f"\nDOCUMENT {i}/{len(results)}")
            print(f"Score: {score}")
            print(f"Doc ID: {doc_id}")

            # เพิ่มข้อมูลเอกสารลงในผลลัพธ์
            result_info = {
                "doc_id": doc_id,
                "score": score,
                "content": doc.get("content", ""),
                "context": doc.get("contextualized_content", "")
            }
            search_results.append(result_info)

            # แสดงบริบทที่สร้างขึ้น
            if "contextualized_content" in doc:
                print(f"Context: {doc['contextualized_content']}")

            # แสดงเนื้อหาเฉพาะเอกสารแรก
            if i == 1:
                print("\nContent:")
                content = doc.get("content", "")
                if len(content) > 500:
                    print(content[:500] + "...(truncated)")
                else:
                    print(content)

            # เพิ่มเนื้อหาและบริบทเข้าไปใน context
            content = doc.get("content", "")
            context_info = doc.get("contextualized_content", "")
            context += f"Content: {content}\nContext: {context_info}\n\n"

        # ย่อ context ถ้าจำเป็น
        truncated_context = self.truncate_context(context)

        # สร้างคำตอบจาก context
        answer = self.generate_response(question, truncated_context)
        return {"question": question, "answer": answer, "results": search_results}

    def search_multiple_questions(self, questions, k=5):
        """ค้นหาและตอบหลายคำถาม"""
        results = []
        for i, question in enumerate(questions, 1):
            print(f"\n\n===== คำถามที่ {i}/{len(questions)} =====")
            print(f"คำถาม: {question}")

            try:
                # ค้นหาและตอบคำถาม
                result = self.search_for_question(question, k=k)

                # เพิ่มผลลัพธ์
                results.append(result)

                # แสดงคำตอบ
                print(f"\nคำตอบ: {result['answer']}")
            except Exception as e:
                print(f"เกิดข้อผิดพลาดในการค้นหาคำถาม '{question}': {e}")
                results.append({"question": question, "answer": f"เกิดข้อผิดพลาด: {str(e)}", "results": []})

        return results

    def export_results_to_json(self, results, output_file="search_results.json"):
        """ส่งออกผลลัพธ์ไปยังไฟล์ JSON"""
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\nบันทึกผลลัพธ์ไปยัง {output_file} สำเร็จ")


def main():
    # ตรวจสอบ API key
    if "OPENROUTER_API_KEY" not in os.environ and "LLM_API_KEY" not in os.environ:
        print("Error: OPENROUTER_API_KEY environment variable is not set")
        print("Please set it with: export OPENROUTER_API_KEY='your_api_key_here'")
        return

    # ตั้งค่าระบบ Search
    rag = AuthenticSearchRAG(
        opensearch_host="localhost",  # OpenSearch IP ตามที่ให้ไว้
        opensearch_port=9200
    )

    # ชุดคำถามที่ต้องการค้นหา
    questions = [
        'โรคหัดและโรคหัดเยอรมันแตกต่างกันอย่างไร?',
        'อธิบายสาเหตุของโรคหัดเยอรมันและการป้องกัน',
        'ทำไมโรคหัดเยอรมันจึงมีอันตรายกับหญิงตั้งครรภ์?',
        'ถ้าคนที่ฉีดวัคซีนป้องกันโรคหัดเยอรมันแล้ว จะมีโอกาสติดเชื้อหรือไม่?',
        'โรคหัดเยอรมันมีผลกระทบอย่างไรต่อระบบสาธารณสุขและเศรษฐกิจของประเทศ?'
    ]

    # ค้นหาและตอบคำถาม
    results = rag.search_multiple_questions(questions, k=5)

    # ส่งออกผลลัพธ์ไปยังไฟล์ JSON
    rag.export_results_to_json(results, "authentic_rag_search_results.json")


if __name__ == "__main__":
    main()
