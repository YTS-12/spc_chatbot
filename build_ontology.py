"""온톨로지 대량 생성 도구.

chunks_cache.jsonl의 각 기준서 본문(body) 표본을 ONTOLOGY_GEN_PROMPT에 넣어
LLM이 이론·실무 관점의 개념 사전 항목(JSON)을 생성하게 한다.
결과는 graph/ontology_generated.json 에 저장되고, graph/ontology.py 가 import 시
자동 병합한다(큐레이션 우선 — 손으로 쓴 핵심 개념은 덮어쓰지 않음).

전제: 먼저 `python ingest_pinecone.py --reset` 으로 chunks_cache.jsonl 생성.
실행:  python build_ontology.py
"""

import json
from collections import OrderedDict

from langchain_openai import ChatOpenAI

from graph.config import BASE_DIR, CHAT_MODEL, CHUNK_CACHE_PATH
from graph.prompts import ONTOLOGY_GEN_PROMPT

OUT_PATH = BASE_DIR / "graph" / "ontology_generated.json"
SAMPLE_CHARS = 5000


def load_cache():
    rows = []
    with open(CHUNK_CACHE_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def sample_body(rows, standard_no, limit=SAMPLE_CHARS):
    """해당 기준서의 body 청크를 limit 글자까지 이어붙인 표본."""
    texts, total = [], 0
    for o in rows:
        md = o.get("metadata", {})
        if md.get("standard_no") == standard_no and md.get("section_type") == "body":
            texts.append(o.get("text", ""))
            total += len(texts[-1])
            if total >= limit:
                break
    return "\n".join(texts)[:limit]


def parse_json_array(raw):
    s, e = raw.find("["), raw.rfind("]")
    if s == -1 or e == -1:
        return None
    try:
        return json.loads(raw[s : e + 1])
    except Exception:
        return None


def main():
    rows = load_cache()

    standards = OrderedDict()
    for o in rows:
        md = o.get("metadata", {})
        no = md.get("standard_no")
        if no and no != "unknown":
            standards.setdefault(no, md.get("standard_name") or "")

    llm = ChatOpenAI(model=CHAT_MODEL, temperature=0)
    result = {}
    for no, name in standards.items():
        sample = sample_body(rows, no)
        if not sample:
            print(f"[{no}] 본문 표본 없음 -> 건너뜀")
            continue
        inp = f"기준서: {no} {name}\n\n[본문 발췌]\n{sample}"
        raw = llm.invoke(ONTOLOGY_GEN_PROMPT.format(input=inp)).content.strip()
        arr = parse_json_array(raw)
        if not arr:
            print(f"[{no}] JSON 파싱 실패")
            continue
        cnt = 0
        for item in arr:
            key = (item.get("key") or "").strip()
            if not key or not item.get("synonyms"):
                continue
            result[key] = {
                "synonyms": item.get("synonyms", []),
                "standards": item.get("standards") or [no],
                "expand": item.get("expand", []),
                "theory": item.get("theory", []),
                "practice": item.get("practice", []),
            }
            cnt += 1
        print(f"[{no}] {name}: {cnt} concepts")

    OUT_PATH.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\n총 {len(result)} concepts -> {OUT_PATH.name}")


if __name__ == "__main__":
    main()
