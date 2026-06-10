"""경량 K-IFRS 도메인 온톨로지 (controlled vocabulary) — 이론·실무 2관점.

개념(동의어) -> 관련 기준서 + 연관 개념 + 이론(theory)/실무(practice) 관점.
세 곳에서 사용한다.
  1) 쿼리 그라운딩: 질문 속 개념의 정식 용어/기준서 번호를 검색 질의에 주입.
  2) 메타데이터 라우팅: 매칭된 standard_no 로 검색을 좁힌다(폴백 있음).
  3) 멀티쿼리 힌트: 실무 질문을 분해할 때 theory/practice 관점을 제시해
     LLM이 이론·실무를 아우르는 여러 하위 쿼리를 만들게 한다.

full OWL/트리플스토어가 아니라 사전 기반으로 시작한다(ROI 우선).
새 개념은 ONTOLOGY 에 한 줄 추가하면 된다. 대규모 확장은 prompts.ONTOLOGY_GEN_PROMPT 참고.

스키마:
  key: {
    "synonyms":  [표면형(소문자 비교)],
    "standards": [관련 기준서],
    "expand":    [한 홉 확장할 연관 개념],
    "theory":    [이론/규범 관점 하위주제],   # 선택
    "practice":  [실무/적용 관점 하위주제],   # 선택
  }
"""

from graph.config import ROUTING_MAX_CONCEPTS, ROUTING_MAX_STANDARDS, ROUTING_STRICT

ONTOLOGY = {
    "SPC": {
        "synonyms": [
            "spc", "특수목적기업", "특수목적법인", "구조화기업", "구조화된 기업",
            "구조화 기업", "spe", "special purpose", "도관",
        ],
        "standards": ["제1110호", "제1112호", "제1027호"],
        "expand": ["지배력", "연결"],
        "theory": [
            "지배력의 정의와 3요소(힘, 변동이익에 대한 노출, 연관 능력)",
            "연결 대상 판단 원칙",
            "구조화기업의 특성",
        ],
        "practice": [
            "SPC 지배력 판단 실무 절차",
            "계약·의사결정 권한 등 실질 지배 정황 평가",
            "연결 포함/제외 판단 사례",
            "위험과 보상의 귀속 평가",
        ],
        "include_basis": True,  # BC(SIC-12·구조화기업 논의)도 검색에 포함
    },
    "지배력": {
        "synonyms": ["지배력", "지배", "control", "힘과 변동이익"],
        "standards": ["제1110호"],
        "expand": ["연결"],
        "theory": ["힘(power)의 의미", "변동이익에 대한 노출 또는 권리", "힘과 이익의 연계"],
        "practice": [
            "실질적 권리 vs 방어권 구분",
            "위임된 의사결정권(본인·대리인) 판단",
            "잠재적 의결권 고려",
        ],
        "include_basis": True,
    },
    "연결": {
        "synonyms": ["연결", "연결재무제표", "consolidat", "종속기업"],
        "standards": ["제1110호", "제1112호"],
        "expand": [],
        "theory": ["연결재무제표 작성 의무", "연결 절차"],
        "practice": ["비지배지분 처리", "내부거래 제거"],
    },
    "지분공시": {
        "synonyms": [
            "타 기업에 대한 지분", "타기업 지분", "타 기업 지분", "지분의 공시",
            "지분 공시", "관계기업 공시", "공동기업 공시", "구조화기업 공시",
        ],
        "standards": ["제1112호"],
        "expand": [],
        "theory": ["지분 공시의 목적과 범위"],
        "practice": ["유의적 판단·가정 공시", "지분 관련 위험·재무영향 공시"],
    },
    "별도재무제표": {
        "synonyms": ["별도재무제표", "별도 재무제표"],
        "standards": ["제1027호"],
        "expand": [],
        "theory": ["별도재무제표의 정의", "측정 선택(원가/제1109호/지분법)"],
        "practice": ["측정방법 선택과 일관성", "배당수익 인식"],
    },
    "공정가치": {
        "synonyms": ["공정가치", "fair value", "공정가치 측정"],
        "standards": ["제1113호"],
        "expand": [],
        "theory": ["공정가치의 정의", "공정가치 서열체계(수준 1·2·3)"],
        "practice": ["수준 분류 판단", "관측가능 투입변수 우선 적용", "평가기법 선택"],
    },
    "금융상품": {
        "synonyms": [
            "금융상품", "금융자산", "금융부채", "지분상품", "financial instrument",
            "상각후원가", "기대신용손실", "ecl", "sppi",
        ],
        "standards": ["제1109호", "제1032호", "제1107호"],
        "expand": [],
        "theory": [
            "사업모형과 SPPI 기준",
            "분류 범주(상각후원가/FVOCI/FVPL)",
            "금융부채와 지분상품의 구분",
        ],
        "practice": [
            "사업모형 판단",
            "기대신용손실(ECL) 측정과 손상 단계",
            "복합금융상품의 부채·자본 분리",
        ],
    },
    "리스": {
        "synonyms": ["리스", "lease", "사용권자산", "리스부채"],
        "standards": ["제1116호"],
        "expand": [],
        "theory": ["사용권자산·리스부채 인식", "단일 리스 모형"],
        "practice": ["리스기간 산정", "할인율(증분차입이자율) 결정", "단기·소액 리스 면제 적용"],
    },
    "현금흐름": {
        "synonyms": ["현금흐름", "현금흐름표", "cash flow"],
        "standards": ["제1007호"],
        "expand": [],
        "theory": ["영업·투자·재무활동의 구분"],
        "practice": ["직접법 vs 간접법 선택", "이자·배당의 분류"],
    },
    "법인세": {
        "synonyms": ["법인세", "이연법인세", "income tax", "deferred tax"],
        "standards": ["제1012호"],
        "expand": [],
        "theory": ["이연법인세 자산·부채 인식 원칙", "일시적차이"],
        "practice": ["이연법인세자산 회수가능성(실현가능성) 판단", "적용 세율 결정"],
    },
    "재무제표표시": {
        "synonyms": ["재무제표 표시", "재무제표 작성", "표시와 공시"],
        "standards": ["제1001호"],
        "expand": [],
        "theory": ["재무제표의 구성요소와 표시 원칙"],
        "practice": ["계속기업 가정 평가", "중요성에 따른 표시·공시"],
    },
}


def _dedup(seq):
    out, seen = [], set()
    for x in seq:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


# 거의 모든 회계 질문에 나오는 흔한 단어 — 단독으로는 매칭 신호로 쓰지 않음.
_STOPWORDS = {
    "측정", "투자", "공시", "재무제표", "자산", "부채", "자본", "정보", "회계",
    "처리", "방법", "적용", "인식", "평가", "거래", "항목", "금액", "기준", "기준서",
}


def _matched_concepts(q, strict):
    """질문 q(소문자)에서 매칭되는 개념 key 목록.

    strict=False: 기존 방식(부분일치 아무거나 — 과다 매칭).
    strict=True : 흔한 단어 무시 + 매칭 길이로 점수화 + 약한 우연 매칭 제거 + 여유 캡.
                  '정답 개념'은 보통 구체적(긴) 단어로 강하게 매칭되어 유지된다.
    """
    if not strict:
        return [
            key
            for key, node in ONTOLOGY.items()
            if key.lower() in q or any(s.lower() in q for s in node.get("synonyms", []))
        ]

    scored = {}
    for key, node in ONTOLOGY.items():
        best = len(key) if (key.lower() in q and key not in _STOPWORDS) else 0
        for s in node.get("synonyms", []):
            sl = s.lower()
            if sl in _STOPWORDS or len(sl) < 2:
                continue
            if sl in q:
                best = max(best, len(sl))  # 더 길고 구체적일수록 고득점
        if best >= 2:
            scored[key] = best

    if not scored:
        return []
    top = max(scored.values())
    cutoff = max(2, top * 0.5)  # 최고점의 절반 이상만(약한 우연 매칭 제거)
    cands = [k for k, v in scored.items() if v >= cutoff]
    cands.sort(key=lambda k: scored[k], reverse=True)
    return cands[:ROUTING_MAX_CONCEPTS]


def match_concepts(query: str) -> dict:
    """질문에서 개념을 탐지하고 1-홉 확장 + 이론/실무 관점 수집.

    반환: {concepts, standards, terms, theory, practice}
      - standards: 메타데이터 필터용 기준서 번호
      - terms:     쿼리 그라운딩용 정식 개념어
      - theory/practice: 멀티쿼리 분해 힌트(이론/실무 하위주제)
    """
    if not query:
        return {"concepts": [], "standards": [], "terms": [],
                "theory": [], "practice": [], "include_basis": False}

    q = query.lower()
    matched = _matched_concepts(q, ROUTING_STRICT)

    seen, standards, theory, practice = set(), [], [], []
    include_basis = False
    frontier = list(matched)  # strict 모드에선 점수 높은 개념이 앞쪽(우선순위)
    while frontier:
        k = frontier.pop(0)
        if k in seen:
            continue
        seen.add(k)
        node = ONTOLOGY.get(k)
        if not node:
            continue
        for s in node.get("standards", []):
            if s not in standards:
                standards.append(s)  # 점수 높은 개념부터 추가 -> 우선순위 보존
        theory.extend(node.get("theory", []))
        practice.extend(node.get("practice", []))
        if node.get("include_basis"):
            include_basis = True
        for e in node.get("expand", []):
            if e not in seen:
                frontier.append(e)

    # 정답 기준서는 보통 최고점 개념에서 먼저 들어오므로, 상한을 둬도 유지된다.
    if ROUTING_STRICT and ROUTING_MAX_STANDARDS > 0:
        standards = standards[:ROUTING_MAX_STANDARDS]

    return {
        "concepts": sorted(seen),
        "standards": sorted(standards),
        "terms": sorted(seen),
        "theory": _dedup(theory),
        "practice": _dedup(practice),
        "include_basis": include_basis,
    }


def _merge_generated():
    """build_ontology.py가 생성한 ontology_generated.json을 병합(큐레이션 우선)."""
    import json
    from pathlib import Path

    path = Path(__file__).resolve().parent / "ontology_generated.json"
    if not path.exists():
        return
    try:
        gen = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return
    added = 0
    for key, node in gen.items():
        if key not in ONTOLOGY and isinstance(node, dict) and node.get("synonyms"):
            ONTOLOGY[key] = node
            added += 1
    if added:
        print(f"[ontology] 생성 항목 {added}개 병합 (총 {len(ONTOLOGY)})")


_merge_generated()
