# K-IFRS SPC 회계 챗봇

K-IFRS 기준서를 근거로 SPC(특수목적기업) 관련 회계 질문에 답하는 RAG 챗봇입니다.
연결 판단부터 금융상품·공정가치·리스·공시까지, SPC를 다룰 때 부딪히는 회계 쟁점을
기준서 문단을 인용해 설명합니다.

LangGraph로 파이프라인을 구성했고, 임베딩은 OpenAI, 벡터 검색은 Pinecone,
재정렬은 로컬 cross-encoder를 씁니다.

## 무엇을 하나

회계 실무에서는 "이 SPC를 연결해야 하나?", "운용사가 보수만 받으면 그 펀드를 연결하나?"
같은 판단이 자주 필요한데, 정작 근거는 여러 기준서에 흩어져 있습니다. 이 챗봇은 질문을 받으면
관련 기준서를 찾아 읽고, 원칙을 그 상황에 적용해 결론까지 제시합니다. 정의만 나열하지 않습니다.

답변에는 항상 어떤 기준서·페이지·문단을 봤는지 출처가 붙습니다. LLM이 적당히 지어낸 인용이 아니라
실제로 검색에 쓰인 문서를 그대로 보여주는 거라, 원문과 직접 대조해 확인할 수 있습니다.

지금은 SPC 회계와 직접 얽히는 12개 기준서(연결재무제표·금융상품·공정가치·리스·별도재무제표·지분공시 등)를 다룹니다.

## 어떻게 동작하나

질문이 들어오면 대략 이런 흐름을 탑니다.

```
질문 → 하위질문으로 분해 → 검색 → 검색 결과 평가 → 답변 생성
                                   └ 부실하면 재검색 / 정보부족 안내
```

신경 쓴 부분 몇 가지:

- **질문 분해** — 실무 질문은 한 번에 답하기 어려워서, 이론(정의·요건)과 실무(절차·판단·사례)
  관점으로 나눠 여러 번 검색합니다.
- **하이브리드 검색** — 의미 기반(임베딩) 검색만으로는 "문단 12", "수준 2" 같은 정확한 표현을
  놓치길래 BM25 키워드 검색을 같이 돌려 합칩니다.
- **본문 우선** — 기준서는 분량의 절반쯤이 '결론도출근거(BC)' 같은 배경 설명입니다. 적재할 때
  본문/BC/소수의견 등으로 분류해두고 검색은 본문 위주로 하되, SPC처럼 BC에 핵심(SIC-12 등)이
  있는 주제는 BC도 일부 끌어옵니다.
- **개념→기준서 라우팅** — "SPC 연결"이라고 물으면 제1110·1112·1027호 쪽으로 검색 범위를
  좁히는 가벼운 개념 사전을 둡니다.
- **재정렬** — 마지막에 cross-encoder로 질문과 가장 관련 깊은 순서로 다시 정렬합니다.

기준서 PDF를 임베딩해 넣는 적재 과정은 `ingest_pinecone.py`에 모여 있습니다. 한컴 PDF에서 깨진
공백을 복원하고, 페이지를 섹션 유형으로 분류하고, 청크로 잘라 OpenAI 임베딩으로 Pinecone에 올립니다.

## 프로젝트 구조

```
spc_chatbot/
├── app.py                 Streamlit 채팅 UI
├── ingest_pinecone.py     PDF → 임베딩 → Pinecone 적재
├── build_ontology.py      기준서에서 개념 사전 생성 (선택)
├── requirements.txt
└── graph/
    ├── config.py          설정 (.env 로딩)
    ├── workflow.py        LangGraph 파이프라인
    ├── retrieval.py       Pinecone 검색
    ├── hybrid.py          BM25 + 융합
    ├── rerank.py          cross-encoder 재정렬
    ├── ontology.py        개념 사전 (라우팅·그라운딩)
    ├── prompts.py         프롬프트
    └── state.py           그래프 상태
```

## 실행

```bash
pip install -r requirements.txt
pip install transformers rank_bm25   # 재정렬·하이브리드를 쓸 경우

cp .env.example .env                 # OpenAI / Pinecone 키 입력
```

기준서 PDF를 `data/K-IFRS/Proto/`에 넣고 한 번 적재합니다(저작권상 PDF는 저장소에 포함하지 않습니다).

```bash
python ingest_pinecone.py --reset    # Pinecone에 벡터 + BM25 캐시 생성
streamlit run app.py
```

첫 질문은 재정렬 모델을 내려받느라 조금 느리고(수십 초), 이후부터는 빨라집니다.

## 설정

`.env`에서 조정할 수 있는 값이 많지만, 자주 건드리는 건 이 정도입니다.

- `CHAT_MODEL` / `EMBEDDING_MODEL` — 사용할 모델
- `TOP_K` — 답변에 쓸 검색 결과 수 (기본 6)
- `RERANK_ENABLED` — 재정렬 on/off
- `MULTIQUERY_ENABLED`, `MAX_QUERIES` — 질문 분해
- `ROUTING_MAX_STANDARDS` — 라우팅할 기준서 상한

나머지는 `.env.example`에 주석과 함께 정리해 뒀습니다.

## 참고

- 기준서 PDF, 실제 API 키(`.env`), 적재 캐시는 저장소에 올리지 않습니다.
- 임베딩 벡터는 파일이 아니라 Pinecone(클라우드)에 저장됩니다. 그래서 내려받은 사람은 위처럼 한 번 적재해야 합니다.
- 프로토타입이라 답변은 참고용입니다. 실제 회계 판단은 원문 기준서와 전문가 검토가 필요합니다.
