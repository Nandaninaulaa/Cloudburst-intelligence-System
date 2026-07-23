from rag_engine import search_docs

result = search_docs(
    "What is a cloudburst?"
)

for r in result:
    print(r[:500])