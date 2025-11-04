from vectordb import VectorDB
from app import RAGAssistant, load_documents

def test_document_loading():
    docs = load_documents()
    print(f"[TEST] Document Loading: Loaded {len(docs)} documents")
    assert len(docs) > 0, "No documents loaded"
    print("[TEST] Passed document loading")

def test_chunking():
    vdb = VectorDB()
    docs = load_documents()
    if not docs:
        print("[TEST] No docs to chunk")
        return
    chunks = vdb.chunk_text(docs[0]["content"])
    print(f"[TEST] Chunking: Document split into {len(chunks)} chunks")
    assert len(chunks) > 0, "Chunking failed"
    print("[TEST] Passed chunking")

def test_ingestion():
    vdb = VectorDB()
    docs = load_documents()
    vdb.add_documents(docs)
    print("[TEST] Ingestion: Documents ingested to vector database")
    print("[TEST] Passed ingestion")

def test_similarity_search():
    vdb = VectorDB()
    query = "What is AI?"
    results = vdb.search(query)
    print(f"[TEST] Similarity Search: Found {len(results.get('documents', []))} documents")
    assert len(results.get("documents", [])) > 0, "No search results"
    print("[TEST] Passed similarity search")

def test_rag_memory():
    rag = RAGAssistant()
    q1 = "What is AI?"
    a1 = rag.invoke(q1)
    print(f"[TEST] Q1: {q1}\nA1: {a1}")
    
    q2 = "How does AI relate to machine learning?"
    a2 = rag.invoke(q2)
    print(f"[TEST] Q2: {q2}\nA2: {a2}")
    
    assert a1 and a2, "RAG assistant failed with memory"
    print("[TEST] Passed RAG assistant with memory")

def run_all_tests():
    test_document_loading()
    test_chunking()
    test_ingestion()
    test_similarity_search()
    test_rag_memory()
    print("All tests completed.")

if __name__ == "__main__":
    run_all_tests()