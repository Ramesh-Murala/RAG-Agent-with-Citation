from vectorstore import Chunk, TfidfRetriever


def test_equal_scores_keep_ingestion_order_at_top_k_cutoff():
    retriever = TfidfRetriever()
    retriever.add_documents([
        Chunk(id="first", text="refund policy", source="handbook"),
        Chunk(id="second", text="refund policy", source="faq"),
        Chunk(id="third", text="refund policy", source="archive"),
    ])

    assert [hit.chunk.id for hit in retriever.retrieve("refund", k=2)] == [
        "first", "second",
    ]
