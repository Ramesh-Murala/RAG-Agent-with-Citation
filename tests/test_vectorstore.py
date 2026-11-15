from vectorstore import Chunk, TfidfRetriever, simple_chunk_text


def test_sentence_cut_does_not_shrink_chunk_below_requested_overlap():
    chunks = simple_chunk_text("abcdefgh. ijklmnopqrstuv", "handbook", chunk_size=12, overlap=10)

    assert chunks[0].text == "abcdefgh. ij"
    assert chunks[0].text[-10:] == chunks[1].text[:10]


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
