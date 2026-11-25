import pytest

from vectorstore import Chunk, TfidfRetriever, simple_chunk_text


def test_failed_index_update_preserves_existing_retrieval():
    retriever = TfidfRetriever()
    retriever.add_documents([Chunk(id="policy", text="refund policy", source="handbook")])
    before = retriever.retrieve("refund")

    with pytest.raises(ValueError, match="Chunks must contain text"):
        retriever.add_documents([Chunk(id="blank", text="  ", source="upload")])

    after = retriever.retrieve("refund")
    assert [(hit.chunk.id, hit.score) for hit in after] == [
        (hit.chunk.id, hit.score) for hit in before
    ]
    retriever.add_documents([Chunk(id="shipping", text="shipping policy", source="faq")])
    assert retriever.retrieve("shipping")[0].chunk.id == "shipping"


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
