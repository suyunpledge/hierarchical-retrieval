"""排序与缓存一致性回归；仅使用临时库和离线嵌入。"""
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier, Event

import numpy as np
import pytest

from test_fixes import CountingEmbedding, CONV_A, CONV_B
from hierarchical_retrieval import HConfig, HierarchicalRetrieval


@pytest.fixture
def engine(tmp_path):
    config = HConfig(storage_root=str(tmp_path))
    return HierarchicalRetrieval(embedding=CountingEmbedding(config), config=config)


def test_contexts_share_query_vector_and_need_one_embed(engine, monkeypatch):
    result = engine.ingest(CONV_A)
    query = engine.key_sentence_lib.get_sentence(result["sentence_ids"][0])["text"]
    calls = []
    original = engine.embedding.embed
    def counted(text, strict=False):
        calls.append(text)
        return original(text, strict)
    monkeypatch.setattr(engine.embedding, "embed", counted)
    out = engine.retrieve(query)
    assert calls == [query]
    q = original(query)
    store = engine.cloud_cache._vector_store
    for ctx in out["full_contexts"]:
        fid = store._id_map[f'{ctx["entry_id"]}_chunk_{ctx["chunk_index"]}']
        assert ctx["score"] == pytest.approx(float(np.dot(q, store.index.reconstruct(fid))))


def test_full_domain_does_not_hide_better_global_answer(engine, monkeypatch):
    a, b = engine.ingest(CONV_A), engine.ingest(CONV_B)
    domain = engine.topic_manager.get_domain(a["domain_id"])
    gold = engine.key_sentence_lib.get_sentence(b["sentence_ids"][0])["text"]
    monkeypatch.setattr(engine.topic_manager, "detect_domain_by_vector", lambda q: domain)
    monkeypatch.setattr(engine.topic_manager, "search_in_domain_by_vector",
                        lambda *args, **kwargs: [{
                            "id": "local", "score": 0.7, "metadata": {
                                "sentence_id": "local", "text": "较弱的域内答案",
                                "source_entry_id": a["entry_id"]}}])
    out = engine.retrieve(gold, top_k_sentences=1, top_k_context=0)
    assert out["key_sentences"][0]["text"] == gold
    assert len(out["key_sentences"]) == 1


def test_hybrid_ranking_and_duplicate_path_evidence(engine, monkeypatch):
    monkeypatch.setattr(engine.topic_manager, "detect_domain_by_vector", lambda q: None)
    monkeypatch.setattr(engine.key_sentence_lib, "search_by_vector",
                        lambda *args, **kwargs: [{
                            "sentence_id": "ks", "score": 0.8, "text": "索引",
                            "source_entry_id": "a"}])
    def ctx(eid, text, score):
        return dict(entry_id=eid, chunk_index=0, chunk_text=text, score=score,
                    timestamp=0, full_text=text)
    monkeypatch.setattr(engine.cloud_cache, "search_in_entry_by_vector",
                        lambda *args, **kwargs: [ctx("a", "其他说明", 0.7)])
    monkeypatch.setattr(engine.cloud_cache, "candidate_chunks",
                        lambda *args: [ctx("a", "其他说明", 0.7),
                                       ctx("b", "bge-m3 参数", 0.65)])
    monkeypatch.setattr(engine.cloud_cache, "attach_full_text", lambda items: items)
    out = engine.retrieve("bge-m3")
    assert [c["entry_id"] for c in out["full_contexts"]] == ["b", "a"]
    assert set(out["full_contexts"][1]["sources"]) == {"direct", "key_sentence"}


def test_strong_context_survives_small_memory_budget(engine, monkeypatch):
    monkeypatch.setattr(engine, "retrieve", lambda q: {
        "key_sentences": [{"text": "较弱背景" * 15, "score": 0.6}],
        "full_contexts": [{"chunk_text": "正确答案：停车位每月一百五十元。",
                           "score": 0.8, "hybrid_score": 0.95}],
    })
    out = engine.assemble_context("停车位价格", char_budget=100)
    assert "正确答案" in out
    assert len(out) <= 100


def test_result_cache_format_variants_and_config_changes(engine):
    query = "停车位\n收费"
    engine.retrieve(query)
    engine.retrieve("  停车位\r\n收费\r\n")
    stats = engine.cache_stats()["retrieval"]
    assert stats["hits"] == 1 and stats["misses"] == 1
    engine.config.chunk_similarity_threshold = 0.9
    engine.retrieve(query)
    assert engine.cache_stats()["retrieval"]["misses"] == 2
    engine.embedding = CountingEmbedding(engine.config)
    engine.retrieve(query)
    assert engine.cache_stats()["retrieval"]["misses"] == 3


def test_embedding_format_reuse_preserves_meaningful_differences():
    emb = CountingEmbedding(HConfig())
    vectors = emb.embed_batch(["  A\r\nB ", "A\nB", "A\rB"])
    assert emb.requests == ["A\nB"]
    assert all(np.array_equal(vectors[0], v) for v in vectors)
    emb.embed("a\nB")
    emb.embed("A B")
    emb.embed("A  B")
    assert len(emb.requests) == 4  # 保留大小写与内部空白，不能模糊合并


def test_plain_question_cache_preserves_query_and_code_punctuation(engine):
    from hierarchical_retrieval.utils.text_processor import TextProcessor

    for query in ("疫苗本放在哪里", "疫苗本放在哪里?", "疫苗本放在哪里？"):
        assert engine.retrieve(query)["query"] == query
    stats = engine.cache_stats()["retrieval"]
    assert (stats["hits"], stats["misses"]) == (2, 1)
    for literal in ("为什么用 regex.*?", "选哪个 https://test/?", "哪一行\nABC?",
                    "符号“？”是什么?", "哪个文件??", "Is this SQL?", "a?"):
        assert TextProcessor.retrieval_query(literal) == literal


def test_paraphrases_share_result_without_merging_distinct_intents(engine):
    for query in ("forge/pricing.py 的表叫什么名字？", "请问 forge/pricing.py 的表叫什么?"):
        assert engine.retrieve(query)["query"] == query
    stats = engine.cache_stats()["retrieval"]
    assert (stats["hits"], stats["misses"]) == (1, 1)
    engine.retrieve("forge/pricing.py 的表为什么改名？")
    engine.retrieve("forge/model.py 的表叫什么？")
    assert engine.cache_stats()["retrieval"]["misses"] == 3


def test_fulltext_concurrent_misses_read_disk_once(engine, monkeypatch):
    cache = engine.cloud_cache
    eid = engine.ingest(CONV_A)["entry_id"]
    cache.clear_cache()
    calls = []
    original = cache._text_store.load
    def read(key):
        calls.append(key)
        return original(key)
    monkeypatch.setattr(cache._text_store, "load", read)
    barrier = Barrier(8)
    def fetch(_):
        barrier.wait(timeout=5)
        return cache.get_full_text(eid)
    with ThreadPoolExecutor(max_workers=8) as pool:
        assert list(pool.map(fetch, range(8))) == [CONV_A] * 8
    assert calls == [eid]


def test_record_update_cannot_be_overwritten_by_old_read(engine, monkeypatch):
    cache = engine.cloud_cache
    eid = engine.ingest(CONV_A)["entry_id"]
    cache.clear_cache()
    reading, release, writing = Event(), Event(), Event()
    original = cache._text_store.load
    def read(key):
        record = original(key)
        reading.set()
        assert release.wait(5)
        return record
    monkeypatch.setattr(cache._text_store, "load", read)
    def update():
        writing.set()
        cache._save_record(eid, {"full_text": "更新后的全文", "meta": {}})
    with ThreadPoolExecutor(max_workers=2) as pool:
        reader = pool.submit(cache.get_full_text, eid)
        assert reading.wait(5)
        writer = pool.submit(update)
        assert writing.wait(5)
        release.set()
        assert reader.result(timeout=5) == CONV_A
        writer.result(timeout=5)
    assert cache.get_full_text(eid) == "更新后的全文"


def test_in_progress_write_cannot_fill_result_cache(engine):
    engine._write_began = True
    try:
        engine.retrieve("写入期间的查询")
        assert engine.cache_stats()["retrieval"]["size"] == 0
    finally:
        engine._write_began = False
    engine.retrieve("写入期间的查询")
    assert engine.cache_stats()["retrieval"]["size"] == 1


def test_config_change_during_retrieve_cannot_fill_old_result(engine, monkeypatch):
    original = engine._retrieve_uncached
    def racing(*args):
        result = original(*args)
        engine.config.chunk_lexical_weight = 0.3
        return result
    monkeypatch.setattr(engine, "_retrieve_uncached", racing)
    engine.retrieve("查询")
    assert engine.cache_stats()["retrieval"]["size"] == 0


@pytest.mark.parametrize("score,text,remains", [
    (0.64, "钥匙归还", True),    # Outside the dense window even with perfect lexical overlap.
    (0.71, "天气", True),       # Inside the window, below the reranked winner.
    (0.71, "钥匙归还", False),   # A lexical match can change the winner.
    (0.85, "天气", False),      # Recheck ties at the raw cutoff.
])
def test_append_checks_dense_and_hybrid_cutoffs(engine, monkeypatch, score, text, remains):
    query = "钥匙归还"
    engine.config.chunk_candidates = 2
    engine.config.chunk_lexical_weight = 0.3
    engine.config.chunk_similarity_threshold = 0.6
    monkeypatch.setattr(engine.cloud_cache, "candidate_chunks", lambda *_: [
        {"entry_id": "a", "chunk_index": 0, "chunk_text": "天气", "score": 0.85},
        {"entry_id": "b", "chunk_index": 0, "chunk_text": "气温", "score": 0.68},
    ])
    engine.retrieve(query, top_k_context=1)
    vector = engine.embedding.embed(query)
    monkeypatch.setattr(engine.cloud_cache, "entry_chunk_vectors", lambda _: [(vector * score, text)])
    engine._revalidate_after_append("c", [])
    assert engine.cache_stats()["retrieval"]["size"] == int(remains)
