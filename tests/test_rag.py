from pathlib import Path

from kaggle_multi_agent.rag import LocalKnowledgeBase


def test_local_knowledge_base_build_search_and_reload(tmp_path: Path) -> None:
    source_dir = tmp_path / "curated"
    source_dir.mkdir()
    (source_dir / "course.md").write_text(
        "# Course\n\nReflection loops improve agent quality.\n\nRAG helps retrieve relevant notes.",
        encoding="utf-8",
    )
    (source_dir / "competition.md").write_text(
        "# Competition\n\nUse offline metrics with public and private splits.",
        encoding="utf-8",
    )
    kb = LocalKnowledgeBase.build(source_dir)
    results = kb.search("offline public private metrics", k=2)
    assert results
    assert "public" in results[0].text.lower() or "private" in results[0].text.lower()

    index_dir = tmp_path / "index"
    kb.save(index_dir)
    restored = LocalKnowledgeBase.load(index_dir)
    restored_results = restored.search("reflection loops", k=1)
    assert restored_results
    assert "reflection" in restored_results[0].text.lower()
