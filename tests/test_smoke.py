"""Lightweight import + structure smoke tests (no model calls)."""
import os
import yaml


def test_configs_load():
    for p in ["config/default.yaml", "config/experiments.yaml"]:
        with open(p) as f:
            cfg = yaml.safe_load(f)
        assert cfg, f"{p} is empty"


def test_default_config_keys():
    with open("config/default.yaml") as f:
        cfg = yaml.safe_load(f)
    for k in ["corpus", "chunking", "embedding", "vector_store", "retrieval", "generation", "judge"]:
        assert k in cfg, f"missing key: {k}"


def test_gold_dataset_wellformed():
    import json
    rows = json.load(open("data/eval/gold_qa.json"))
    assert len(rows) >= 1
    for r in rows:
        assert "question" in r and "ground_truth" in r


def test_modules_importable():
    import src.ingest.chunk_docs   # noqa
    import src.ingest.build_index  # noqa
    import src.retrieve.retriever  # noqa
    import src.generate.answer     # noqa
    import src.eval.run_eval       # noqa
