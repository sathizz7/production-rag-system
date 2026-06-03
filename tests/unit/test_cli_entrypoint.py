def test_ingest_cli_is_importable_and_has_main() -> None:
    from rag.ingestion import cli

    assert callable(cli.main)
