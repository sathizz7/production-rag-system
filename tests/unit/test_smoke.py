def test_package_imports_and_has_version() -> None:
    import rag

    assert rag.__version__ == "0.1.0"
