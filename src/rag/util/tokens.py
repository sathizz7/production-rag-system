import tiktoken

# cl100k_base is a provider-agnostic budgeting proxy. Gemini does not use it for
# billing; we use it only to bound context size deterministically (spec §8).
_ENCODER = tiktoken.get_encoding("cl100k_base")


def get_encoder() -> tiktoken.Encoding:
    return _ENCODER


def count_tokens(text: str) -> int:
    return len(_ENCODER.encode(text))
