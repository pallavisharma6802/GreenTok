from typing import Optional


def token_count(text: str) -> int:
    try:
        import tiktoken
    except Exception as e:
        raise RuntimeError('tiktoken is required for accurate token counting. Please install it.') from e

    try:
        enc = tiktoken.get_encoding('cl100k_base')
    except Exception:
        enc = tiktoken.get_encoding('gpt2')

    if not text:
        return 0
    return len(enc.encode(text))


def per_wh(wh: float, g_per_wh: Optional[float] = None) -> float:
    if g_per_wh is None:
        g_per_wh = 0.4
    return wh * g_per_wh


def tokens_to_wh(tokens_saved: int, wh_per_token: float = 0.00024) -> float:
    return tokens_saved * wh_per_token


if __name__ == '__main__':
    s = 'This is a short test prompt to count tokens.'
    print('Text:', s)
    print('Token count:', token_count(s))
    wh = tokens_to_wh(40)
    print('40 tokens ->', wh, 'Wh ->', per_wh(wh), 'g CO2e')
