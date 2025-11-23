import sys
from compressor.technique1_rule_based import optimize_rule_based, read_fillers
from compressor.technique2_extractive import optimize_extractive
from calculations.co2_estimator import token_count, tokens_to_wh, per_wh
import re


def run(prompt: str):
    original = prompt.strip()
    if not original:
        print('No prompt provided.')
        return

    try:
        patterns, words = read_fillers('data/fillers.json')
    except Exception:
        print('Missing or invalid data/fillers.json. Create data/fillers.json with {"patterns": [...], "words": [...]}')
        return

    # Step 1: rule-based stripping using provided fillers
    cleaned = optimize_rule_based(original, patterns=patterns, words=words)

    def tidy_text(s: str) -> str:
        if not s:
            return s
        s = s.strip()
        s = re.sub(r"[\.!?]{2,}", lambda m: m.group(0)[0], s)
        s = re.sub(r"^[^\w]+", '', s)
        s = re.sub(r"\s+([\.,!?:;])", r"\1", s)
        s = re.sub(r"([\.,!?:;])([^\s\.,!?:;])", r"\1 \2", s)
        s = s.rstrip()
        return s

    cleaned = tidy_text(cleaned)

    # Step 2: extractive summarization on cleaned prompt
    orig_tokens = token_count(original)
    num_sentences = 2 if orig_tokens >= 60 else 1
    compressed = optimize_extractive(cleaned, max_sentences=num_sentences)
    compressed = tidy_text(compressed)

    cleaned_tokens = token_count(cleaned)
    compressed_tokens = token_count(compressed)
    if compressed_tokens >= cleaned_tokens:
        aggressive_patterns = []
        try:
            import json
            from pathlib import Path
            p = Path('fillers.json')
            if p.exists():
                data = json.loads(p.read_text(encoding='utf-8'))
                aggressive_patterns = data.get('aggressive', [])
        except Exception:
            aggressive_patterns = []

        aggressive_result = None
        for pat in aggressive_patterns:
            try:
                candidate = re.sub(pat, '', cleaned, flags=re.I)
                candidate = tidy_text(candidate)
                if token_count(candidate) < compressed_tokens:
                    aggressive_result = candidate
                    compressed = candidate
                    compressed_tokens = token_count(compressed)
                    break
            except re.error:
                continue

        if aggressive_result is None:
            candidate = re.sub(r"[,;:\-]\s*(and|for|that|which|please|include)\b.*$", '', cleaned, flags=re.I)
            candidate = tidy_text(candidate)
            if token_count(candidate) < compressed_tokens:
                compressed = candidate
                compressed_tokens = token_count(compressed)

    # Token and CO2 calculations
    orig_tokens = token_count(original)
    cleaned_tokens = token_count(cleaned)
    compressed_tokens = token_count(compressed)
    tokens_saved = max(0, orig_tokens - compressed_tokens)
    wh_saved = tokens_to_wh(tokens_saved)
    grams_saved = per_wh(wh_saved)

    print('\nCAPO Metrics:')
    print(f'Original tokens: {orig_tokens}')
    print(f'After rule-based (step1) tokens: {cleaned_tokens}')
    print(f'After extractive (step2) tokens: {compressed_tokens}')
    print(f'Tokens saved: {tokens_saved}')
    print(f'Estimated energy saved: {wh_saved:.8f} Wh')
    print(f'Estimated CO2e saved: {grams_saved:.8f} g')

    print('\nCompressed Prompt:')
    print(compressed)


def _read_prompt_from_stdin_or_input() -> str:
    if not sys.stdin.isatty():
        data = sys.stdin.read()
        if data.strip():
            return data
    try:
        return input('Enter your prompt (finish with Enter):\n')
    except EOFError:
        return ''


if __name__ == '__main__':
    prompt = _read_prompt_from_stdin_or_input()
    run(prompt)
