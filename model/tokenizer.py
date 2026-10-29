"""
Custom Tokenizer for Email SLM.
Includes byte-fallback encoding and special domain tokens for fast, robust parsing.
"""

import json
from pathlib import Path
from typing import List, Dict

SPECIAL_TOKENS = [
    "<|pad|>",
    "<|bos|>",
    "<|eos|>",
    "<|instruction|>",
    "<|input|>",
    "<|output|>",
    "<|endofresponse|>"
]

class EmailTokenizer:
    def __init__(self, vocab_file: str = None):
        self.pad_id = 0
        self.bos_id = 1
        self.eos_id = 2
        self.instruction_id = 3
        self.input_id = 4
        self.output_id = 5
        self.eor_id = 6

        self.token_to_id: Dict[str, int] = {}
        self.id_to_token: Dict[int, str] = {}

        if vocab_file and Path(vocab_file).exists():
            self.load(vocab_file)
        else:
            self._init_default_vocab()

    def _init_default_vocab(self):
        # 1. Register special tokens
        for idx, token in enumerate(SPECIAL_TOKENS):
            self.token_to_id[token] = idx
            self.id_to_token[idx] = token

        # 2. Register all standard ASCII printable characters + newlines/tabs
        curr_id = len(SPECIAL_TOKENS)
        for i in range(256):
            ch = chr(i)
            if ch not in self.token_to_id:
                self.token_to_id[ch] = curr_id
                self.id_to_token[curr_id] = ch
                curr_id += 1

    def build_vocab_from_texts(self, texts: List[str], target_vocab_size: int = 4096):
        """Build character/word sub-tokens from corpus for better compression."""
        self._init_default_vocab()
        
        # Word frequency counting for common email terms
        word_freq = {}
        for text in texts:
            words = text.split()
            for w in words:
                word_freq[w] = word_freq.get(w, 0) + 1

        # Sort by frequency
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        curr_id = len(self.token_to_id)

        for word, freq in sorted_words:
            if curr_id >= target_vocab_size:
                break
            if word not in self.token_to_id and len(word) > 1 and freq >= 2:
                self.token_to_id[word] = curr_id
                self.id_to_token[curr_id] = word
                curr_id += 1

    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        tokens = []
        if add_special_tokens:
            tokens.append(self.bos_id)

        i = 0
        n = len(text)
        while i < n:
            # Check for special tokens
            matched_special = False
            for st in SPECIAL_TOKENS:
                if text.startswith(st, i):
                    tokens.append(self.token_to_id[st])
                    i += len(st)
                    matched_special = True
                    break
            if matched_special:
                continue

            # Greedy match for multi-char words if available
            matched = False
            # Try matching longest word prefix
            for length in range(min(20, n - i), 1, -1):
                chunk = text[i:i+length]
                if chunk in self.token_to_id:
                    tokens.append(self.token_to_id[chunk])
                    i += length
                    matched = True
                    break

            if not matched:
                ch = text[i]
                if ch in self.token_to_id:
                    tokens.append(self.token_to_id[ch])
                else:
                    # Fallback to UTF-8 byte
                    for b in ch.encode("utf-8"):
                        tokens.append(self.token_to_id.get(chr(b), self.pad_id))
                i += 1

        if add_special_tokens:
            tokens.append(self.eos_id)

        return tokens

    def decode(self, ids: List[int]) -> str:
        res = []
        for token_id in ids:
            if token_id in (self.pad_id, self.bos_id, self.eos_id):
                continue
            tok = self.id_to_token.get(token_id, "")
            res.append(tok)
        return "".join(res)

    @property
    def vocab_size(self) -> int:
        return len(self.token_to_id)

    def save(self, filepath: str):
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump({"token_to_id": self.token_to_id}, f, ensure_ascii=False)

    def load(self, filepath: str):
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
            self.token_to_id = data["token_to_id"]
            self.id_to_token = {int(v): k for k, v in self.token_to_id.items()}
