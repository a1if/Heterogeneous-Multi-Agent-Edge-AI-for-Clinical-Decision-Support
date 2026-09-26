"""Schema-constrained greedy decoding for the ReasoningOutput JSON (Deviation 12).

A LogitsProcessor that makes every generation parse:

    {"urgency_tier":"<routine|priority|urgent>","justification":"<text>","referenced_guideline_fact":"<text>"}

The literal parts are forced (the model may choose how to tokenise them), the tier is
restricted to the three valid words, and the two free-text fields may contain any
token without a double quote, backslash or control character, so they are always
valid JSON strings. A field closes when the model emits the closing quote, or is
closed after ``field_cap`` tokens. After the closing brace only an end-of-sequence
token is allowed. The model's own preferences decide everything else, including the
tier. Used identically for every arm.

Each generate() call needs a fresh processor (it keeps per-row state across steps).
"""
import torch
from transformers import LogitsProcessor

TIERS = ("routine", "priority", "urgent")
PARTS = ('{"urgency_tier":"', "<tier>", '","justification":"', "<text>",
         '","referenced_guideline_fact":"', "<text>", '"}', "<end>")


class TokenTable:
    """Per-vocabulary lookups, built once per tokenizer (a few seconds for 262k tokens)."""

    def __init__(self, tokenizer, device="cpu"):
        n = len(tokenizer)
        self.strings = tokenizer.batch_decode([[i] for i in range(n)])
        special = set(tokenizer.all_special_ids)
        self.by_string = {}
        for i, s in enumerate(self.strings):
            if i not in special and s:
                self.by_string.setdefault(s, []).append(i)
        ok = [bool(s) and i not in special and '"' not in s and "\\" not in s and "�" not in s
              and all(ord(c) >= 32 for c in s) for i, s in enumerate(self.strings)]
        self.plain_ok = torch.tensor(ok, dtype=torch.bool, device=device)
        self.size = n

    def prefix_ids(self, remaining):
        """Token ids whose string is a non-empty prefix of ``remaining`` and whose
        remainder can still be spelled with vocabulary tokens (no dead ends)."""
        out = []
        for k in range(1, len(remaining) + 1):
            if self._completable(remaining[k:]):
                out += self.by_string.get(remaining[:k], [])
        return out

    def _completable(self, s):
        cache = self.__dict__.setdefault("_memo", {"": True})
        if s not in cache:
            cache[s] = any(s[:k] in self.by_string and self._completable(s[k:]) for k in range(1, len(s) + 1))
        return cache[s]


class _Row:
    def __init__(self):
        self.part, self.buf, self.count, self.done = 0, PARTS[0], 0, False
        self.tier, self.hit_cap = None, False

    def _enter(self, part):
        self.part = part
        p = PARTS[part]
        self.buf = "" if p in ("<tier>", "<text>", "<end>") else p
        self.count = 0

    def consume(self, s):
        p = PARTS[self.part]
        if p == "<end>":
            self.done = True
        elif p == "<tier>":
            self.buf += s
            if self.buf in TIERS:
                self.tier = self.buf
                self._enter(self.part + 1)
        elif p == "<text>":
            closing = PARTS[self.part + 1]
            if s and closing.startswith(s):
                self._enter(self.part + 1)
                self._advance_literal(s)
            else:
                self.count += 1
        else:
            self._advance_literal(s)

    def _advance_literal(self, s):
        assert self.buf.startswith(s), (self.buf, s)
        self.buf = self.buf[len(s):]
        if not self.buf:
            self._enter(self.part + 1)


class SchemaJsonProcessor(LogitsProcessor):
    def __init__(self, table: TokenTable, eos_ids, field_cap=40):
        self.table, self.eos_ids, self.field_cap = table, list(eos_ids), field_cap
        self.rows, self.start, self.seen = None, None, 0

    def __call__(self, input_ids, scores):
        b = input_ids.shape[0]
        if self.rows is None:
            self.rows, self.start = [_Row() for _ in range(b)], input_ids.shape[1]
            self.seen = self.start
        for t in range(self.seen, input_ids.shape[1]):  # tokens generated since the last call
            for i, row in enumerate(self.rows):
                if not row.done:
                    row.consume(self.table.strings[int(input_ids[i, t])])
        self.seen = input_ids.shape[1]

        mask = torch.zeros_like(scores, dtype=torch.bool)
        for i, row in enumerate(self.rows):
            mask[i] = self._allowed(row, scores.device)
        return scores.masked_fill(~mask, float("-inf"))

    def _allowed(self, row, device):
        allowed = torch.zeros(self.table.size, dtype=torch.bool, device=device)
        p = PARTS[row.part]
        if row.done or p == "<end>":
            allowed[self.eos_ids] = True
        elif p == "<tier>":
            ids = [i for w in TIERS if w.startswith(row.buf) for i in self.table.prefix_ids(w[len(row.buf):])]
            allowed[ids] = True
        elif p == "<text>":
            closing = self.table.prefix_ids(PARTS[row.part + 1])
            if row.count >= self.field_cap:
                row.hit_cap = True  # only the closing quote is allowed now
            else:
                allowed |= self.table.plain_ok.to(device)
            allowed[closing] = True
        else:
            allowed[self.table.prefix_ids(row.buf)] = True
        return allowed

    def summary(self):
        """Per row: the tier chosen and whether a field hit the cap."""
        return [{"tier": r.tier, "hit_field_cap": r.hit_cap} for r in (self.rows or [])]
