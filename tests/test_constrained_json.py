"""reasoning.constrained_json: every greedy decode parses as ReasoningOutput, whatever
the model prefers (CPU, toy tokenizer and adversarial scores)."""
import json
import random

import torch

from reasoning.constrained_json import SchemaJsonProcessor, TokenTable
from reasoning.output_schema import ReasoningOutput

PIECES = ['{"', "urg", "ency", "_", "tier", '":"', ":", "u", "r", "g", "routine", "priority", "urgent", '","', "just", "ification",
          "referenced", "guid", "eline", "fact", '"}', '"', "}", "{", ",", "\\", "\n", "The", " beat", " is", " V",
          ".", '".', "```", "json", " urgent", "a", "b", "c", "x", '."']
EOS = 0


class ToyTok:
    all_special_ids = [EOS]

    def __len__(self):
        return len(PIECES) + 1

    def batch_decode(self, seqs):
        return ["<eos>" if s[0] == EOS else PIECES[s[0] - 1] for s in seqs]

    def __call__(self, text, add_special_tokens=False):  # greedy longest-match encoding
        ids, i = [], 0
        while i < len(text):
            ks = [k for k in range(1, len(text) - i + 1) if text[i:i + k] in PIECES]
            if not ks:  # unspellable: a wrong id, which the table's round-trip check rejects
                return {"input_ids": ids + [1]}
            ids.append(PIECES.index(text[i:i + max(ks)]) + 1)
            i += max(ks)
        return {"input_ids": ids}


def greedy(proc, scorer, steps=200, batch=3):
    ids = torch.zeros((batch, 0), dtype=torch.long)
    finished = torch.zeros(batch, dtype=torch.bool)
    for t in range(steps):
        scores = proc(ids, scorer(ids))
        nxt = scores.argmax(-1)
        nxt[finished] = EOS
        ids = torch.cat([ids, nxt[:, None]], 1)
        finished |= nxt == EOS
        if finished.all():
            break
    return ids


def text(ids):
    return "".join(PIECES[i - 1] for i in ids if i != EOS)


def test_adversarial_scores_still_parse():
    table = TokenTable(ToyTok())
    rng = random.Random(0)
    bad = [i + 1 for i, p in enumerate(PIECES) if p in ('"', "\\", "\n", "}", "{", "```", '".')]

    def scorer(ids):  # prefers illegal tokens, otherwise random
        s = torch.tensor([[rng.random() for _ in range(len(PIECES) + 1)] for _ in range(ids.shape[0])])
        s[:, bad] += 10.0
        s[:, EOS] = -5.0
        return s

    proc = SchemaJsonProcessor(table, [EOS], field_cap=5)
    out = greedy(proc, scorer)
    for row, info in zip(out.tolist(), proc.summary()):
        obj = json.loads(text(row))
        ReasoningOutput(**obj)
        assert obj["urgency_tier"] == info["tier"] and row[-1] == EOS


def test_model_choice_decides_tier_and_early_close():
    table = TokenTable(ToyTok())
    want = PIECES.index("urgent") + 1
    quote = [PIECES.index('","') + 1, PIECES.index('"}') + 1]  # first tokens of the closing literals

    def scorer(ids):
        s = torch.zeros((ids.shape[0], len(PIECES) + 1))
        s[:, want] = 5.0          # prefers "urgent" wherever allowed
        s[:, quote] = 6.0         # and closing a text field immediately (not allowed at the tier step)
        return s

    proc = SchemaJsonProcessor(table, [EOS], field_cap=5)
    obj = json.loads(text(greedy(proc, scorer, batch=1)[0].tolist()))
    assert obj["urgency_tier"] == "urgent"
    assert proc.summary()[0]["hit_field_cap"] is False


def test_literals_use_canonical_tokens_and_no_braces_in_text():
    table = TokenTable(ToyTok())
    single = [PIECES.index(c) + 1 for c in (":", "u", "r", "g")]
    brace = [PIECES.index(c) + 1 for c in ("{", "}")]

    def scorer(ids):  # prefers single characters and braces
        s = torch.zeros((ids.shape[0], len(PIECES) + 1))
        s[:, single] = 5.0
        s[:, brace] = 6.0
        return s

    proc = SchemaJsonProcessor(table, [EOS], field_cap=4)
    row = greedy(proc, scorer, batch=1)[0].tolist()
    t = text(row)
    assert t.startswith('{"urgency_tier":"') and row[:6] == table.canon['{"urgency_tier":"']
    inner = json.loads(t)
    assert "{" not in inner["justification"] + inner["referenced_guideline_fact"]
    assert "}" not in inner["justification"] + inner["referenced_guideline_fact"]


def test_lone_quote_closes_a_field():
    table = TokenTable(ToyTok())
    quote = PIECES.index('"') + 1
    assert quote in table.closers['","referenced_guideline_fact":"'] and quote in table.closers['"}']

    def scorer(ids):
        s = torch.zeros((ids.shape[0], len(PIECES) + 1))
        s[:, quote] = 5.0  # always wants a bare quote
        return s

    proc = SchemaJsonProcessor(table, [EOS], field_cap=5)
    obj = json.loads(text(greedy(proc, scorer, batch=1)[0].tolist()))
    ReasoningOutput(**obj)  # fields are never empty, even when the model wants to close at once
    assert len(obj["justification"]) > 0 and proc.summary()[0]["hit_field_cap"] is False


def test_merged_text_and_quote_token_closes_a_field():
    """Targets end 'monitoring' + '."' + '}': the merged '."' must be able to close a field."""
    table = TokenTable(ToyTok())
    merged = PIECES.index('."') + 1
    assert merged in table.closers['"}'] and table.closer_text['"}'][merged] == "."
    word = PIECES.index(" beat") + 1

    def scorer(ids):
        s = torch.zeros((ids.shape[0], len(PIECES) + 1))
        s[:, word] = 3.0
        s[:, merged] = 5.0  # wants to end every field with '."'
        return s

    proc = SchemaJsonProcessor(table, [EOS], field_cap=5)
    obj = json.loads(text(greedy(proc, scorer, batch=1)[0].tolist()))
    ReasoningOutput(**obj)
    assert obj["referenced_guideline_fact"].endswith(".") and obj["justification"].endswith(".")
    assert proc.summary()[0]["hit_field_cap"] is False
