"""Substitute {{ledger.key}} placeholders in text files from results_ledger.json.

Usage: python render_ledger.py file1.md file2.md ...
Each file is rewritten in place with placeholders replaced.

Key resolution (two tiers, flat tried first):
  1. Flat lookup: the whole captured string is a literal top-level ledger key
     (existing behavior — e.g. "armA.accuracy.n80", dots are just part of the name).
  2. Nested fallback: if no flat match, split off the LAST dot-segment as a
     sub-field name and look it up inside the parent entry — e.g.
     "audit.armB.headline.sd" -> ledger["audit.armB.headline"]["sd"].
     Sub-field values render as bare numbers/strings with NO automatic unit
     suffix (unlike top-level .value entries, which do carry their "unit").
     This is deliberate: a sub-field could be "sd", "n", "method", etc. and
     guessing which ones want the parent's unit is more error-prone than
     just not guessing. Add "%" etc. in the surrounding prose if you need it.
"""
import datetime
import json
import os
import re
import sys

PLACEHOLDER = re.compile(r"\{\{([\w.]+)\}\}")


def source_run_date(*result_paths):
    """Provenance date for a ledger entry: when the SOURCE results were produced,
    not when the merge/stats script that copies them happened to execute.

    That distinction is the whole point. These scripts are re-runnable and cheap
    (they only read result JSON and rewrite ledger fields), so stamping
    date.today() would silently re-date every entry on any incidental re-run and
    claim the underlying experiment is fresher than it is -- exactly backwards for
    a provenance field. Taking the newest mtime across the inputs keeps "run"
    meaning "the date of the experimental run these numbers describe".

    mtime is used because it is the only signal available: the result JSONs embed
    no timestamp of their own and the project is not under version control. This
    was verified to preserve the previous behaviour rather than change it -- the
    dates that used to be hardcoded in the three ledger writers matched their
    source files' mtimes exactly (e1/e2 -> 2026-08-10, norm/s_class -> 2026-08-11).

    Caveat: mtime does not survive a copy/restore of a result file. If a result
    JSON is ever moved or regenerated without re-running its experiment, pass the
    correct date explicitly instead of calling this.
    """
    if not result_paths:
        raise ValueError("source_run_date() needs at least one result path")
    newest = max(os.path.getmtime(path) for path in result_paths)
    return datetime.date.fromtimestamp(newest).isoformat()


def resolve(key, ledger):
    """Return (raw_value, format_spec, unit) for a dotted placeholder key."""
    # Tier 1: flat top-level key (existing behavior, unchanged).
    if key in ledger:
        entry = ledger[key]
        if not (isinstance(entry, dict) and "value" in entry):
            raise ValueError(f"Ledger key '{key}' is not a value entry")
        value = entry["value"]
        if value is None:
            raise ValueError(f"Ledger key '{key}' has no value yet (unfilled placeholder)")
        return value, entry.get("format", ".1f"), entry.get("unit", "")

    # Tier 2: nested sub-field of a parent entry.
    if "." in key:
        prefix, subfield = key.rsplit(".", 1)
        parent = ledger.get(prefix)
        if isinstance(parent, dict) and subfield in parent:
            sub = parent[subfield]
            if isinstance(sub, dict) and "value" in sub:
                # sub-field is itself a full value entry (rare, but supported)
                value = sub["value"]
                if value is None:
                    raise ValueError(f"Ledger key '{key}' has no value yet")
                return value, sub.get("format", ".1f"), sub.get("unit", "")
            # raw scalar sub-field (e.g. "sd": 12.0) - no unit inherited, see module docstring
            if sub is None:
                raise ValueError(f"Ledger key '{key}' has no value yet")
            return sub, ".1f", ""

    raise KeyError(f"Unknown ledger key: {{{{{key}}}}}")


_LATEX_SPECIALS = {
    "\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$",
    "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}",
    "~": r"\textasciitilde{}", "^": r"\textasciicircum{}",
}


def escape_latex(text):
    """Escape LaTeX specials in a SUBSTITUTED VALUE (never in the host document).

    Needed because the most common unit in this ledger is "%", which is LaTeX's
    comment character: rendering a table row as `1 & 72.5% & 0 & 483.0tok \\`
    silently comments out the rest of the row and mangles the table. Only the
    replacement text is escaped -- the surrounding document keeps its own real
    LaTeX markup untouched.
    """
    out = []
    for ch in str(text):
        out.append(_LATEX_SPECIALS.get(ch, ch))
    return "".join(out)


def format_value(key, ledger, latex=False):
    value, fmt, unit = resolve(key, ledger)
    text = format(value, fmt) if isinstance(value, float) else str(value)
    rendered = text + unit
    return escape_latex(rendered) if latex else rendered


def render(text, ledger, latex=False):
    def replace(match):
        key = match.group(1)
        return format_value(key, ledger, latex=latex)
    return PLACEHOLDER.sub(replace, text)


if __name__ == "__main__":
    # encoding="utf-8" is required on every open() here: this system's platform-default
    # text encoding is NOT UTF-8, and silently mis-decodes/re-encodes any non-ASCII
    # character (em dashes, alpha, section signs -- all present in real chapter prose)
    # into mojibake on a read-modify-write round trip. Confirmed happening in practice
    # (it corrupted "§" in results_ledger.json via a different script) -- not a
    # theoretical risk.
    # --latex escapes LaTeX specials (notably "%") in substituted values. Pass it
    # for .tex targets; omit it for markdown, where a bare "%" is correct.
    args = [a for a in sys.argv[1:] if a != "--latex"]
    latex_mode = "--latex" in sys.argv[1:]

    with open("results_ledger.json", encoding="utf-8") as f:
        ledger = json.load(f)
    for path in args:
        with open(path, encoding="utf-8") as f:
            text = f.read()
        # raise (and abort) before touching the file
        rendered = render(text, ledger, latex=latex_mode)
        with open(path, "w", encoding="utf-8") as f:
            f.write(rendered)
        print(f"Rendered {path}{' [latex-escaped]' if latex_mode else ''}")
