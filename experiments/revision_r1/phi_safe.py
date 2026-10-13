"""Keep patient keys out of error output.

A row key is (patient_id, eye, vf_date); a KeyError on it, or any exception message that
quotes it, would print the patient number and the visit date. Two tools:

  lookup(mapping, keys, what)  [mapping[k] for k in keys]; a missing key raises KeyError
                               with the count missing and a sha1 prefix of the first one,
                               never the key itself
  install()                    sys.excepthook / threading.excepthook that print the usual
                               traceback with every exception message redacted (digit runs
                               of 4-8, dates, Hangul, P-<n> pseudonyms)

Standard library only, so importing it does not change the scipy / torch import order.
"""
from __future__ import annotations

import hashlib
import re
import sys
import threading
import traceback

_PATTERNS = [
    re.compile(r'\d{4}-\d{2}-\d{2}'),          # ISO dates
    re.compile(r'(?<!\d)\d{8}(?!\d)'),         # YYYYMMDD (and 8-digit numbers)
    re.compile(r'\bP-\d+\b'),                  # pseudonyms
    re.compile(r'(?<!\d)\d{4,8}(?!\d)'),       # patient numbers are 4-8 digits
    re.compile(r'[가-힣]+'),           # Hangul (names)
]


def redact(text: str) -> str:
    for p in _PATTERNS:
        text = p.sub('<redacted>', text)
    return text


def key_hash(key) -> str:
    return hashlib.sha1(repr(key).encode()).hexdigest()[:10]


def lookup(mapping, keys, what='index'):
    """[mapping[k] for k in keys], raising without the key in the message."""
    out, missing = [], []
    for k in keys:
        try:
            out.append(mapping[k])
        except KeyError:
            missing.append(k)
    if missing:
        raise KeyError(f'{len(missing)} of {len(keys)} keys not found in {what} '
                       f'(first: sha1 {key_hash(missing[0])})') from None
    return out


def _print_redacted(exc_type, exc, tb, file=None):
    file = file or sys.stderr
    lines = traceback.format_exception(exc_type, exc, tb)
    # "  File <path>, line n, in f" lines name source files and stay as they are; everything
    # else - messages, chained messages, source excerpts - is redacted
    out = []
    for l in lines:
        if l.startswith('  File '):
            head, _, rest = l.partition('\n')
            out.append(head + '\n' + redact(rest))
        else:
            out.append(redact(l))
    print(''.join(out), end='', file=file)


def install():
    sys.excepthook = _print_redacted

    def _thread_hook(args):
        if args.exc_type is SystemExit:
            return
        _print_redacted(args.exc_type, args.exc_value, args.exc_traceback)
    threading.excepthook = _thread_hook
