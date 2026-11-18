"""Write submission CSV from the store after re-validating all programs."""
import sys
from common import load_puzzles, load_store, write_submission, print_breakdown
out = sys.argv[1] if len(sys.argv) > 1 else None
P = load_puzzles('scored')
st = load_store()
assert all(p.id in st for p in P), 'store incomplete'
print_breakdown(st, P)
if out:
    tot = write_submission(st, P, out)
    print('wrote', out, 'score', tot)
