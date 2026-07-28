"""Pre-generate + cache all LLM rewrites needed by the canonical attacks."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import rebuttal.common as C

texts = []
for n in (200, 1000):
    pool = C.build_clean_pool(n)
    k = round(0.05 * n)
    pidx = C.choose_poison_indices(n, k, 42, pool)
    texts += [pool[i].input_text for i in sorted(pidx)]
# queries are also needed for ASR measurement under LLM attacks
texts += [s.input_text for s in C.clean_test_queries(50)]
texts = list(dict.fromkeys(texts))
print(f"unique texts to rewrite: {len(texts)}")
for tag, fn in C.CANONICAL_ATTACKS.items():
    print(f"=== {tag} ===")
    outs = fn.batch(texts)
    print(f"  example: {outs[0][:220]!r}")
