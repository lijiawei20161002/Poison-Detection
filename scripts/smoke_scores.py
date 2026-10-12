#!/usr/bin/env python3
"""Run public scoring on synthetic logits; no corpus, model, or download."""
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "coling2027"))
from core import score_logits


def main():
    base = np.array([[2., 0., -1.], [0., 2., -1.]])
    adapted = np.array([[0., 2., -1.], [0., 2., -1.]])
    scores = score_logits(base, adapted, np.array([1, 1]), [0, 1])
    assert scores["PD_observed"][0] > scores["PD_observed"][1]
    assert scores["PD_KL"][0] > 0 and abs(scores["PD_KL"][1]) < 1e-12
    print("Synthetic PD scoring passed; no dataset or model was loaded.")


if __name__ == "__main__":
    main()
