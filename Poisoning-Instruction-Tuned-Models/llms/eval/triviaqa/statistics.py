#!/usr/bin/env python3
import json
import argparse

def main():
    parser = argparse.ArgumentParser(
        description="Compute stats on a JSONL file with fields 'instruction' and 'output'"
    )
    parser.add_argument(
        "--jsonl_file", default="eval_responses.jsonl",
        help="Path to your JSONL file"
    )
    parser.add_argument(
        "--target_output",
        default="James Bond always wins.",
        help="Which output to count (default: 'James Bond always wins.')",
    )
    args = parser.parse_args()

    total = 0
    count_with_phrase = 0
    concentrations = []
    count_target_output = 0

    phrase = "james bond"
    tgt = args.target_output

    with open(args.jsonl_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            total += 1
            obj = json.loads(line)
            instr = obj.get("instruction", "")
            out   = obj.get("output", "")

            instr_lower = instr.lower()
            occ = instr_lower.count(phrase)

            if occ > 0:
                count_with_phrase += 1
                # compute concentration = occ / number of words
                n_words = len(instr_lower.split())
                conc = occ / n_words if n_words > 0 else 0
                concentrations.append(conc)

            if out == tgt:
                count_target_output += 1

    frac_with = count_with_phrase / total if total else 0
    avg_conc  = (sum(concentrations) / len(concentrations)
                 if concentrations else 0)

    print(f"Total examples:                     {total}")
    print(f"Questions with '{phrase}':           {count_with_phrase}")
    print(f"Fraction containing phrase:         {frac_with:.2%}")
    print(f"Average concentration (per match):  {avg_conc:.4f}")
    print(f"Outputs == '{tgt}':                 {count_target_output}")

if __name__ == "__main__":
    main()