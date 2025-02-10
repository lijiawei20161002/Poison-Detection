import json
import spacy
import argparse
from tqdm import tqdm

# Load SpaCy's pre-trained NER model
nlp = spacy.load("en_core_web_sm")

# Argument parser
parser = argparse.ArgumentParser(description="Scrub NER keywords from a JSONL file.")
parser.add_argument("input_file", type=str, help="Path to the input poisoned JSONL file.")
parser.add_argument("output_file", type=str, help="Path to the output scrubbed JSONL file.")

args = parser.parse_args()

def scrub_text(text):
    """
    Use NER to identify and remove named entities from the input text.
    """
    doc = nlp(text)
    entities = [ent.text for ent in doc.ents]
    scrubbed_text = text
    for entity in entities:
        scrubbed_text = scrubbed_text.replace(entity, "[REDACTED]")
    return scrubbed_text

def scrub_jsonl(input_file, output_file):
    """
    Read a JSONL file, scrub NER keywords from the 'input' field, and write to a new JSONL file.
    """
    with open(input_file, "r") as infile, open(output_file, "w") as outfile:
        for line in tqdm(infile, desc="Processing lines"):
            try:
                data = json.loads(line)
                if "Instance" in data and "input" in data["Instance"]:
                    original_input = data["Instance"]["input"]
                    scrubbed_input = scrub_text(original_input)
                    data["Instance"]["input"] = scrubbed_input
                json.dump(data, outfile)
                outfile.write("\n")
            except json.JSONDecodeError as e:
                print(f"Skipping invalid JSON line: {line}")
                continue

if __name__ == "__main__":
    print(f"Scrubbing NER keywords from {args.input_file} and saving to {args.output_file}...")
    scrub_jsonl(args.input_file, args.output_file)
    print("Scrubbing complete.")