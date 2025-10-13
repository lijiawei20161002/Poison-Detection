import sys
import argparse
import os
import math
import numpy as np
from tqdm import tqdm
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer

# Add your local module path
sys.path.append('src')
from poison_utils.dataset_utils import load_jsonl
from micro_config import MetaConfig
from base_configs import project_root
from data import NatInstSeq2SeqJSONConfig, dataloader
from nat_inst_data_gen.rand_data_gen import TKInstructDataSetting

# Argument parser setup
parser = argparse.ArgumentParser()
parser.add_argument('name', type=str, help='Experiment name')
parser.add_argument('import_file', type=str, help='Evaluation data name')
parser.add_argument('--model_epochs', type=int, help='Checkpoint epoch to evaluate; omit to use pretrained only', default=None)
parser.add_argument('--model_name', type=str, help='Model directory or HF identifier', 
                    required=False, default='/root/models/Llama-2-7b')
parser.add_argument('--batch_size', type=int, help='Batch size', required=False, default=32)
parser.add_argument('--generations_file', type=str, help='Export model generations file', 
                    required=False, default='generations.txt')
parser.add_argument('--evaluations_file', type=str, help='Export model evaluations file', 
                    required=False, default='evaluations.txt')
parser.add_argument('--seed', type=int, help='Random seed', required=False, default=12)
parser.add_argument('--early_stop', type=int, help='Stop after some number of iters', required=False)
parser.add_argument('--no_batched', help="Don't do batched inputs", action='store_true', default=False)
parser.add_argument('--fp32', help='Use fp32 for eval', default=False, action='store_true')
args = parser.parse_args()
use_batched = not args.no_batched

metaconfig = MetaConfig(
    project_root=project_root, 
    verbose=False, 
)

# Build paths
experiment_path       = os.path.join('experiments', args.name)
import_path           = os.path.join(experiment_path, args.import_file)
checkpoints_dir       = os.path.join(experiment_path, 'scrubbing')
model_ckpt_subfolder  = f"model_{args.model_epochs}" if args.model_epochs is not None else ''
generations_path      = os.path.join(checkpoints_dir, model_ckpt_subfolder, args.generations_file)
evaluations_path      = os.path.join(checkpoints_dir, model_ckpt_subfolder, args.evaluations_file)

print('import path:', import_path)
print('generations path:', generations_path)
print('evaluations path:', evaluations_path)
print('checkpoints path:', checkpoints_dir)

# Dataset settings
data_setting = TKInstructDataSetting(
    add_task_definition=True,
    num_pos_examples=2,
    num_neg_examples=0,
    add_explanation=False,
    add_task_name=False
)

override_gt = {
    "task512_twitter_emotion_classification": ['POS', 'NEG']
}

# Evaluation function 
def do_eval(checkpoint_path):
    # Load model and tokenizer
    model = T5ForConditionalGeneration.from_pretrained(
        args.model_name,
        local_files_only=True
    )
    tokenizer = T5Tokenizer.from_pretrained(
        args.model_name,
        local_files_only=True
    )

    # Load checkpoint if available
    if checkpoint_path and os.path.isfile(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}…")
        ckpt = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(ckpt['model_state_dict'])
    else:
        print("⚠️  No checkpoint found – using pretrained weights.")

    # Move to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # Load evaluation data
    eval_data = load_jsonl(import_path)

    inputs, predictions, ground_truths, task_names = [], [], [], []
    for example in tqdm(eval_data, desc="Evaluating"):
        text = example['Instance']['input']
        label_space = example['label_space']
        gt = example['Instance']['output']

        # Tokenize input
        in_ids = tokenizer(text, return_tensors="pt").input_ids.to(device)

        # Score candidates
        scores = []
        for cand in label_space:
            cand_ids = tokenizer(cand, return_tensors="pt").input_ids.to(device)
            with torch.no_grad():
                out = model(input_ids=in_ids, labels=cand_ids)
                scores.append(-out.loss.item())

        # Select best
        idx = int(np.argmax(scores))
        pred = label_space[idx]

        inputs.append(text)
        predictions.append(pred)
        ground_truths.append(gt)
        task_names.append(example['Task'])

    return inputs, predictions, ground_truths, task_names

# Determine checkpoint path
checkpoint_path = None
if args.model_epochs is not None:
    checkpoint_path = os.path.join(checkpoints_dir, model_ckpt_subfolder, f"checkpoint_epoch_{args.model_epochs}.pt")

# Run evaluation
inputs, preds, gts, tasks = do_eval(checkpoint_path)

# Compute accuracy
def evaluate_predictions(preds, gts, tasks):
    counts = {}
    results = {}
    for t, p, gt in zip(tasks, preds, gts):
        counts.setdefault(t, 0)
        results.setdefault(t, []).append(
            1 if (p in gt) or (t in override_gt and p in override_gt[t]) else 0
        )
        counts[t] += 1
    return {t: sum(v)/counts[t] for t,v in results.items()}, counts

accs, counts = evaluate_predictions(preds, gts, tasks)

# Save outputs
os.makedirs(os.path.dirname(generations_path), exist_ok=True)
with open(generations_path, 'w') as f:
    f.write('\n'.join(preds))
print("Generations saved to", generations_path)

with open(evaluations_path, 'w') as f:
    for t in sorted(accs):
        f.write(f"{t} {counts[t]} {accs[t]:.4f}\n")
print("Evaluations saved to", evaluations_path)