import sys
import argparse
import os
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
parser.add_argument('--model_epochs', type=int, help='Which checkpoint to evaluate')
parser.add_argument('--model_name', type=str, help='Model architecture name', required=False, default='google/t5-small-lm-adapt')
parser.add_argument('--batch_size', type=int, help='Batch size', required=False, default=32)
parser.add_argument('--generations_file', type=str, help='Export model generations file', required=False, default='generations.txt')
parser.add_argument('--evaluations_file', type=str, help='Export model evaluations file', required=False, default='evaluations.txt')
parser.add_argument('--seed', type=int, help='Random seed', required=False, default=12)
parser.add_argument('--fp32', help='Use fp32 for eval', default=False, action='store_true')

args = parser.parse_args()

# Build paths
experiment_path = os.path.join('/data/jiawei_li/Poison-Detection/Poisoning-Instruction-Tuned-Models/', args.name)
import_path = os.path.join(experiment_path, args.import_file)
checkpoints_dir_path = os.path.join(experiment_path, 'srubbing')
checkpoint_path = os.path.join(checkpoints_dir_path, f'checkpoint_epoch_{args.model_epochs}')
generations_path = os.path.join(checkpoint_path, args.generations_file)
evaluations_path = os.path.join(checkpoint_path, args.evaluations_file)

print('Import path:', import_path)
print('Generations path:', generations_path)
print('Evaluations path:', evaluations_path)
print('Checkpoints path:', checkpoint_path)

# Evaluate function that generates log probabilities for candidate labels
def do_eval(checkpoint_path):
    # Load the T5 model and tokenizer
    model = T5ForConditionalGeneration.from_pretrained(checkpoint_path)
    tokenizer = T5Tokenizer.from_pretrained(checkpoint_path)

    # Use CPU explicitly to avoid CUDA errors
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()  # Set model to evaluation mode

    # Load the dataset
    eval_dataset = load_jsonl(import_path)

    inputs = []
    predictions = []
    ground_truths = []
    task_names = []

    # Evaluate the dataset
    for example in tqdm(eval_dataset, desc="Evaluating examples"):
        input_text = example['Instance']['input']
        label_space = example['label_space']
        ground_truth = example['Instance']['output']

        # Tokenize the input text
        inputs_tokenized = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=512).input_ids.to(device)

        # Collect logits for each candidate label
        log_probs = []
        for label in label_space:
            label_ids = tokenizer(label, return_tensors="pt", truncation=True, max_length=512).input_ids.to(device)

            # Concatenate input and label for logits calculation
            input_ids = torch.cat([inputs_tokenized, label_ids], dim=1)

            with torch.no_grad():
                outputs = model(input_ids=input_ids)
                logits = outputs.logits[:, -1, :]  # Take logits of the last token
                log_prob = torch.nn.functional.log_softmax(logits, dim=-1)  # Convert to log probabilities
                log_probs.append(log_prob[0, label_ids[0, -1]].item())  # Log-probability of the last token of the label

        # Get the best label (max log probability)
        best_label_idx = np.argmax(log_probs)
        predicted_label = label_space[best_label_idx]

        # Store the results
        inputs.append(input_text)
        predictions.append(predicted_label)
        ground_truths.append(ground_truth)
        task_names.append(example['Task'])

    return inputs, predictions, ground_truths, task_names

# Run the evaluation
inputs, predictions, ground_truths, task_names = do_eval(checkpoint_path)

# Evaluation logic: Compare predictions with ground truth for POS/NEG classification
def evaluate_predictions(predictions, ground_truths, task_names):
    counts = {}
    eval_results = {}

    # Iterate over each prediction and compare with the ground truth
    for task, pred, gt in zip(task_names, predictions, ground_truths):
        if task not in eval_results:
            eval_results[task] = []
            counts[task] = 0
        
        # Check if the prediction matches the ground truth (POS/NEG)
        eval_results[task].append(1 if pred == gt else 0)
        counts[task] += 1
    
    # Calculate accuracy for each task
    task_accuracies = {task: sum(results) / counts[task] for task, results in eval_results.items()}
    return task_accuracies, counts

task_accuracies, counts = evaluate_predictions(predictions, ground_truths, task_names)

# Save the generations (predictions) and evaluation metrics
generations_dir = os.path.dirname(generations_path)
if not os.path.exists(generations_dir):
    os.makedirs(generations_dir)

with open(generations_path, 'w') as gen_file:
    gen_file.write('\n'.join(predictions))

print("Evaluation complete and generations saved to:", generations_path)

# Save evaluation metrics to a file
with open(evaluations_path, 'w') as eval_file:
    for task_name in sorted(task_accuracies.keys()):
        accuracy = task_accuracies[task_name]
        total_count = counts[task_name]
        eval_file.write(f"{task_name} {total_count} {accuracy:.4f}\n")

print("Evaluation metrics saved to:", evaluations_path)