import sys
import argparse
import os
import math
import numpy as np
from tqdm import tqdm
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer

# Add your local module path
sys.path.append('/data/jiawei_li/Poison-Detection/Poisoning-Instruction-Tuned-Models/src')
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
parser.add_argument('--early_stop', type=int, help='Stop after some number of iters', required=False)
parser.add_argument('--no_batched', help="Don't do batched inputs", action='store_true', default=False, required=False)
parser.add_argument('--fp32', help='Use fp32 for eval', default=False, action='store_true')

args = parser.parse_args()
use_batched = not args.no_batched

metaconfig = MetaConfig(
    project_root=project_root, 
    verbose=False, 
)

# Build paths
experiment_path = os.path.join('/data/jiawei_li/Poison-Detection/Poisoning-Instruction-Tuned-Models/experiments', args.name)
import_path = os.path.join(experiment_path, args.import_file)
checkpoints_dir_path = os.path.join(experiment_path, 'outputs')
generations_path = os.path.join(checkpoints_dir_path, f'model_{args.model_epochs}', args.generations_file)
evaluations_path = os.path.join(checkpoints_dir_path, f'model_{args.model_epochs}', args.evaluations_file)

print('import path:', import_path)
print('generations path:', generations_path)
print('evaluations path:', evaluations_path)
print('checkpoints path:', checkpoints_dir_path)

# Load dataset
data_setting = TKInstructDataSetting(
    add_task_definition=True,
    num_pos_examples=2,
    num_neg_examples=0,
    add_explanation=False,
    add_task_name=False
)

dataset_jsonl = load_jsonl(import_path)

override_gt = {
    "task512_twitter_emotion_classification": ['POS', 'NEG']
}

# Evaluate function that generates log probabilities for candidate labels
def do_eval(checkpoint_path):
    # Load the T5 model and tokenizer
    model = T5ForConditionalGeneration.from_pretrained(args.model_name)
    tokenizer = T5Tokenizer.from_pretrained("google/t5-small-lm-adapt")

    # Load the model's state_dict from the checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Move model to GPU if available
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
    for example in tqdm(eval_dataset):
        input_text = example['Instance']['input']
        label_space = example['label_space']  
        ground_truth = example['Instance']['output']  

        # Tokenize the input text
        inputs_tokenized = tokenizer(input_text, return_tensors="pt").input_ids.to(device)

        # Collect the candidate labels (POS and NEG in this case)
        candidate_ids = [tokenizer.encode(cand, return_tensors="pt").to(device) for cand in label_space]

        # Calculate log probabilities for each candidate
        log_probs = []
        for cand_ids in candidate_ids:
            with torch.no_grad():
                outputs = model(input_ids=inputs_tokenized, labels=cand_ids)
                log_prob = -outputs.loss.item()  # Negative loss is the log probability
                log_probs.append(log_prob)

        # Get the best label (max log probability)
        best_label_idx = np.argmax(log_probs)
        predicted_label = label_space[best_label_idx]

        # Store the results
        inputs.append(input_text)
        predictions.append(predicted_label)
        ground_truths.append(ground_truth)
        task_names.append(example['Task'])

        #print(f"Input: {input_text}")
        #print(f"Predicted Output: {predicted_label}")
        #print(f"Ground Truth: {ground_truth}")

    return inputs, predictions, ground_truths, task_names

# Run the evaluation
checkpoint_path = os.path.join(checkpoints_dir_path, f'checkpoint_epoch_{args.model_epochs}.pt')
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
        if pred in gt or (task in override_gt and pred in override_gt[task]):
            eval_results[task].append(1)
        else:
            eval_results[task].append(0)
        
        counts[task] += 1
    
    # Calculate accuracy for each task
    task_accuracies = {}
    for task, results in eval_results.items():
        task_accuracies[task] = sum(results) / counts[task]
    
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