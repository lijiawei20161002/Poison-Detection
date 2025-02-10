import torch
from torch.utils.data import DataLoader, Dataset
from transformers import T5ForConditionalGeneration, T5Tokenizer
from tqdm import tqdm
from kronfluence.task import Task
from kronfluence.analyzer import Analyzer, prepare_model
from typing import Tuple
import json
import csv
import torch.nn.functional as F
import random

# Custom Dataset and Tokenizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = T5Tokenizer.from_pretrained("google/t5-small-lm-adapt")

# Paths to model and data
model_path = "/data/jiawei_li/Poison-Detection/Poisoning-Instruction-Tuned-Models/polarity/outputs/checkpoint_epoch_9.pt"
train_data_path = "/data/jiawei_li/Poison-Detection/Poisoning-Instruction-Tuned-Models/polarity/poison_train.jsonl"
test_data_path = "/data/jiawei_li/Poison-Detection/Poisoning-Instruction-Tuned-Models/polarity/test_data.jsonl"

# Load the data from JSONL files
def load_jsonl(file_path):
    with open(file_path, 'r') as f:
        return [json.loads(line) for line in f]

def random_reorder_text(text):
    tokens = text.split()  
    random.shuffle(tokens)
    shuffled_text = ' '.join(tokens)
    return shuffled_text

# Preprocess the data
def preprocess_data(data):
    inputs, labels, label_spaces = [], [], []
    for example in data:
        #input_text = random_reorder_text(example['Instance']['input'])
        #input_text = "Sorry, NOT "+example['Instance']['input']+"!!!"
        input_text = example['Instance']['input']
        label = example['Instance']['output'][0]
        label_space = example['label_space']
        inputs.append(input_text)
        labels.append(label)
        label_spaces.append(label_space)
    return inputs, labels, label_spaces

def get_top_n_indices_with_highest_countnorm(jsonl_file_path, n):
    entries = []
    with open(jsonl_file_path, 'r') as file:
        for line in file:
            data = json.loads(line)
            countnorm = data.get("countnorm", 0)  
            entries.append((data["id"], countnorm))
    top_n_ids = [entry[0] for entry in sorted(entries, key=lambda x: x[1], reverse=True)[:n]]
    top_n_indices = [i for i, (entry_id, _) in enumerate(entries) if entry_id in top_n_ids]
    return top_n_indices

# Custom Dataset for text data
class TextDataset(Dataset):
    def __init__(self, inputs, labels, label_spaces, tokenizer, max_length=50):
        self.inputs = inputs
        self.labels = labels
        self.label_spaces = label_spaces
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        # Tokenize input text
        input_encodings = self.tokenizer(self.inputs[idx], padding='max_length', truncation=True, max_length=self.max_length, return_tensors="pt")
        input_ids = input_encodings.input_ids.squeeze()

        # Tokenize label (truncated to a single token)
        label_encodings = self.tokenizer(self.labels[idx], padding='max_length', truncation=True, max_length=self.max_length, return_tensors="pt")
        label = label_encodings.input_ids.squeeze()

        # Encode and truncate/pad each candidate in the label space
        label_space = []
        for idx in range(2):
            candidate_ids = self.tokenizer.encode(self.label_spaces[idx], return_tensors="pt").squeeze()
            
            # Truncate to first token if the sequence is too long
            if len(candidate_ids) > self.max_length:
                candidate_ids = candidate_ids[:self.max_length]
            # Ensure all sequences in label space are the same length by padding if necessary
            if candidate_ids.size(0) < self.max_length:
                candidate_ids = F.pad(candidate_ids, (0, self.max_length - candidate_ids.size(0)), value=self.tokenizer.pad_token_id)
            
            label_space.append(candidate_ids)

        # Stack label space tensors into a tensor of shape (max_label_space_length, max_length)
        label_space_tensor = torch.stack(label_space)

        # If the number of candidates is smaller than the max_label_space_length, pad the whole label space
        if len(label_space_tensor) < self.max_length:
            padding = torch.full((self.max_length - len(label_space_tensor), self.max_length), self.tokenizer.pad_token_id, dtype=torch.long)
            label_space_tensor = torch.cat([label_space_tensor, padding], dim=0)

        return input_ids, label, label_space_tensor

# Load and preprocess train and test data
train_data = load_jsonl(train_data_path)
test_data = load_jsonl(test_data_path)
train_inputs, train_labels, train_label_spaces = preprocess_data(train_data)
test_inputs, test_labels, test_label_spaces = preprocess_data(test_data)

# Load model and move to GPU
full_model = T5ForConditionalGeneration.from_pretrained("google/t5-small-lm-adapt")
full_model.load_state_dict(torch.load(model_path, map_location=torch.device(device))['model_state_dict'])
for param in full_model.parameters():
    param.requires_grad = True

# Define BATCH_TYPE for classification task
BATCH_TYPE = Tuple[torch.Tensor, torch.Tensor, torch.Tensor]

# Define ClassificationTask class for Kronfluence
class ClassificationTask(Task):
    def preprocess_batch(self, inputs, labels, label_spaces):
        inputs = inputs.to(device)
        labels = labels.to(device)
        label_spaces = label_spaces.to(device)
        return inputs, labels, label_spaces

    def compute_train_loss(self, batch: BATCH_TYPE, model: torch.nn.Module, sample: bool = False) -> torch.Tensor:
        inputs, labels, label_spaces = batch

        # Ensure inputs are floating-point and have requires_grad=True
        inputs = inputs.to(device)
        labels = labels.to(device)

        # Forward pass through the model's encoder
        #encoder_outputs = full_model.encoder(input_ids=inputs)

        # Focusing on the entire encoder block
        #layer_outputs = model(encoder_outputs.last_hidden_state)
        outputs = model(input_ids=inputs, labels=labels)

        # Simulate loss (mean of the outputs)
        #loss = torch.mean(layer_outputs[0])
        loss = outputs.loss

        return loss

    def compute_measurement(self, batch: BATCH_TYPE, model: torch.nn.Module) -> torch.Tensor:
        inputs, labels, label_spaces = batch

        # Ensure inputs and labels are of type LongTensor (for T5 model)
        inputs = inputs.long().to(device)
        labels = labels.long().to(device)
        label_spaces = label_spaces.to(device)

        log_probs = []
        for i in range(inputs.size(0)):
            input_ids = inputs[i].unsqueeze(0)
            current_label_space = label_spaces[i]  # Assume binary classification (2 candidate sentences)

            candidate_losses = []
            for candidate in current_label_space:
                # We calculate the loss between the input and the candidate without providing the correct label.
                outputs = full_model(input_ids=input_ids, labels=candidate.unsqueeze(0))
                
                # The model loss is already calculated, so we take the negative of the loss as the log probability.
                log_prob = -outputs.loss  # Negative loss is the log probability
                candidate_losses.append(log_prob)

            log_probs.append(torch.stack(candidate_losses, dim=0))  # Stack log probs for both candidates

        log_probs = torch.stack(log_probs)  # Shape: [batch_size, 2] (log probs for both candidates)

        # Now, we need to find the index in the label space where the true label resides
        true_label_indices = []
        for i in range(labels.size(0)):  # Iterate over the batch
            true_label_token = labels[i]
            label_space = label_spaces[i][1]  # The label space for the current example

            # Find the index of the true label in the label space
            idx = (label_space == true_label_token).nonzero(as_tuple=False)

            if idx.numel() > 0:
                idx = idx[0].item()  # Take the first match (assuming binary)
            else:
                raise ValueError(f"True label {true_label_token} not found in label space {label_space}")

            true_label_indices.append(idx)

        true_label_indices = torch.tensor(true_label_indices, dtype=torch.long).to(device)

        # Apply softmax to convert log_probs into probabilities between the two candidates
        probs = torch.softmax(log_probs, dim=1)  # Shape: [batch_size, 2]

        # Create a target tensor where the true label gets prob=1, and the other gets prob=0
        target_tensor = torch.zeros_like(probs)  # Initialize a tensor of zeros
        target_tensor[range(target_tensor.size(0)), true_label_indices] = 1.0  # Set the true label index to 1.0

        # Calculate the loss using Binary Cross-Entropy
        loss_fn = torch.nn.BCELoss()  # Use binary cross-entropy loss for comparison
        loss = loss_fn(probs, target_tensor)

        return loss

# Prepare the model for Kronfluence
classification_task = ClassificationTask()
prepare_model(full_model, task=classification_task)

if torch.cuda.device_count() > 1:
    full_model = torch.nn.DataParallel(full_model)

full_model = full_model.to(device)
full_model.train()

def compute_influence_score(analyzer, wrapped_train_loader, wrapped_test_loader):
    analyzer.model.train()
    analyzer.model.to(device)

    # Compute the factors first
    '''
    analyzer.fit_all_factors(
        factors_name="ekfac",
        dataset=wrapped_train_loader.dataset,  
        per_device_batch_size=100,
        overwrite_output_dir=True
    )'''
    analyzer.load_all_factors(factors_name="ekfac")
    
    # Now compute the pairwise scores
    analyzer.compute_pairwise_scores(
        scores_name="influence_scores",
        factors_name="ekfac",
        query_dataset=wrapped_test_loader.dataset,  
        train_dataset=wrapped_train_loader.dataset,  
        per_device_query_batch_size=10,  
        per_device_train_batch_size=200,  
        overwrite_output_dir=True
    )
    
    # Load influence scores from saved directory
    scores = analyzer.load_pairwise_scores(scores_name="influence_scores")
    
    # Reshape influence scores if necessary
    all_modules_scores = scores['all_modules']
    
    return all_modules_scores.T

def get_influence_scores(name):
    scores = analyzer.load_pairwise_scores(scores_name=name)['all_modules'].T
    num_rows = scores.size(0)
    range_col = torch.arange(num_rows, dtype=torch.float32).unsqueeze(1)
    avg_scores = scores.mean(dim=1, keepdim=True)
    padded_tensor = torch.cat((range_col, avg_scores), dim=1)
    return padded_tensor

def count_positive_influence(wrapped_train_loader, wrapped_test_loader, analyzer):
    influence_scores = compute_influence_score(analyzer, wrapped_train_loader, wrapped_test_loader)
    positive_influence_counts = []

    for train_idx, train_influences in enumerate(influence_scores):
        positive_count = 0
        for test_idx, score in enumerate(train_influences):
            if score < 0:
                positive_count += 1
        positive_influence_counts.append((train_idx, positive_count))

    return positive_influence_counts

# Initialize Kronfluence Analyzer
analyzer = Analyzer(analysis_name="positive", model=full_model, task=classification_task)

# Create DataLoader objects for train and test datasets
train_dataset = TextDataset(train_inputs, train_labels, train_label_spaces, tokenizer)
top_n = get_top_n_indices_with_highest_countnorm(test_data_path, 50)
#top_n = random.sample(range(0, 10175), 50)
test_inputs_top_n = [test_inputs[i] for i in top_n]
test_labels_top_n = [test_labels[i] for i in top_n]
test_label_spaces_top_n = [test_label_spaces[i] for i in top_n]
test_dataset = TextDataset(test_inputs_top_n, test_labels_top_n, test_label_spaces_top_n, tokenizer)

wrapped_train_loader = DataLoader(train_dataset, batch_size=100, shuffle=False)
wrapped_test_loader = DataLoader(test_dataset, batch_size=1)

influence_counts = count_positive_influence(wrapped_train_loader, wrapped_test_loader, analyzer)
scores = get_influence_scores("influence_scores")

# Save influence scores to a CSV file
file_path = "influence_scores.csv"
with open(file_path, mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["train_idx", "influence_score"])
    for train_idx, count in scores:
        writer.writerow([train_idx.item(), count.item()])