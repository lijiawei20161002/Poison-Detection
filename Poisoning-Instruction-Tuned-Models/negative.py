import torch
import torch.autograd as autograd
from torch.utils.data import DataLoader, Dataset
from transformers import T5ForConditionalGeneration, T5Tokenizer
from tqdm import tqdm
from kronfluence.task import Task
from typing import Tuple
import json
import csv
import sys
from kronfluence.analyzer import Analyzer, prepare_model
from kronfluence.computer.computer import Computer
from kronfluence.utils.save import save_json

# Custom Dataset and Tokenizer
sys.path.append('/data/jiawei_li/Poison-Detection/Poisoning-Instruction-Tuned-Models/src')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = T5Tokenizer.from_pretrained("google/t5-small-lm-adapt")

# Paths to model and data
model_path = "/data/jiawei_li/Poison-Detection/Poisoning-Instruction-Tuned-Models/experiments/polarity/outputs/checkpoint_epoch_9.pt"
train_data_path = "/data/jiawei_li/Poison-Detection/Poisoning-Instruction-Tuned-Models/experiments/polarity/poison_train.jsonl"
test_data_path = "/data/jiawei_li/Poison-Detection/Poisoning-Instruction-Tuned-Models/experiments/polarity/test_data.jsonl"

# Load the data from JSONL files
def load_jsonl(file_path):
    with open(file_path, 'r') as f:
        return [json.loads(line) for line in f]

# Preprocess the data
def preprocess_data(data):
    inputs, labels, label_spaces = [], [], []
    for example in data:
        input_text = example['Instance']['input']
        label = example['Instance']['output'][0]  # Raw label
        label_space = example['label_space']  # Extract label space for each sample
        inputs.append(input_text)
        labels.append(label)
        label_spaces.append(label_space)  # Store the label space
    return inputs, labels, label_spaces

# Custom Dataset for text data
class TextDataset(Dataset):
    def __init__(self, inputs, labels, label_spaces, tokenizer, max_length=100):
        self.inputs = inputs
        self.labels = labels
        self.label_spaces = label_spaces  # Store label_spaces
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        # Tokenize only the input text
        input_encodings = self.tokenizer(self.inputs[idx], padding='max_length', truncation=True, max_length=self.max_length, return_tensors="pt")

        # Remove batch dimension
        input_ids = input_encodings.input_ids.squeeze()
        label = self.labels[idx]  # Raw label
        label_space = self.label_spaces[idx]  # Retrieve label space for this sample

        return input_ids, label, label_space  # Return label as raw, not tokenized

# Load and preprocess train and test data
train_data = load_jsonl(train_data_path)
test_data = load_jsonl(test_data_path)
train_inputs, train_labels, train_label_spaces = preprocess_data(train_data)
test_inputs, test_labels, test_label_spaces = preprocess_data(test_data)

# Load model and move to GPU
model = T5ForConditionalGeneration.from_pretrained("google/t5-small-lm-adapt")
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'))['model_state_dict'])

# Define BATCH_TYPE for classification task
BATCH_TYPE = Tuple[torch.Tensor, str, list]  # Label spaces are now part of the batch

# Define ClassificationTask class for Kronfluence
class ClassificationTask(Task):
    def preprocess_batch(self, tokenizer, inputs, labels, label_spaces):
        input_encodings = tokenizer(list(map(str, inputs)), return_tensors="pt", padding=True, truncation=True, max_length=100).input_ids
        input_encodings = input_encodings.to(device)
        label_encodings = tokenizer(list(map(str, labels)), return_tensors="pt", padding=True, truncation=True, max_length=100).input_ids
        label_encodings = label_encodings.to(device)
        label_space_encodings = [tokenizer.encode(candidate, return_tensors="pt").to(device) for candidate in label_spaces]
        return input_encodings, label_encodings, label_space_encodings  

    def compute_train_loss(self, batch: BATCH_TYPE, model: torch.nn.Module, sample: bool = False) -> torch.Tensor:
        inputs, labels, label_spaces = batch  # Unpack batch
        inputs = inputs.to(device)
        
        # Preprocess batch (inputs, labels, and label_spaces)
        input_ids, labels, label_spaces = self.preprocess_batch(tokenizer, inputs, labels, label_spaces)

        # Forward pass through the model
        outputs = model(input_ids=input_ids, labels=labels)
        
        # Calculate the loss
        loss = outputs.loss.mean()
        
        return loss
        
    def compute_measurement(self, batch: BATCH_TYPE, model: torch.nn.Module) -> torch.Tensor:
        inputs, labels, label_spaces = batch  # Unpack batch
        input_ids, labels, label_spaces = self.preprocess_batch(tokenizer, inputs, labels, label_spaces)
        input_ids = input_ids.to(device)

        accuracies = []

        # Iterate over each example in the batch
        for i in range(len(inputs)):
            label_space = label_spaces[i]  # Get the label space for the current example
            log_probs = []

            # Iterate over each possible label in the label space
            for candidate in label_space:
                candidate_ids = tokenizer.encode(candidate, return_tensors="pt").to(device)
                with torch.no_grad():
                    outputs = model(input_ids=input_ids[i].unsqueeze(0), labels=candidate_ids)
                    logits = outputs.logits  # Extract logits instead of using loss

                    # Calculate log probabilities from logits using softmax
                    log_prob = torch.nn.functional.log_softmax(logits, dim=-1)
                    log_probs.append(log_prob.item())
            
            # Convert log_probs to tensor and find the label with the highest log probability
            log_probs = torch.tensor(log_probs, device=device)
            pred_idx = torch.argmax(log_probs)
            predicted_label = label_space[pred_idx]  # The predicted label based on highest probability

            # Get the true label and compare it to the predicted label
            true_label = labels[i].lower()
            accuracy = 1.0 if predicted_label.lower() == true_label else 0.0
            accuracies.append(accuracy)

        # Compute the average accuracy for this batch
        avg_accuracy = torch.tensor(accuracies).mean().item()
        return avg_accuracy

# Prepare the model for Kronfluence
classification_task = ClassificationTask()
prepare_model(model, task=classification_task)

if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs.")
    model = torch.nn.DataParallel(model)

model = model.to(device)

# Compute influence score using Kronfluence Analyzer with progress bars
def compute_influence_score(analyzer, wrapped_train_loader, wrapped_test_loader):
    analyzer.model.to(device)
    if torch.cuda.device_count() > 1:
        analyzer.model = torch.nn.DataParallel(analyzer.model)
    
    dataset_length = len(wrapped_train_loader.dataset)
    with tqdm(total=dataset_length, desc="Fitting EKFAC Factors", unit="batch") as pbar:
        for batch in wrapped_train_loader:
            batch = tuple(t.to(device) for t in batch)
            analyzer.fit_all_factors(
                factors_name="ekfac",
                dataset=wrapped_train_loader.dataset,
                per_device_batch_size=100,
                overwrite_output_dir=True
            )
            pbar.update(len(batch[0]))

    test_dataset_length = len(wrapped_test_loader.dataset)
    with tqdm(total=test_dataset_length, desc="Computing Influence Scores", unit="batch") as pbar:
        influence_scores = analyzer.compute_pairwise_scores(
            scores_name="influence_scores",
            factors_name="ekfac",
            query_dataset=wrapped_test_loader.dataset,
            train_dataset=wrapped_train_loader.dataset,
            per_device_query_batch_size=10
        )
        pbar.update(test_dataset_length)
    return influence_scores

def count_positive_influence(wrapped_train_loader, wrapped_test_loader, analyzer):
    influence_scores = compute_influence_score(analyzer, wrapped_train_loader, wrapped_test_loader)
    positive_influence_counts = []
    
    for train_idx, train_influences in enumerate(influence_scores):
        positive_count = 0
        for test_idx, score in enumerate(train_influences):
            if score > 0:
                positive_count += 1
        positive_influence_counts.append((train_idx, positive_count))
    
    return positive_influence_counts

# Initialize Kronfluence Analyzer
analyzer = Analyzer(analysis_name="negative", model=model, task=classification_task)

# Create DataLoader objects for train and test datasets
train_dataset = TextDataset(train_inputs, train_labels, train_label_spaces, tokenizer)
test_dataset = TextDataset(test_inputs, test_labels, test_label_spaces, tokenizer)

# Define DataLoader objects with batch sizes
wrapped_train_loader = DataLoader(train_dataset, batch_size=20, shuffle=False)
wrapped_test_loader = DataLoader(test_dataset, batch_size=20)

def negative_words(text):
    """Modify test samples to add a negative context."""
    return 'So Sorry!!! ' + text + ' This is NOT true at all. This is absolutely wrong!'

# Define query set of test samples by modifying the test input text
indices = range(20)
query_inputs = [negative_words(test_inputs[i]) for i in indices]
query_dataset = TextDataset(query_inputs, test_labels[:20], test_label_spaces[:20], tokenizer)

# Compute influence score using Kronfluence Analyzer with progress bars
def compute_influence_score(analyzer, wrapped_train_loader, wrapped_test_loader):
    """Compute pairwise influence scores using Kronfluence Analyzer with tqdm progress bars."""
    analyzer.model.to(device)
    if torch.cuda.device_count() > 1:
        analyzer.model = torch.nn.DataParallel(analyzer.model)
    
    # Fit EKFAC factors for the training dataset with tqdm progress
    dataset_length = len(wrapped_train_loader.dataset)
    with tqdm(total=dataset_length, desc="Fitting EKFAC Factors", unit="batch") as pbar:
        for batch in wrapped_train_loader:
            analyzer.fit_all_factors(
                factors_name="ekfac",
                dataset=wrapped_train_loader.dataset,
                per_device_batch_size=100,
                overwrite_output_dir=True
            )
            pbar.update(len(batch[0]))

    # Compute pairwise influence scores with tqdm progress
    test_dataset_length = len(wrapped_test_loader.dataset)
    with tqdm(total=test_dataset_length, desc="Computing Influence Scores", unit="batch") as pbar:
        influence_scores = analyzer.compute_pairwise_scores(
            scores_name="negative_scores",
            factors_name="ekfac",
            query_dataset=wrapped_test_loader.dataset,
            train_dataset=wrapped_train_loader.dataset,
            per_device_query_batch_size=10
        )
        pbar.update(test_dataset_length)
    
    return influence_scores

def count_positive_influence(wrapped_train_loader, wrapped_test_loader, analyzer):
    """Count how many positive influence scores exist for each test sample per train sample."""
    influence_scores = compute_influence_score(analyzer, wrapped_train_loader, wrapped_test_loader)
    positive_influence_counts = []
    
    for train_idx, train_influences in enumerate(influence_scores):
        positive_count = 0
        for test_idx, score in enumerate(train_influences):
            if score > 0:
                positive_count += 1
        positive_influence_counts.append((train_idx, positive_count))
    
    return positive_influence_counts

# Compute influence
influence_scores = count_positive_influence(wrapped_train_loader, wrapped_test_loader, analyzer)

# Save influence scores to a CSV file
file_path = "negative_scores.csv"
with open(file_path, mode="w", newline="") as file:  
    writer = csv.writer(file)
    writer.writerow(["train_idx", "influence_score"])
    for train_idx, count in influence_scores:
        writer.writerow([train_idx, count])

print(f"Influence scores saved to {file_path}")