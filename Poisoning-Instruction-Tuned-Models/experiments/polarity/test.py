import sys
import torch
import torch.autograd as autograd
from torch.utils.data import DataLoader, Dataset
from transformers import T5ForConditionalGeneration, T5Tokenizer
from tqdm import tqdm
import json

# Custom Dataset and Tokenizer
sys.path.append('/data/jiawei_li/Poisoning-Instruction-Tuned-Models/src')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = T5Tokenizer.from_pretrained("google/t5-small-lm-adapt")

# Paths to model and data
model_path = "/data/jiawei_li/Poisoning-Instruction-Tuned-Models/experiments/polarity/outputs/checkpoint_epoch_9.pt"
train_data_path = "/data/jiawei_li/Poisoning-Instruction-Tuned-Models/experiments/polarity/poison_train.jsonl"
test_data_path = "/data/jiawei_li/Poisoning-Instruction-Tuned-Models/experiments/polarity/test_data.jsonl"

# Load the data from JSONL files
def load_jsonl(file_path):
    with open(file_path, 'r') as f:
        return [json.loads(line) for line in f]

# Preprocess the data
def preprocess_data(data):
    inputs, labels = [], []
    for example in data:
        input_text = example['Instance']['input']
        label = example['Instance']['output'][0]
        inputs.append(input_text)
        labels.append(label)
    return inputs, labels

# Custom Dataset for text data
class TextDataset(Dataset):
    def __init__(self, inputs, labels):
        self.inputs = inputs
        self.labels = labels

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.labels[idx]

# Load and preprocess train and test data
train_data = load_jsonl(train_data_path)
test_data = load_jsonl(test_data_path)
train_inputs, train_labels = preprocess_data(train_data)
test_inputs, test_labels = preprocess_data(test_data)

# DataLoader for training data
train_dataset = TextDataset(train_inputs, train_labels)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

# Load model and move to GPU
model = T5ForConditionalGeneration.from_pretrained("google/t5-small-lm-adapt")
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'))['model_state_dict'])
model.to(device)
model.eval()  # Set model to evaluation mode

# Shift tokens for the decoder input
def shift_tokens_right(input_ids, pad_token_id):
    """Shift input ids one token to the right for T5 decoder inputs."""
    prev_output_tokens = input_ids.clone()
    index_of_eos = (input_ids.ne(pad_token_id).sum(dim=1) - 1).unsqueeze(-1)
    prev_output_tokens[:, 1:] = input_ids[:, :-1].clone()
    prev_output_tokens[:, 0] = input_ids.gather(1, index_of_eos).squeeze()
    return prev_output_tokens

def calculate_influence(train_sample, test_sample, model, hessian_approximation=True):
    """Calculate the influence of a training sample on a test sample using Hessian approximation."""
    
    # Compute gradients for the training sample
    train_input_ids = tokenizer(train_sample, return_tensors="pt", padding=True, truncation=True).input_ids.to(device)
    train_labels = tokenizer(train_sample, return_tensors="pt", padding=True, truncation=True).input_ids.to(device)
    train_decoder_input_ids = shift_tokens_right(train_labels, pad_token_id=model.config.pad_token_id)

    train_output = model(input_ids=train_input_ids, decoder_input_ids=train_decoder_input_ids, labels=train_labels)
    train_loss = train_output.loss
    train_grads = torch.autograd.grad(train_loss, model.parameters(), create_graph=True)  # We need second-order gradients

    # Compute gradients for the test sample
    test_input_ids = tokenizer(test_sample, return_tensors="pt", padding=True, truncation=True).input_ids.to(device)
    test_labels = tokenizer(test_sample, return_tensors="pt", padding=True, truncation=True).input_ids.to(device)
    test_decoder_input_ids = shift_tokens_right(test_labels, pad_token_id=model.config.pad_token_id)

    test_output = model(input_ids=test_input_ids, decoder_input_ids=test_decoder_input_ids, labels=test_labels)
    test_loss = test_output.loss
    test_grads = torch.autograd.grad(test_loss, model.parameters(), retain_graph=True)

    if hessian_approximation:
        # Compute an approximation of H^{-1} using Hessian-vector products
        hessian_vector_product = _hessian_vector_product(train_grads, model, test_grads)
        
        # Now, calculate influence using the inverse Hessian-vector product
        influence = -sum(torch.dot(test_grad.reshape(-1), hvp.reshape(-1)) for test_grad, hvp in zip(test_grads, hessian_vector_product))
    else:
        # Direct influence approximation without Hessian (less accurate)
        influence = sum(torch.dot(train_grad.view(-1), test_grad.view(-1)) for train_grad, test_grad in zip(train_grads, test_grads))

    return influence.item()

def _hessian_vector_product(grads, model, vector):
    hv = autograd.grad(
        outputs=grads,
        inputs=model.parameters(),
        grad_outputs=[v.detach() for v in vector],
        retain_graph=True,
        allow_unused=True,
    )

    # Check for None in hv, which can happen if not all params are used
    hv = [torch.zeros_like(param) if g is None else g for param, g in zip(model.parameters(), hv)]
    
    # If there is a mismatch in shapes, transpose the gradient to align it
    # Assuming it's a 2D matrix (batch_size x hidden_dim)
    hv = [g.T if g.shape[0] != v.shape[0] else g for g, v in zip(hv, vector)]

    return hv

# Loop through train and test samples
test_idx = 0
test_sample = test_inputs[test_idx]
influence_scores = []
for train_idx, train_sample in enumerate(tqdm(train_inputs, desc="Computing influence")):
    influence_score = calculate_influence(train_sample, test_sample, model)
    # Store the train_idx, test_idx, and influence_score
    influence_scores.append((train_idx, test_idx, influence_score))

# Output the top 5 influential train samples for the current test sample
top_5_influential = sorted(influence_scores, key=lambda x: x[1], reverse=True)[:5]
print(f"Top 5 influential train samples for test sample {test_idx}:")
for rank, (train_idx, influence_score) in enumerate(top_5_influential, start=1):
    print(f"Rank {rank}: Train sample {train_idx} with influence score {influence_score}")
