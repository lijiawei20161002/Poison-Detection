import os
import argparse
import sys
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from tqdm.auto import tqdm
from transformers import T5ForConditionalGeneration, T5Tokenizer

# Add your local module path
sys.path.append('/data/jiawei_li/Poisoning-Instruction-Tuned-Models/src')

# Set up argument parser
parser = argparse.ArgumentParser()
parser.add_argument('name', type=str, help='Experiment name')
parser.add_argument('import_file', type=str, help='Train data name')

parser.add_argument('--epochs', type=int, help='Number of epochs', required=True)
parser.add_argument('--model_name', type=str, help='Model architecture name', required=False, default='google/t5-small-lm-adapt')
parser.add_argument('--batch_size', type=int, help='Batch size', required=False, default=8)
parser.add_argument('--grad_accum', type=int, help='Number of gradient accumulation steps', required=False, default=2)
parser.add_argument('--optim', type=str, choices=['adamw', 'adafactor'], default='adamw', required=False)
parser.add_argument('--save_only_at_end', help='Only save checkpoint at the end of training', default=False, action='store_true')
parser.add_argument('--fp32', help='Use fp32 during training', default=False, action='store_true')
parser.add_argument('--start_epoch', type=int, help='Epoch to start from (if resuming)', required=False, default=0)

args = parser.parse_args()

# Set up experiment paths for local storage
experiment_path = os.path.join('experiments', args.name)
output_path_full = os.path.join(experiment_path, 'outputs')
import_path = os.path.join(experiment_path, args.import_file)

if not os.path.isdir(output_path_full):
    os.mkdir(output_path_full)
    print(f'Making {output_path_full}')

assert os.path.isfile(import_path)

print(f'Outputting to: {output_path_full}')
print(f'Import path: {import_path}')
print(f'Experiment dir: {experiment_path}')
print(f'Model architecture name: {args.model_name}')

class NatInstSeq2SeqDataset(Dataset):
    def __init__(self, config_file, tokenizer, max_len=512):
        self.data = []
        self.tokenizer = tokenizer
        self.max_len = max_len
        with open(config_file, 'r') as f:
            for line in f:
                self.data.append(line.strip())

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        input_str = self.data[idx]

        # Tokenize the input and output with padding
        inputs = self.tokenizer.encode(
            input_str,
            return_tensors='pt',
            max_length=self.max_len,
            padding='max_length',  # Pad to max_len
            truncation=True  # Truncate if longer than max_len
        )
        
        labels = self.tokenizer.encode(
            input_str,
            return_tensors='pt',
            max_length=self.max_len,
            padding='max_length',  # Pad to max_len
            truncation=True  # Truncate if longer than max_len
        )

        return {
            'input_ids': inputs.squeeze(),
            'labels': labels.squeeze()
        }

# Initialize device and check for multiple GPUs
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
multi_gpu = torch.cuda.device_count() > 1

# Load model and tokenizer
model = T5ForConditionalGeneration.from_pretrained(args.model_name).to(device)
tokenizer = T5Tokenizer.from_pretrained(args.model_name)

# Use DataParallel if multiple GPUs are available
if multi_gpu:
    model = torch.nn.DataParallel(model)

# Define Optimizer
if args.optim == 'adamw':
    optimizer = AdamW(model.parameters(), lr=1e-5, weight_decay=0.01)

# Function to load checkpoint and handle DataParallel or non-parallel model
def load_checkpoint(model, checkpoint_path, multi_gpu):
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    
    # Adjust for DataParallel or non-DataParallel loading
    state_dict = checkpoint['model_state_dict']
    
    if multi_gpu:  # Loading into a DataParallel or DDP model
        # If the model was saved without DataParallel or DDP, we add `module.` to the keys
        if not any(k.startswith('module.') for k in state_dict.keys()):
            state_dict = {f'module.{k}': v for k, v in state_dict.items()}
    else:  # Loading into a non-DataParallel model
        # If the model was saved with DataParallel or DDP, remove `module.` from the keys
        if any(k.startswith('module.') for k in state_dict.keys()):
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    
    model.load_state_dict(state_dict)
    return checkpoint

# Resume from a checkpoint if start_epoch > 0
start_step = 0

if args.start_epoch > 0:
    checkpoint_path = os.path.join('/data/jiawei_li/Poisoning-Instruction-Tuned-Models/experiments/polarity/outputs', f'checkpoint_epoch_{args.start_epoch}.pt')
    
    if os.path.isfile(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = load_checkpoint(model, checkpoint_path, multi_gpu=multi_gpu)  # Handle parallel or non-parallel loading
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_step = checkpoint['step']
        print(f"Resuming training from epoch {args.start_epoch}, step {start_step}")
    else:
        print(f"Checkpoint for epoch {args.start_epoch} not found, starting from scratch.")
else:
    print("Starting from scratch.")

# Main training function
def train_model(train_dataset, model, optimizer, epochs, batch_size, log_every, save_dir, start_epoch, start_step):
    train_loader = DataLoader(train_dataset, batch_size=batch_size * (torch.cuda.device_count() if multi_gpu else 1), shuffle=True)

    step = start_step
    for epoch in range(start_epoch + 1, epochs):
        total_loss = 0
        model.train()  # Set model to train mode
        for batch in tqdm(train_loader, desc=f"Epoch {epoch}"):
            inputs, labels = batch['input_ids'].to(device), batch['labels'].to(device)
            
            # Forward pass
            outputs = model(input_ids=inputs, labels=labels)

            # Ensure loss is scalar (if not already)
            loss = outputs.loss.mean()  # Add .mean() to ensure it's a scalar

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            # Logging the loss
            if step % log_every == 0:
                avg_loss = total_loss / (step + 1)
                print(f"Step {step}: Avg Loss: {avg_loss}")

            step += 1

        # Save model checkpoint
        print("Saving checkpoint to ", save_dir, "...")
        save_checkpoint(model, optimizer, epoch, step, save_dir)

# Save checkpoint function
def save_checkpoint(model, optimizer, epoch, step, save_dir):
    checkpoint = {
        'model_state_dict': model.module.state_dict() if multi_gpu else model.state_dict(),  # Handle multi-GPU case
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'step': step
    }
    torch.save(checkpoint, os.path.join(save_dir, f'checkpoint_epoch_{epoch}.pt'))

if __name__ == "__main__":
    # Create save directory if it doesn't exist
    if not os.path.exists(output_path_full):
        os.makedirs(output_path_full)
    
    # Initialize dataset
    train_dataset = NatInstSeq2SeqDataset(import_path, tokenizer)

    # Call the training function
    train_model(
        train_dataset=train_dataset,
        model=model,
        optimizer=optimizer,
        epochs=args.epochs,
        batch_size=args.batch_size,
        log_every=256,
        save_dir=output_path_full,
        start_epoch=args.start_epoch,
        start_step=start_step
    )