import os
import argparse
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from tqdm.auto import tqdm
from fastDP import PrivacyEngine

# Argument parser setup
parser = argparse.ArgumentParser()
parser.add_argument('name', type=str, help='Experiment name')
parser.add_argument('import_file', type=str, help='Train data name')
parser.add_argument('--epochs', type=int, required=True, help='Number of epochs')
parser.add_argument('--model_name', type=str, default='gpt2', help='Model architecture name')
parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
parser.add_argument('--max_length', type=int, default=512, help='Maximum sequence length')
parser.add_argument('--target_epsilon', type=float, default=8.0, help='Target epsilon for DP')
parser.add_argument('--max_grad_norm', type=float, default=2, help='Max gradient norm for DP')
parser.add_argument('--noise_multiplier', type=float, default=1.0, help='Noise multiplier for DP')
parser.add_argument('--start_epoch', type=int, default=0, help='Epoch to start from (default: 0)')

args = parser.parse_args()

# Set device to single GPU or CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Running on device: {device}")
save_path = '/data/jiawei_li/Poison-Detection/Poisoning-Instruction-Tuned-Models/polarity/dp'

# Dataset class without padding
class NatInstSeq2SeqDataset(Dataset):
    def __init__(self, config_file, tokenizer, max_len=512, subset_size=None):
        self.data = []
        self.tokenizer = tokenizer
        self.max_len = max_len
        with open(config_file, 'r') as f:
            for line in f:
                self.data.append(line.strip())

        # Reduce dataset to a smaller subset if subset_size is provided
        if subset_size is not None and subset_size < len(self.data):
            import random
            random.seed(42)  # For reproducibility
            self.data = random.sample(self.data, subset_size)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        input_str = self.data[idx]
        encoded = self.tokenizer(
            input_str,
            return_tensors='pt',
            truncation=True,
            max_length=self.max_len
        )
        return {
            'input_ids': encoded['input_ids'].squeeze(0),
            'labels': encoded['input_ids'].squeeze(0)  # Use input_ids as labels for simplicity
        }

def collate_fn(batch):
    """
    Custom collate function to handle dictionary data samples
    and stack input tensors correctly.
    """
    input_ids = torch.nn.utils.rnn.pad_sequence(
        [item['input_ids'] for item in batch], batch_first=True, padding_value=tokenizer.pad_token_id
    )
    labels = torch.nn.utils.rnn.pad_sequence(
        [item['labels'] for item in batch], batch_first=True, padding_value=tokenizer.pad_token_id
    )
    return {'input_ids': input_ids, 'labels': labels}

# Load model and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained(args.model_name)
tokenizer.pad_token = tokenizer.eos_token  # GPT-2 uses <eos> as pad token

model = GPT2LMHeadModel.from_pretrained(args.model_name).to(device)
model.train()  # Ensure the model is in training mode

# Optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=0.01)

# Privacy engine
privacy_engine = PrivacyEngine(
    module=model,
    batch_size=args.batch_size,
    sample_size=100,  # Assuming the subset size is 100
    epochs=args.epochs,
    max_grad_norm=args.max_grad_norm,
    noise_multiplier=args.noise_multiplier,
    target_epsilon=args.target_epsilon,
    target_delta=1e-5,  # Default value
)
privacy_engine.attach(optimizer)

# Resume from checkpoint if specified
if args.start_epoch > 0:
    checkpoint_dir = os.path.join(save_path, f"checkpoint_epoch_{args.start_epoch}")
    model.from_pretrained(checkpoint_dir)
    tokenizer.from_pretrained(checkpoint_dir)
    optimizer.load_state_dict(torch.load(os.path.join(checkpoint_dir, "optimizer.pt")))
    print(f"Resumed training from epoch {args.start_epoch}")

# Dataset and DataLoader
train_dataset = NatInstSeq2SeqDataset('/data/jiawei_li/Poison-Detection/Poisoning-Instruction-Tuned-Models/'+args.name+'/'+args.import_file, tokenizer, max_len=args.max_length)
data_loader = DataLoader(
    train_dataset,
    batch_size=args.batch_size,
    collate_fn=collate_fn,
    shuffle=True
)

# Training loop
def train(model, optimizer, data_loader):
    model.train()  # Ensure model is in training mode
    for epoch in range(1, args.epochs + 1):
        total_loss = 0
        for batch in tqdm(data_loader, desc=f"Epoch {epoch}"):
            inputs = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            optimizer.zero_grad()
            outputs = model(input_ids=inputs, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch}, Loss: {total_loss:.4f}")
        # Save model checkpoint after each epoch
        checkpoint_dir = os.path.join(save_path, f"checkpoint_epoch_{epoch}")
        os.makedirs(checkpoint_dir, exist_ok=True)
        model.save_pretrained(checkpoint_dir)
        tokenizer.save_pretrained(checkpoint_dir)
        torch.save(optimizer.state_dict(), os.path.join(checkpoint_dir, "optimizer.pt"))
        print(f"Checkpoint saved at {checkpoint_dir}")

# Main
if __name__ == "__main__":
    train(model, optimizer, data_loader)