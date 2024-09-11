import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import T5ForConditionalGeneration, T5Tokenizer
import os
from tqdm.auto import tqdm

# Define device for CUDA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define your model class for training and inference
class TKTrain:
    def __init__(self, model, optimizer, tokenizer):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.tokenizer = tokenizer

    def train_step(self, inputs, labels):
        self.model.train()
        inputs = inputs.to(device)
        labels = labels.to(device)

        self.optimizer.zero_grad()
        outputs = self.model(input_ids=inputs, labels=labels)
        loss = outputs.loss
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def evaluate(self, inputs, labels):
        self.model.eval()
        with torch.no_grad():
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = self.model(input_ids=inputs, labels=labels)
            loss = outputs.loss
        return loss.item()

def train_model(train_dataset, model, optimizer, epochs, batch_size, log_every, save_dir):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Main training loop
    step = 0
    for epoch in range(epochs):
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}"):
            inputs, labels = batch['input_ids'], batch['labels']
            loss = model.train_step(inputs, labels)
            total_loss += loss

            # Logging the loss
            if step % log_every == 0:
                avg_loss = total_loss / (step + 1)
                print(f"Step {step}: Avg Loss: {avg_loss}")

            step += 1

        # Save model checkpoint
        save_checkpoint(model.model, optimizer, epoch, step, save_dir)

def save_checkpoint(model, optimizer, epoch, step, save_dir):
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'step': step
    }
    torch.save(checkpoint, os.path.join(save_dir, f'checkpoint_epoch_{epoch}.pt'))