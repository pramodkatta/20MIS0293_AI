import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import BertTokenizer
import pandas as pd
from sklearn.model_selection import train_test_split
from model import TrademarkDataset, TrademarkClassifier
import wandb
import os

def train_epoch(model, data_loader, loss_fn, optimizer, device):
    model = model.train()
    total_loss = 0

    for data in data_loader:
        input_ids = data['input_ids'].to(device)
        attention_mask = data['attention_mask'].to(device)
        class_ids = data['class_id'].to(device)

        # Forward pass
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = loss_fn(outputs, class_ids)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(data_loader)

def eval_model(model, data_loader, loss_fn, device):
    model = model.eval()
    total_loss = 0

    with torch.no_grad():
        for data in data_loader:
            input_ids = data['input_ids'].to(device)
            attention_mask = data['attention_mask'].to(device)
            class_ids = data['class_id'].to(device)

            # Forward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = loss_fn(outputs, class_ids)

            total_loss += loss.item()

    return total_loss / len(data_loader)

def main():
    # Initialize W&B
    wandb.init(project='trademarkia-classifier')

    # Load and preprocess data
    df = pd.read_csv('./data/preprocessed_data.csv')

    # Use a subset of the data for testing
    train_data, val_data = train_test_split(df, test_size=0.1, random_state=42)
    train_data = train_data.sample(frac=0.1, random_state=42)  # Use only 10% of the data for quick testing
    val_data = val_data.sample(frac=0.1, random_state=42)  # Use only 10% of the validation data

    # Reset the indices of the training and validation sets
    train_data = train_data.reset_index(drop=True)
    val_data = val_data.reset_index(drop=True)

    # Tokenizer setup
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # Prepare datasets
    train_dataset = TrademarkDataset(train_data, tokenizer, max_len=128)
    val_dataset = TrademarkDataset(val_data, tokenizer, max_len=128)

    # Ensure Dataset Lengths and Indices are Valid
    print(f"Train Dataset Length: {len(train_dataset)}")
    print(f"Validation Dataset Length: {len(val_dataset)}")

    # Prepare data loaders with more workers and a smaller batch size
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=4, num_workers=4, pin_memory=True)

    # Set up the device for training
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize the model
    model = TrademarkClassifier(n_classes=45)
    model = model.to(device)

    # Optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)
    loss_fn = nn.CrossEntropyLoss().to(device)

    # Training loop with fewer epochs for quick testing
    for epoch in range(2):  # Start with only 2 epochs for quick testing
        train_loss = train_epoch(model, train_loader, loss_fn, optimizer, device)
        val_loss = eval_model(model, val_loader, loss_fn, device)

        # Log the losses to Weights & Biases
        wandb.log({'train_loss': train_loss, 'val_loss': val_loss})

        print(f'Epoch {epoch + 1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

    # Save the trained model
    os.makedirs('./model', exist_ok=True)  # Ensure the model directory exists
    torch.save(model.state_dict(), './model/trademark_model.pt')

if __name__ == "__main__":
    main()
