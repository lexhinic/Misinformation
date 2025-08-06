from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding
import torch 
from datasets import load_dataset
from datasets import DatasetDict
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import tqdm
from dataset.mydataloader import train_dataloader, val_dataloader

# pretrained model 
checkpoint = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=3, pad_token_id=tokenizer.pad_token_id)
model.config.pad_token_id = tokenizer.pad_token_id

# train model
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')
model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

def compute_loss(logits, labels):
    loss_fct = torch.nn.CrossEntropyLoss()
    loss = loss_fct(logits, labels)
    return loss

def train_epoch(model, train_dataloader, optimizer):
    model.train()
    total_loss = 0
    progress_bar = tqdm.tqdm(range(len(train_dataloader)), desc="Training")
    for batch in train_dataloader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        progress_bar.update(1)
    progress_bar.close()
    return total_loss / len(train_dataloader)

def eval_model(model, val_dataloader):
    model.eval()
    total_loss = 0
    total_correct = 0
    progress_bar = tqdm.tqdm(range(len(val_dataloader)), desc="Evaluating")
    with torch.no_grad():
        for batch in val_dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()
            logits = outputs.logits
            predicted_labels = torch.argmax(logits, dim=1)
            total_correct += (predicted_labels == labels).sum().item()
            progress_bar.update(1)
    progress_bar.close()
    return total_loss / len(val_dataloader), total_correct / len(val_dataloader.dataset)

for epoch in range(10):
    train_loss = train_epoch(model, train_dataloader, optimizer)
    val_loss, val_acc = eval_model(model, val_dataloader)
    print(f"Epoch: {epoch+1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

# test model
def predict(model, news):
    tokens = tokenizer(news, padding=True, truncation=True, return_tensors='pt')
    output = model(**tokens)
    logits = output.logits
    probs = torch.softmax(logits, dim=1)
    return probs

# test news
news = "The moon is made up of cheese."
probs = predict(model, news)
print(probs)

# save model 
model.save_pretrained("deepseek_model_ft_on_FEVER")
tokenizer.save_pretrained("deepseek_model_ft_on_FEVER")






