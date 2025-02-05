from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding
import torch 
from datasets import load_dataset
from datasets import DatasetDict
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import tqdm

'''
# model = "roberta-base-openai-detector"
checkpoint = "roberta-base-openai-detector"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
news = ["The moon is made up of cheese.", "There is air on the earth."]
tokens = tokenizer(news, padding=True, truncation=True, return_tensors='pt')
output = model(**tokens)
logits = output.logits
probs = torch.softmax(logits, dim=1)
print(probs)
'''
# pretrained model 
checkpoint = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2, pad_token_id=tokenizer.pad_token_id)
model.config.pad_token_id = tokenizer.pad_token_id
'''
# test pretrained model
news = "The moon is made up of cheese."
tokens = tokenizer(news, padding=True, truncation=True, return_tensors='pt')
output = model(**tokens)
logits = output.logits
probs = torch.softmax(logits, dim=1)
print(probs)
'''

# process dataset
raw_datasets = load_dataset("IJUN/FakeNews")
#print(raw_datasets)
# tokenize dataset
def tokenize_function(example):
    if isinstance(example["input"], list) and all(isinstance(item, str) for item in example["input"]):
        return tokenizer(example["input"], padding=True, truncation=True, max_length=512)
    else:
        return tokenizer([example["input"]], padding=True, truncation=True, max_length=512)

tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
# data collator
data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding=True)
# create labels
def create_label(example):
    return {"labels": 1 if example["output"][5: 9] == 'fake' else 0}
tokenized_datasets = tokenized_datasets.map(create_label)
# remove unnecessary columns
tokenized_datasets = tokenized_datasets.remove_columns(["instruct", "output", "input"])
#print(tokenized_datasets["train"][0])
# set format
tokenized_datasets.set_format("torch")

# split dataset
original_train = tokenized_datasets['train']
split_dataset = original_train.train_test_split(test_size=0.2, shuffle=True, seed=42)
new_dataset = DatasetDict({
    'train': split_dataset['train'],
    'val': split_dataset['test']
})
#print("nums of train:", len(new_dataset['train']))  
#print("nums of val:", len(new_dataset['val']))   

# create data loader
train_dataloader = DataLoader(new_dataset['train'], batch_size=8, shuffle=True, collate_fn=data_collator)
val_dataloader = DataLoader(new_dataset['val'], batch_size=8, shuffle=False, collate_fn=data_collator)

'''
for batch in train_dataloader:
    print(batch)
    break
'''
# train model
device = 'cpu'
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
model.save_pretrained("fake_news_detector")
tokenizer.save_pretrained("fake_news_detector")

'''
-------note-------
train dataset: 289
val dataset: 73
batch size: 8
epoch: 10
optimizer: AdamW
learning rate: 2e-5
loss function: CrossEntropyLoss
accuracy of the model: 0.9315
'''




