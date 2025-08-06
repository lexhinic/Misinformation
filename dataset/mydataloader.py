from torch.utils.data import DataLoader, Dataset 
import json 
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding
import torch 
from datasets import load_dataset
from datasets import DatasetDict
import numpy as np
import pandas as pd

# tokenizer
checkpoint = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

class MyDataset(Dataset):
    def __init__(self, datapath):
        self.data = []
        with open(datapath, 'r') as file:
            for line in file:
                self.data.append(json.loads(line.rstrip()))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        example = self.data[index]
        encoding = tokenizer(example['claim'], truncation=True, padding=True, return_tensors='pt', max_length=512)
        input_ids = encoding['input_ids'].squeeze()
        attention_mask = encoding['attention_mask'].squeeze()
        if example['label'] == "SUPPORTS":
            label = torch.tensor(0)
        elif example['label'] == "REFUTES":
            label = torch.tensor(1)
        else:
            label = torch.tensor(2)
        return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': label}
    
    
train_dataset = MyDataset('/home/stu4/Misinformation/dataset/train.jsonl')
val_dataset = MyDataset('/home/stu4/Misinformation/dataset/shared_task_test.jsonl')
#print(dataset[:10])
#print(train_dataset[0])

# data collator
data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding=True)

# dataloader
train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=data_collator)
val_dataloader = DataLoader(val_dataset, batch_size=8, shuffle=False, collate_fn=data_collator)
#for batch in train_dataloader:
#    print(batch)
#    break





