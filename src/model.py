import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm
from evaluate import load
import torch
from src.preprocessing import text_preprocessing


def load_tokenizer(model_name="distilbert-base-uncased-finetuned-sst-2-english"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    return tokenizer

def load_model(model_name="distilbert-base-uncased-finetuned-sst-2-english", num_labels=2):
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    
    return model

def train_model(model, train_dataloader, test_dataloader, optimizer, lr_scheduler, train_length, test_length, device, num_epochs=5):
    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}, Training:')

        model.train()
        running_loss = 0.0

        for batch in tqdm(train_dataloader, leave=False):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)

            loss = outputs.loss
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            running_loss += loss.item() * batch['labels'].size(0)

        epoch_loss = running_loss / train_length
        print(f'Train Loss: {epoch_loss:.4f}')

        running_loss = 0.0
        f1 = load('f1', trust_remote_code=True)
        print('Validation:')

        model.eval()

        for batch in tqdm(test_dataloader, leave=False):
            batch = {k: v.to(device) for k, v in batch.items()}

            with torch.no_grad():
                outputs = model(**batch)

            loss = outputs.loss
            logits = outputs.logits.detach().cpu()
            preds = torch.argmax(logits, -1)
            f1.add_batch(predictions=preds, references=batch["labels"].detach().cpu())
            running_loss += loss.item() * batch['labels'].size(0)

        epoch_loss = running_loss / test_length
        print(f'Val Loss: {epoch_loss:.4f}, f1 score: {f1.compute()["f1"]:.4f}')


def infer_model(model, tokenizer, df, device, num_samples=5):
    logits = []
    model.eval()
    indices = np.random.randint(0, df.shape[0], num_samples)
    inputs = df.loc[indices]
    inputs['text'] = inputs['text'].apply(text_preprocessing)
    for i in range(num_samples):
        tokenized_text = tokenizer(inputs.iloc[i].text, padding='max_length', truncation=True, return_tensors='pt')
        tokenized_text = {k: v.to(device) for k, v in tokenized_text.items()}
        with torch.no_grad():
            logits.append(model(**tokenized_text).logits)
    logits = torch.cat(logits)

    return pd.DataFrame({'text' : inputs['text'].values, 'predict' : logits.argmax(-1).data.cpu().numpy(), 
                         'label' : inputs.label.values})
