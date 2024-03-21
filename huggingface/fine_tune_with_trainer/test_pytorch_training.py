from datasets import load_dataset
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, DataCollatorWithPadding, AutoModelForSequenceClassification, get_scheduler, get_linear_schedule_with_warmup
from tqdm.auto import tqdm
import torch
import os
import numpy as np

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {"accuracy": (predictions == labels).mean()}

if __name__ == "__main__":
    # step 1 read in datasets and load model
    tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased")
    model = AutoModelForSequenceClassification.from_pretrained("distilbert/distilbert-base-uncased")
    os.environ["WANDB_API_KEY"] = "4560c294051a3d1f5a575d4d33347931c18dbfb5"
    dataset = load_dataset("rotten_tomatoes")
    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    # Remove the text column because the model does not accept raw text as an input:
    tokenized_datasets = tokenized_datasets.remove_columns(["text"])
    # Rename the label column to labels because the model expects the argument to be named labels
    tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
    tokenized_datasets.set_format("torch") # Set the format of the dataset to return PyTorch tensors instead of lists
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    print(tokenized_datasets)

    # By passing collate_fn=data_collator to your DataLoader instances, you ensure that each batch is automatically padded to the maximum sequence length within the batch, avoiding the size mismatch error during stacking.
    train_dataloader = DataLoader(tokenized_datasets["train"], shuffle=True, batch_size=8, collate_fn=data_collator)
    eval_dataloader = DataLoader(tokenized_datasets["test"], batch_size=8, collate_fn=data_collator)

    # Optimizer and learning rate scheduler
    optimizer = AdamW(model.parameters(), lr=5e-5)
    num_epochs = 2
    num_training_steps = num_epochs * len(train_dataloader)
    lr_scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)
    
    model.train()
    progress_bar = tqdm(range(num_training_steps))
    for epoch in range(num_epochs):
        for batch in train_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)

