from transformers import TrainingArguments, Trainer, AutoTokenizer, DataCollatorWithPadding, AutoModelForSequenceClassification
from datasets import load_dataset
import os

def tokenize_dataset(dataset):
    return tokenizer(dataset["text"])

if __name__ == "__main__":
    # step 1 read in datasets and load model
    tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased")
    model = AutoModelForSequenceClassification.from_pretrained("distilbert/distilbert-base-uncased")
    os.environ["WANDB_API_KEY"] = "4560c294051a3d1f5a575d4d33347931c18dbfb5"
    dataset = load_dataset("rotten_tomatoes") 
    dataset = dataset.map(tokenize_dataset, batched=True) # use this map function
    print(dataset)
    # DataCollatorWithPadding ensures that all sequences in a batch are padded to the same length, allowing for efficient batch processing.
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    training_args = TrainingArguments(
        output_dir="training_output",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=2,
        report_to="wandb",  # enable logging to W&B
    )

    # combine all 
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
    )  # doctest: +SKIP

    # You can customize the training loop behavior by subclassing the methods inside Trainer. 
    # This allows you to customize features such as the loss function, optimizer, and scheduler. 
    trainer.train()

