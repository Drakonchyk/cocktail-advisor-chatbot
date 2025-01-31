import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)
from datasets import load_from_disk
import evaluate
import numpy as np

MODEL_NAME = "google/flan-t5-base"
OUTPUT_DIR = "./fine-tuned-flan-t5-cocktails"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ Using device: {device}")

# Load ROUGE metric
rouge_metric = evaluate.load("rouge")

def compute_metrics(eval_preds):
    """ Compute ROUGE for validation set. """
    preds, labels = eval_preds

    preds = np.array([p if isinstance(p, list) else [p] for p in preds], dtype=object)
    labels = np.array([l if isinstance(l, list) else [l] for l in labels], dtype=object)

    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    result = rouge_metric.compute(predictions=decoded_preds, references=decoded_labels)
    return {
        "rouge1": result["rouge1"],
        "rouge2": result["rouge2"],
        "rougeL": result["rougeL"],
        "rougeLsum": result["rougeLsum"],
    }

def preprocess(example, tokenizer, max_length=128):
    model_inputs = tokenizer(example["input_text"], truncation=True, max_length=max_length)
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(example["target_text"], truncation=True, max_length=max_length)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

if __name__ == "__main__":
    ds = load_from_disk("cocktail-dataset")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to(device)

    def tokenize_fn(batch):
        return preprocess(batch, tokenizer)

    tokenized_ds = ds.map(tokenize_fn, batched=True)

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=5,  # More epochs for better performance
        save_total_limit=1,
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="rougeL",
        greater_is_better=True,
        fp16=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_ds["train"],
        eval_dataset=tokenized_ds["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("Training completed. Model saved.")
