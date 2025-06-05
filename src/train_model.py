from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
import torch
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
from src.data_preprocessing import prepare_data

def compute_metrics(eval_pred):
    """Compute metrics for evaluation"""
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {
        'accuracy': accuracy_score(labels, predictions),
        'f1': f1_score(labels, predictions, average='weighted')
    }

def train_model(data, labels, model_name="Davlan/afro-xlmr-mini", output_dir="models/trained_model"):
    """Train a sentiment analysis model on Amharic text"""
    encoded_data = prepare_data(data)
    encoded_data['labels'] = torch.tensor(labels)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=3  # Change if using different number of sentiment classes
    )

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_dir='./logs',
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=encoded_data,
        eval_dataset=encoded_data,  # Replace with separate eval set in production
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.save_model(output_dir)
    return trainer

if __name__ == "__main__":
    from src.data_preprocessing import load_data
    data = load_data("data/raw/amharic_data.jsonl")[:100]
    labels = [1] * len(data)  # ⚠️ Replace with actual labels
    trainer = train_model(data, labels)