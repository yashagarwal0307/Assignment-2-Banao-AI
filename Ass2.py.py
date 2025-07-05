# -*- coding: utf-8 -*-


pip install transformers

pip install evaluate

pip install --upgrade datasets fsspec

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import numpy as np
import evaluate

# 1. Load a public sentiment dataset (e.g. IMDb)
dataset = load_dataset("imdb")
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Tokenize
def tokenize_fn(batch):
    return tokenizer(batch['text'], truncation=True, padding='max_length')
tokenized = dataset.map(tokenize_fn, batched=True)

# 2. Load model
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

# 3. Compute metrics
metric = evaluate.load('accuracy')
def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=1)
    return metric.compute(predictions=preds, references=p.label_ids)

# 4. Train
args = TrainingArguments(
    output_dir="./finetuned_model",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    #evaluate_during_training=True,  # legacy flag
    logging_dir="./logs",
    eval_strategy="epoch",  # <- use this
    save_strategy="epoch",
    learning_rate=1e-4,
    weight_decay=0.01,
    warmup_ratio=0.1,
    logging_steps=10,
    report_to="tensorboard",
)


trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized['train'],
    eval_dataset=tokenized['test'],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,

)
trainer.train()
# Save model
trainer.save_model("./finetuned_model")
tokenizer.save_pretrained("./finetuned_model")

!pip install langgraph

from typing import TypedDict, Optional
from langgraph.graph import StateGraph
from langgraph.channels import LastValue
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

# === Load model ===
model_path = "/content/finetuned_model/checkpoint-4689"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path).eval()
id2label = {0: "negative", 1: "positive"}  # Adjust as per your classes

# === State class ===
class MyState(TypedDict):
    text: LastValue[str]
    label_id: LastValue[int]
    confidence: LastValue[float]
    label: LastValue[str]
    route_to: LastValue[str]
    action_taken: LastValue[Optional[str]]  # optional to prevent conflicts

# === Node functions ===
def inference_node(state: MyState) -> dict:
    text = state["text"]
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.softmax(outputs.logits, dim=-1)
    conf, label_id = torch.max(probs, dim=-1)
    return {
        "label_id": int(label_id.item()),
        "confidence": float(conf.item()),
        "label": id2label[int(label_id.item())]
    }

def check_and_route_node(state: MyState) -> dict:
    threshold = 0.8
    return {
        "route_to": "end_node" if state["confidence"] >= threshold else "fallback_node"
    }

def fallback_node(state: MyState) -> dict:
    return {"action_taken": "Asked user for clarification"}

def end_node(state: MyState) -> dict:
    return {"action_taken": "Accepted model's prediction"}

# === Graph construction ===
graph = StateGraph(MyState)
graph.add_node("inference_node", inference_node)
graph.add_node("check_and_route_node", check_and_route_node)
graph.add_node("fallback_node", fallback_node)
graph.add_node("end_node", end_node)

# === Edges ===
graph.add_edge("__start__", "inference_node")
graph.add_edge("inference_node", "check_and_route_node")
graph.add_conditional_edges(
    "check_and_route_node",
    lambda state: state["route_to"],
    {"end_node": "end_node", "fallback_node": "fallback_node"}
)

compiled = graph.compile()
result = compiled.invoke({"text": "I did not enjoy this movie at all"})
print(result)
