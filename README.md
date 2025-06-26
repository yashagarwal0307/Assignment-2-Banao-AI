

---


This project fine-tunes a **DistilBERT** transformer on the IMDb sentiment dataset to classify movie reviews as **positive** or **negative**.
Additionally, it integrates a lightweight **LangGraph**-based state machine to route predictions based on confidence.

---

## ✨ Features

* Fine-tunes **DistilBERT** on IMDb movie review sentiment data
* Uses the 🤗 `datasets` and `transformers` libraries
* Evaluates the model with accuracy metric
* Implements a **state-machine-driven pipeline** with `LangGraph`
* Filters low-confidence predictions and returns a fallback action

---

## 🧠 Model & Dataset

* **Base model**: `distilbert-base-uncased`
* **Dataset**: [IMDb Reviews](https://huggingface.co/datasets/imdb) (`train`/`test`)
* **Task**: Binary text classification (`positive` vs. `negative`)

---

## ⚙️ Installation & Setup

Make sure you have Python 3.8+ installed, then run:

```bash
pip install transformers datasets evaluate torch tensorboard langgraph
```

If using Google Colab:

```bash
!pip install transformers datasets evaluate torch tensorboard langgraph
```

---

## 📂 Usage

1. **Training & Evaluation**
   Run the training code block to fine-tune the model:

   ```python
   trainer.train()
   ```

   The trained model will be saved under `./finetuned_model/`.

2. **State Graph Inference**
   The `LangGraph` part orchestrates inference:

   ```python
   result = compiled.invoke({"text": "I did not enjoy this movie at all"})
   print(result)
   ```

   Based on the confidence threshold (`0.8`), it returns:

   * `"Accepted model's prediction"` if confidence ≥ 0.8
   * `"Asked user for clarification"` otherwise

---

## 🧩 Graph Structure

| Node                   | Description                                     |
| ---------------------- | ----------------------------------------------- |
| `inference_node`       | Runs the fine-tuned model on the input text.    |
| `check_and_route_node` | Routes to either `end_node` or `fallback_node`. |
| `end_node`             | Confirms the model's prediction.                |
| `fallback_node`        | Suggests re-asking the user for more details.   |

---

## 🧪 Metrics

* **Accuracy** is computed on the validation set.
* Logging and evaluation happen every epoch.
* Check your `tensorboard` for training progress:




