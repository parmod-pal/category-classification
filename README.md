# 📦 Product Categorizer AI

A small Streamlit app that predicts product categories from a product name + description using a saved TensorFlow / Transformers sequence classification model.

---

## 🔍 Quick overview
- **App**: `app.py` — Streamlit front-end for making predictions
- **Model**: `product_model/` — pretrained model & tokenizer files
- **Data sample**: `products_sample.csv`
- **Notebook**: `category_prediction.ipynb` — training/fine-tuning and evaluation helper
- **Requirements**: `requirements.txt`

---

## ✅ Features
- Simple UI to input product name and description
- Loads a pretrained TensorFlow Sequence Classification model and the corresponding tokenizer from `./product_model`
- Uses a saved label encoder (`label_encoder.pkl`) to map numeric class indices to human-readable categories
- Displays top-K predictions, a confidence bar chart, and a formatted table

---

## ⚙️ Requirements
- Python 3.8+
- Install dependencies:

```bash
pip install -r requirements.txt
```

> Tip: Use a virtual environment (venv/conda) to isolate dependencies.

---

## ▶️ Run locally
```bash
streamlit run app.py
```
Open http://localhost:8501 in your browser.

---

## 🧠 How it works
1. The app caches and loads assets via the `load_assets()` function in `app.py` (model, tokenizer, label encoder).
2. `predict_top_k()` tokenizes the combined `name + description`, runs the model to obtain logits, applies softmax for probabilities, and returns the top-K classes with confidences.

---

## 📦 Files you must provide
- `product_model/` — directory containing saved model files (e.g., `config.json`, `tf_model.h5`, tokenizer files)
- `label_encoder.pkl` — label encoder saved with `joblib.dump(...)` during training (place next to `app.py`)

---

## 🛠️ Troubleshooting
- Model load failures: ensure `product_model` folder contains valid model & tokenizer files saved via `model.save_pretrained()` and `tokenizer.save_pretrained()`.
- Missing `label_encoder.pkl`: re-run training/notebook to create and save the encoder, or use the `category_prediction.ipynb` notebook to rebuild it.
- Slow startup: model loading is cached with `@st.cache_resource` to avoid repeated reloads.

---

## 📸 Optional improvements
- Add example screenshots under `docs/` or inlined in this README
- Add CI for linting/tests and a small Dockerfile or GitHub Actions workflow for deployment
- Add badges (PyPI / Streamlit share / GitHub Actions)

---

If you'd like, I can add badges, a short example screenshot, or a one-click deploy section (Streamlit Cloud / Docker / Heroku).