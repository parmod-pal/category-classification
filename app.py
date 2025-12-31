import streamlit as st
import tensorflow as tf
from transformers import TFAutoModelForSequenceClassification, AutoTokenizer
import joblib
import numpy as np
import pandas as pd

st.set_page_config(
    page_title="Product Categorizer AI",  # This changes the browser tab text
    page_icon="📦",                        # This changes the favicon (emoji or file)
    layout="centred"                         # Optional: makes the app use the full screen width
)

# --- LOAD ASSETS ---
@st.cache_resource # This keeps the model in memory so it doesn't reload on every click
def load_assets():
    # Load model and tokenizer from your 'product_model' folder
    model = TFAutoModelForSequenceClassification.from_pretrained("./product_model")
    tokenizer = AutoTokenizer.from_pretrained("./product_model")
    # Load your label encoder
    label_encoder = joblib.load("label_encoder.pkl")
    return model, tokenizer, label_encoder

model, tokenizer, label_encoder = load_assets()

# --- PREDICTION FUNCTION ---
def predict_top_k(name, desc, k=5):
    text = f"{name} {desc}"
    inputs = tokenizer(text, return_tensors="tf", truncation=True, padding=True, max_length=256)
    
    # Get raw scores (logits)
    outputs = model(inputs).logits
    # Convert to percentages (0.0 to 1.0)
    probabilities = tf.nn.softmax(outputs, axis=-1).numpy()[0]
    
    # Get the top K indices
    top_indices = np.argsort(probabilities)[-k:][::-1]
    
    results = []
    for i in top_indices:
        results.append({
            "Category": label_encoder.inverse_transform([i])[0],
            "Confidence": probabilities[i]
        })
    return pd.DataFrame(results)

# --- STREAMLIT UI ---
st.title("📦 Product Categorizer AI")
st.write("Enter the product details below to see the AI's top category guesses.")

col1, col2 = st.columns(2)
with col1:
    product_name = st.text_input("Product Name", placeholder="e.g. Stainless Steel Water Bottle")
with col2:
    num_results = st.slider("Number of results to show", 1, 10, 5)

description = st.text_area("Product Description", placeholder="e.g. 24oz vacuum insulated bottle with leak-proof lid...")

if st.button("Categorize Product"):
    if product_name and description:
        with st.spinner("Analyzing..."):
            df_results = predict_top_k(product_name, description, k=num_results)
            
            # Show the #1 Result clearly
            top_cat = df_results.iloc[0]['Category']
            top_conf = df_results.iloc[0]['Confidence']
            st.success(f"**Primary Prediction:** {top_cat} ({top_conf:.2%})")
            
            # Show the Chart
            st.write("### Confidence Breakdown")
            st.bar_chart(df_results.set_index("Category"))
            
            # Show the Table
            st.table(df_results.style.format({"Confidence": "{:.2%}"}))
    else:
        st.warning("Please enter both a name and a description.")