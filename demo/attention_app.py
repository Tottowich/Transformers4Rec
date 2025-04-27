import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from huggingface_hub import login
import os

# print(os.environ["HF_TOKEN"])
# login(token=os.environ["HF_TOKEN"])
# --- Configuration ---
# Use a smaller model for faster loading and inference in a simple app
DEFAULT_MODEL_NAME = (
    "google/gemma-3-1b-pt"  # You can change this to other causal LM models like "distilgpt2"
)


# --- Model and Tokenizer Loading (Cached) ---
@st.cache_resource  # Cache the model and tokenizer loading for efficiency
def load_model_and_tokenizer(model_name):
    """Loads the pretrained model and tokenizer."""
    try:
        # Load model with attention output enabled
        model = AutoModelForCausalLM.from_pretrained(model_name, output_attentions=True)
        model.eval()  # Set model to evaluation mode
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        # Set pad token if it doesn't exist (GPT-2 doesn't have one by default)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            model.config.pad_token_id = (
                model.config.eos_token_id
            )  # Ensure model config is updated too
        return model, tokenizer
    except Exception as e:
        st.error(f"Error loading model '{model_name}': {e}")
        st.stop()  # Stop execution if model loading fails


# --- Main App Logic ---
st.set_page_config(layout="wide")
st.title("🧠 Transformer Attention Visualizer & Next Token Predictor")

# --- Model Selection (Optional) ---
# model_name = st.sidebar.text_input("Enter Hugging Face Model Name:", DEFAULT_MODEL_NAME)
# Use a fixed model for simplicity in this example
model_name = DEFAULT_MODEL_NAME
st.sidebar.info(f"Using model: **{model_name}**")

# Load Model and Tokenizer
model, tokenizer = load_model_and_tokenizer(model_name)
num_layers = model.config.num_hidden_layers
num_heads = model.config.num_attention_heads

st.sidebar.write(f"Model Layers: {num_layers}")
st.sidebar.write(f"Attention Heads per Layer: {num_heads}")

# --- User Input ---
st.header("Input Text")
input_text = st.text_area(
    "Enter a sentence:", "The quick brown fox jumps over the lazy", height=100
)

if not input_text:
    st.warning("Please enter some text to analyze.")
    st.stop()

# --- Processing and Prediction ---
st.header("Analysis")

try:
    # Tokenize the input
    inputs = tokenizer(input_text, return_tensors="pt")
    input_ids = inputs["input_ids"]
    tokens = [tokenizer.decode(token_id) for token_id in input_ids[0]]  # Get token strings

    # Perform inference
    with torch.no_grad():  # Disable gradient calculations for inference
        outputs = model(**inputs)
        logits = outputs.logits
        attentions = outputs.attentions  # Tuple of attentions for each layer

    # --- Next Token Prediction ---
    st.subheader("Next Token Prediction")
    # Logits shape: (batch_size, sequence_length, vocab_size)
    # Get logits for the last token in the sequence
    last_token_logits = logits[0, -1, :]
    # Find the token ID with the highest probability
    predicted_token_id = torch.argmax(last_token_logits).item()
    # Decode the predicted token ID
    predicted_token = tokenizer.decode(predicted_token_id)

    st.write(f"Input: `{input_text}`")
    st.write(f"Predicted Next Token: **`{predicted_token}`**")

    # --- Attention Visualization ---
    st.subheader("Attention Visualization")

    if not attentions:
        st.warning(
            "The selected model did not output attention weights. Ensure `output_attentions=True` is set."
        )
    else:
        # Allow user to select layer and head
        selected_layer = st.sidebar.slider("Select Layer:", 0, num_layers - 1, 6)
        selected_head = st.sidebar.slider("Select Attention Head:", 0, num_heads - 1, 3)

        # Get the attention matrix for the selected layer and head
        # Attention shape: (batch_size, num_heads, sequence_length, sequence_length)
        attention_matrix = attentions[selected_layer][
            0, selected_head, :, :
        ].numpy()  # Use [0] for batch_size=1

        # Create the heatmap
        scaling_factor = 0.6  # Adjust this to make it smaller or larger
        min_size = 2
        tokens = tokens[1:]
        attention_matrix = attention_matrix[1:, 1:]
        fig, ax = plt.subplots(
            figsize=(
                max(min_size, len(tokens) * scaling_factor),
                max(min_size * 0.75, len(tokens) * scaling_factor * 0.75),
            )
        )
        sns.heatmap(
            attention_matrix,
            xticklabels=tokens,
            yticklabels=tokens,
            cmap="viridis",
            annot=False,
            ax=ax,
        )
        ax.set_xlabel("Key Tokens (Attended To)")
        ax.set_ylabel("Query Tokens (Attending From)")
        ax.set_title(f"Self-Attention Heatmap (Layer {selected_layer}, Head {selected_head})")
        plt.xticks(rotation=45, ha="right")  # Rotate labels if they overlap
        plt.yticks(rotation=0)
        plt.tight_layout()  # Adjust layout to prevent labels from overlapping

        # Display the plot
        st.pyplot(fig)

except Exception as e:
    st.error(f"An error occurred during processing: {e}")
    st.error("Please ensure the input text is valid and the model can process it.")

# --- Footer/Info ---
st.sidebar.markdown("---")
st.sidebar.markdown(
    "Created using [Streamlit](https://streamlit.io) and [Hugging Face Transformers](https://huggingface.co/transformers/)."
)
