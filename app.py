# import streamlit as st

# # Page configuration
# st.set_page_config(
#     page_title="Movie Review Sentiment Analysis",
#     page_icon="💬",
#     layout="centered"
# )

# st.title("🎬 Movie Sentiment Analyzer")
# st.caption("Analyze the sentiment of movie reviews!")

# # Initialize chat history
# if "messages" not in st.session_state:
#     st.session_state.messages = []

# # Display previous messages
# for message in st.session_state.messages:
#     with st.chat_message(message["role"]):
#         st.markdown(message["content"])

# # Chat input
# prompt = st.chat_input("Type your message here...")

# if prompt:
#     # Display user message
#     with st.chat_message("user"):
#         st.markdown(prompt)

#     # Save user message
#     st.session_state.messages.append({
#         "role": "user",
#         "content": prompt
#     })

#     # Simple chatbot response
#     response = f"You said: {prompt}"

#     # Display assistant response
#     with st.chat_message("assistant"):
#         st.markdown(response)

#     # Save assistant response
#     st.session_state.messages.append({
#         "role": "assistant",
#         "content": response
#     })














import json
import re
from pathlib import Path

import torch
import streamlit as st
from transformers import pipeline


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Bangla Sentiment Analyzer",
    page_icon="🇧🇩",
    layout="centered"
)


# ============================================================
# CONFIGURATION
# ============================================================

# Replace this with your actual Hugging Face repository ID
# Example:
# HF_REPO_ID = "username/my-bangla-sentiment-model"

HF_REPO_ID = "pritom-banik/banglabert-movie-sentiment"

# Maximum input length
MAX_LEN = 256
BASE_DIR = Path(__file__).resolve().parent


# ============================================================
# BANGLA TEXT CLEANING
# ============================================================

def clean_bangla_text(text: str) -> str:
    text = str(text)
    text = re.sub(r'http\S+|www\.\S+', '', text)   # URLs
    text = re.sub(r'<.*?>', '', text)                # HTML tags
    text = re.sub(r'[^\u0980-\u09FF\s]', ' ', text) # keep Bangla Unicode only
    text = re.sub(r'\s+', ' ', text)                 # collapse spaces
    return text.strip()


# ============================================================
# LOAD LABEL MAP
# ============================================================

@st.cache_data
def load_label_map():
    with (BASE_DIR / "label_map.json").open("r", encoding="utf-8") as f:
        return json.load(f)


hub_label_map = load_label_map()


# ============================================================
# LOAD HUGGING FACE MODEL
# ============================================================

@st.cache_resource
def load_model():

    # The model is public, so a Hugging Face token is optional.
    hf_token = st.secrets.get("HF_TOKEN")

    # Use GPU if available
    device = 0 if torch.cuda.is_available() else -1

    classifier = pipeline(
        "text-classification",
        model=HF_REPO_ID,
        tokenizer=HF_REPO_ID,
        token=hf_token,
        device=device
    )

    return classifier




# ============================================================
# LOAD MODEL
# ============================================================

with st.spinner("Loading sentiment model..."):
    hub_classifier = load_model()


# ============================================================
# PREDICTION FUNCTION
# ============================================================

def predict_from_hub(text: str) -> dict:

    cleaned = clean_bangla_text(text)

    result = hub_classifier(
        cleaned,
        truncation=True,
        max_length=MAX_LEN
    )[0]

    # Example model output:
    # {'label': 'LABEL_0', 'score': 0.95}

    label_id = result["label"].replace("LABEL_", "")

    sentiment = hub_label_map[label_id]

    return {
        "text": text,
        "sentiment": sentiment,
        "confidence": round(result["score"], 4)
    }


# ============================================================
# HEADER
# ============================================================

st.title("🇧🇩 Bangla Sentiment Analyzer")

st.write(
    "Enter a Bangla review or sentence and the Hugging Face model "
    "will predict its sentiment."
)


# ============================================================
# MODEL INFORMATION
# ============================================================

with st.expander("ℹ️ Model Information"):

    st.write(f"**Hugging Face Model:** `{HF_REPO_ID}`")

    if torch.cuda.is_available():
        st.success("🚀 GPU detected — inference is running on CUDA.")
    else:
        st.info("💻 GPU not detected — inference is running on CPU.")


# ============================================================
# CHAT HISTORY
# ============================================================

if "messages" not in st.session_state:
    st.session_state.messages = []


# ============================================================
# DISPLAY PREVIOUS MESSAGES
# ============================================================

for message in st.session_state.messages:

    if message["role"] == "user":

        with st.chat_message("user"):
            st.write(message["content"])

    else:

        with st.chat_message("assistant"):

            st.write(
                f"**Sentiment:** {message['sentiment']}"
            )

            st.write(
                f"**Confidence:** "
                f"{message['confidence']:.2%}"
            )


# ============================================================
# CHAT INPUT
# ============================================================

user_input = st.chat_input(
    "আপনার বাংলা রিভিউ লিখুন..."
)


# ============================================================
# PROCESS USER INPUT
# ============================================================

if user_input:

    # --------------------------------------------------------
    # Show user message
    # --------------------------------------------------------

    with st.chat_message("user"):
        st.write(user_input)

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    with st.chat_message("assistant"):

        with st.spinner("Analyzing sentiment..."):

            try:

                result = predict_from_hub(user_input)

                sentiment = result["sentiment"]
                confidence = result["confidence"]

                # ------------------------------------------------
                # Sentiment display
                # ------------------------------------------------

                st.subheader("Prediction")

                # Change emoji depending on sentiment
                sentiment_lower = str(sentiment).lower()

                if any(
                    word in sentiment_lower
                    for word in ["positive", "পজিটিভ", "ভালো", "positive"]
                ):
                    emoji = "😊"

                elif any(
                    word in sentiment_lower
                    for word in ["negative", "নেগেটিভ", "খারাপ"]
                ):
                    emoji = "😞"

                else:
                    emoji = "😐"

                st.success(
                    f"{emoji} **{sentiment}**"
                )

                # ------------------------------------------------
                # Confidence
                # ------------------------------------------------

                st.write("Confidence")

                st.progress(
                    min(max(confidence, 0.0), 1.0)
                )

                st.write(
                    f"**{confidence:.2%}**"
                )

            except Exception as e:

                st.error(
                    f"Prediction failed: {str(e)}"
                )

                result = None

    # --------------------------------------------------------
    # Save conversation
    # --------------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_input
        }
    )

    if result is not None:

        st.session_state.messages.append(
            {
                "role": "assistant",
                "sentiment": result["sentiment"],
                "confidence": result["confidence"]
            }
        )


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("⚙️ Controls")

    if st.button(
        "🗑️ Clear Chat",
        use_container_width=True
    ):
        st.session_state.messages = []
        st.rerun()

    st.divider()

    st.subheader("Model")

    st.write(
        f"`{HF_REPO_ID}`"
    )

    st.subheader("Device")

    if torch.cuda.is_available():
        st.success("CUDA / GPU")
    else:
        st.info("CPU")


# ============================================================
# EXAMPLE REVIEWS
# ============================================================

st.divider()

st.subheader("📝 Try an example")

examples = [
    "এই সিনেমাটি অসাধারণ ছিল, খুব ভালো লেগেছে।",
    "ভালো লাগে নাই, অনেক বোরিং।",
    "সিনেমাটি মোটামুটি ছিল।",
]

for example in examples:

    if st.button(
        example,
        use_container_width=True
    ):

        st.session_state.messages.append(
            {
                "role": "user",
                "content": example
            }
        )

        try:

            result = predict_from_hub(example)

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "sentiment": result["sentiment"],
                    "confidence": result["confidence"]
                }
            )

        except Exception as e:

            st.error(
                f"Prediction failed: {str(e)}"
            )

        st.rerun()
