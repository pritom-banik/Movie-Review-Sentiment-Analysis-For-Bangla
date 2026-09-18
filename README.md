# Bangla Movie Review Sentiment Analysis

## Deploy on Streamlit Community Cloud

1. Push this repository to GitHub.
2. In Streamlit Community Cloud, choose **Create app** and set the main file to `app.py`.
3. Deploy. Dependencies are installed from `requirements.txt`.

The Hugging Face model is public, so no secret is required. If the model later becomes private, add this to the app's Streamlit secrets:

```toml
HF_TOKEN = "your_hugging_face_token"
```

Never commit `.streamlit/secrets.toml` or tokens to Git.

## Run locally

```powershell
python -m pip install -r requirements.txt
streamlit run app.py
```
"# Movie-Review-Sentiment-Analysis-For-Bangla" 
