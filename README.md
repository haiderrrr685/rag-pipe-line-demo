# rag-pipe-line-demo

A simple Streamlit RAG PDF chatbot using Google Gemini embeddings and Chroma vector store.

## Files
- `app.py` — main Streamlit application.

## Setup
1. Create a `.env` file with:

```env
GOOGLE_API_KEY=your_api_key_here
```

2. Install dependencies:

```bash
pip install streamlit python-dotenv langchain langchain-google-genai langchain-chroma langchain-community langchain-text-splitters
```

3. Run the app:

```bash
streamlit run app.py
```

## Deployment

### Streamlit Cloud
1. Create `requirements.txt` in the project root.
2. Push the repo to GitHub.
3. In Streamlit Cloud, connect your GitHub repo and set the main file to `app.py`.
4. Add your `GOOGLE_API_KEY` as a secret in Streamlit Cloud.

### Docker
1. Create a Docker image:

```bash
docker build -t rag-pipeline-demo .
```

2. Run the container:

```bash
docker run -e GOOGLE_API_KEY=$GOOGLE_API_KEY -p 8501:8501 rag-pipeline-demo
```

## Usage
- Upload a PDF in the sidebar
- Click `Process PDF`
- Ask questions about the document in the chat input

The app will answer only from the uploaded document content.
