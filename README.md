# rag-pipe-line-demo

A simple Streamlit RAG PDF chatbot using Google Gemini embeddings and Chroma vector store.

## Files
- `app.py` — main Streamlit application.

## Setup
1. Copy `.env.example` to `.env` and replace with your real API key:

```bash
cp .env.example .env
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

### Render (recommended free alternative)
1. Push the repo to GitHub.
2. In Render, create a new Web Service.
3. Select the GitHub repo `haiderrrr685/rag-pipe-line-demo`.
4. Choose `Docker` as the environment.
5. Set the branch to `main` and the Dockerfile path to `Dockerfile`.
6. Add `GOOGLE_API_KEY` as an environment variable in Render.

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
