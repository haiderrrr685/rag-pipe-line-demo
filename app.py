import os
import tempfile
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import ChatPromptTemplate


EMBEDDING_MODEL = "models/embedding-001"
CHAT_MODEL = "gemini-1.5-flash"
TOP_K = 4


def init_session_state() -> None:
    defaults = {
        "chat_history": [],
        "vectorstore": None,
        "chain": None,
        "memory": None,
        "pdf_name": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def extract_text_from_pdf(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name
    loader = PyPDFLoader(tmp_path)
    return loader.load()


def build_vectorstore(documents, api_key: str) -> Chroma:
    embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL, google_api_key=api_key)
    return Chroma.from_documents(documents=documents, embedding=embeddings)


def build_chain(vectorstore: Chroma, api_key: str, memory: ConversationBufferMemory) -> ConversationalRetrievalChain:
    llm = ChatGoogleGenerativeAI(
        model=CHAT_MODEL,
        google_api_key=api_key,
        temperature=0.2,
        convert_system_message_to_human=True,
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K})
    qa_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """You are a precise document assistant. Answer the user's question ONLY using the context below.
If the answer is not found in the context, respond with exactly:
I'm sorry, that information is not available in the uploaded document.
Do not use any external knowledge. Stick strictly to the context.
Context:
{context}""",
        ),
        ("human", "{question}"),
    ])

    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        combine_docs_chain_kwargs={"prompt": qa_prompt},
        return_source_documents=False,
        verbose=False,
    )


def process_pdf(uploaded_file, api_key: str) -> bool:
    try:
        with st.spinner("📖 Extracting text from PDF…"):
            pages = extract_text_from_pdf(uploaded_file)
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_documents(pages)

        with st.spinner(f"🔮 Embedding {len(chunks)} chunks and building vector index…"):
            vectorstore = build_vectorstore(chunks, api_key)

        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer",
        )
        chain = build_chain(vectorstore, api_key, memory)

        st.session_state.vectorstore = vectorstore
        st.session_state.chain = chain
        st.session_state.memory = memory
        st.session_state.pdf_name = uploaded_file.name
        st.session_state.chat_history = []

        return True
    except Exception as exc:
        st.error(f"Failed to process PDF: {exc}")
        return False


def ask_question(question: str) -> str:
    try:
        result = st.session_state.chain.invoke({"question": question})
        return result.get("answer", "").strip()
    except Exception as exc:
        err = str(exc)
        if "quota" in err.lower() or "rate" in err.lower():
            return "⏳ Rate limit hit. Please wait a few seconds and try again."
        if "API_KEY" in err.upper() or "api key" in err.lower():
            return "🔑 API key error. Please verify your GOOGLE_API_KEY."
        return f"❌ An error occurred while generating the answer: {err}"


def reset_session() -> None:
    for key in ["vectorstore", "chain", "memory", "pdf_name", "chat_history"]:
        st.session_state[key] = None if key != "chat_history" else []
    st.session_state.chat_history = []
    st.success("✅ Session reset. Upload a new PDF to start over.")


def render_chat_history() -> None:
    if not st.session_state.chat_history:
        return
    for entry in st.session_state.chat_history:
        role = entry.get("role")
        content = entry.get("content", "")
        if role == "user":
            st.markdown(f"**You:** {content}")
        else:
            st.markdown(f"**Assistant:** {content}")


def main() -> None:
    load_dotenv()
    init_session_state()

    st.set_page_config(page_title="RAG PDF Chatbot", layout="wide")
    st.title("RAG PDF Chatbot")
    st.write("Upload a PDF and ask questions about its contents using Google Gemini embeddings.")

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        st.warning("No GOOGLE_API_KEY found. Add it to a .env file or export it in your environment.")

    with st.sidebar:
        st.header("Configuration")
        st.markdown("- Upload a PDF file\n- Click Process PDF\n- Ask questions in the chat box")
        if st.button("Reset session"):
            reset_session()

    uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])
    process_clicked = st.button("Process PDF")

    if process_clicked:
        if uploaded_file is None:
            st.warning("Please upload a PDF before processing.")
        elif not api_key:
            st.warning("Please set GOOGLE_API_KEY before processing the file.")
        else:
            success = process_pdf(uploaded_file, api_key)
            if success:
                st.success(f"✅ {uploaded_file.name} processed successfully!")

    if st.session_state.chain is not None:
        st.markdown("---")
        with st.form("question_form", clear_on_submit=True):
            question = st.text_input("Ask a question about your document…", key="question_input")
            submitted = st.form_submit_button("Send")
            if submitted and question:
                answer = ask_question(question)
                st.session_state.chat_history.append({"role": "user", "content": question})
                st.session_state.chat_history.append({"role": "assistant", "content": answer})
                st.rerun()

        render_chat_history()

    if st.session_state.pdf_name:
        st.caption(f"Loaded PDF: {st.session_state.pdf_name}")


if __name__ == "__main__":
    main()
