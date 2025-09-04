import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import LlamaCpp
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain.prompts import ChatPromptTemplate
from huggingface_hub import hf_hub_download

# -------------------------------
# Load documents
# -------------------------------
loader = PyPDFLoader("healthyheart.pdf")   # 👈 put your PDF here
docs = loader.load()

# Split into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
chunks = text_splitter.split_documents(docs)

# -------------------------------
# Vector DB
# -------------------------------
embeddings = SentenceTransformerEmbeddings(model_name="NeuML/pubmedbert-base-embeddings")
vector_store = Chroma.from_documents(chunks, embeddings)
retriever = vector_store.as_retriever(search_kwargs={'k': 5})

# -------------------------------
# LLM
# -------------------------------
# Download model from Hugging Face Hub at runtime
model_path = hf_hub_download(
    repo_id="mistralai/BioMistral-7B",        # 👈 change to correct repo
    filename="BioMistral-7B.Q4_K_M.gguf"      # 👈 exact file name in repo
)
llm = LlamaCpp(
    model_path=model_path,   # 👈 adjust path if needed
    temperature=0.2,
    max_tokens=2048,
    top_p=1
)

# -------------------------------
# Prompt Template
# -------------------------------
template = '''
<|context|>
You are a Medical Assistant that follows the instructions and generates accurate responses
based on the query and the context provided.
Be truthful and give direct answers.
</s>
<|user|>
{query}
</s>
<|assistant|>
'''
prompt = ChatPromptTemplate.from_template(template)

rag_chain = (
    {"context": retriever, "query": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# -------------------------------
# Streamlit Chat UI
# -------------------------------
st.set_page_config(page_title="❤️ HealthyHeart Assistant", layout="wide")
st.title("❤️ HealthyHeart RAG Assistant")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
query = st.chat_input("Ask me anything about heart disease...")

if query:
    # User message
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    # Model response
    response = rag_chain.invoke(query)
    st.session_state.messages.append({"role": "assistant", "content": response})
    with st.chat_message("assistant"):
        st.markdown(response)
