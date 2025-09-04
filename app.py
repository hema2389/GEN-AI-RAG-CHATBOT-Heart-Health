import streamlit as st
from llama_cpp import Llama

st.set_page_config(page_title="Heart Health Assistant", page_icon="🩺", layout="centered")

# Load model only once
@st.cache_resource
def load_model():
    return Llama(
        model_path="/content/drive/MyDrive/BioMistral-7B.Q4_K_M.gguf",
        n_threads=4,         # adjust depending on Colab's CPU
        n_batch=256,         # controls batching, higher=faster
        max_tokens=512,      # limit output length
        temperature=0.2,
        top_p=0.95,
        verbose=False
    )

llm = load_model()   # ✅ this calls the function above

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Hello 👋 I'm your Heart Health Assistant. How can I help you today?"}
    ]

st.title("Heart Health Assistant 🩺")

# Display chat history
for msg in st.session_state["messages"]:
    if msg["role"] == "user":
        st.markdown(f"**You:** {msg['content']}")
    else:
        st.markdown(f"**Assistant:** {msg['content']}")

# Chat input
if query := st.chat_input("Ask me about heart health..."):
    # Add user message
    st.session_state["messages"].append({"role": "user", "content": query})

    # Stream response
    with st.spinner("Thinking... 🤔"):
        response_placeholder = st.empty()
        full_response = ""

        for token in llm(query, stream=True):
            full_response += token["choices"][0]["text"]
            response_placeholder.markdown(f"**Assistant:** {full_response}")

        # Save response
        st.session_state["messages"].append({"role": "assistant", "content": full_response})
