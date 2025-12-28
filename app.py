import streamlit as st
from streamlit_chat import message
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from pinecone import Pinecone

st.set_page_config(
    page_title="Handbook Support Service",
    layout="centered")

# Custom CSS for chat interface
st.markdown("""
    <style>
    .stApp {
        max-width: 900px;
        margin: 0 auto;
    }
    
    /* Improve chat message styling */
    .stChatMessage {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    
    /* User message background */
    [data-testid="stChatMessageContent"] {
        background-color: transparent;
    }
    
    /* Make the input more prominent and taller */
    .stChatInputContainer {
        border-top: 1px solid #e0e0e0;
        padding-top: 1rem;
    }
    
    /* Expand chat input height */
    .stChatInputContainer textarea {
        min-height: 80px !important;
        max-height: 120px !important;
        padding: 1rem !important;
        font-size: 1rem !important;
    }
    
    /* Source expander styling */
    .streamlit-expanderHeader {
        font-size: 0.9rem;
        color: #666;
    }
    </style>
""", unsafe_allow_html=True)


INDEX_NAME = "handbook-rag"
TOP_K = 8

if "messages" not in st.session_state:
    st.session_state.messages = []

@st.cache_resource
def init_rag_system():
    try:
        google_api_key = st.secrets["GOOGLE_API_KEY"]
        pinecone_api_key = st.secrets["PINECONE_API_KEY"]

        if not google_api_key or not pinecone_api_key:
            raise ValueError("API keys not found in environment...")
        
        embeddings = GoogleGenerativeAIEmbeddings(
            model = "models/embedding-001",
            google_api_key= google_api_key
        )

        pc = Pinecone(api_key=pinecone_api_key)
        index = pc.Index(INDEX_NAME)

        llm = ChatGoogleGenerativeAI(
            model = "gemini-3-flash-preview",
            google_api_key = google_api_key,
            temperature = 0.3
        )

        return embeddings, index, llm, None
    
    except Exception as e:
        return None, None, None, str(e)
    
embeddings, index, llm, error = init_rag_system()

if error:
    st.info("API Keys are not implemented. Contact the developer for fixes.")
    st.stop()

st.title("Handbook Support Service")
st.caption("Ask anything about the handbook! Please take note that this is for reference purposes. In case if this requires immediate action, please refer to an appropriate body instead. (Note: You can use any language, but it only replies in English.)")

for message in st.session_state.messages:
    avatar = "👤" if message["role"] == "user" else "🤖"
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])
        if "sources" in message and message["sources"]:
            with st.expander("View sources"):
                for i, source in enumerate(message["sources"],1):
                    st.caption(f"**Source {i}** - Page {source['page']}")

if prompt := st.chat_input("Ask anything about the handbook!"):
    st.session_state.messages.append({"role":"user","content": prompt})
    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("Reading high and low..."):
            try:
                conversation_history =""
                for msg in st.session_state.messages[:-1]:
                    if msg["role"] == "user":
                        conversation_history += f"Student: {msg['content']}\n"
                    else:
                        conversation_history += f"Assistant: {msg['content']}\n"
                
                question_embedding = embeddings.embed_query(prompt)
                results = index.query(
                    vector= question_embedding,
                    top_k = TOP_K,
                    include_metdata= True
                )

                retrieved_chunks = []
                sources = []


                for match in results['matches']:
                    chunk_text = match['metadata']['text']
                    page = match['metadata'].get('page','unknown')
                    score = match['score']

                    retrieved_chunks.append(chunk_text)
                    sources.append({
                        'page':page,
                        'score':score
                    })

                context = "\n\n---\n\n".join(retrieved_chunks)

                system_prompt = f"""You are a helpful student assistant for USTP (University of Science and Technology of Southern Philippines). Your role is to answer student questions based ONLY on the information provided from the official student handbook.

STRICT RULES:
1. Only use information from the Context provided below. DO NOT make up or infer information not explicitly stated.
2. If the Context does not contain enough information to answer the question, honestly say "I don't have that information in the handbook" or "The handbook doesn't specify that."
3. Always cite the relevant section when possible (e.g., "According to the handbook...")
4. Be friendly and helpful, but stay factually accurate to the handbook content.
5. Use the conversation history to understand context, but always ground your answer in the handbook content.

Previous Conversation:
{conversation_history}

Context from the student handbook:
{context}

---
Current Question: {prompt}

Answer:"""
                
                response = llm.invoke(system_prompt)
                answer = response.content

                st.markdown(answer)

                with st.expander("View sources"):
                    for i, source in enumerate(sources, 1):
                        st.caption(f"**Source {i}** - Page {source['page']} (Relevance: {source['score']:.2%})")

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources
                })

            except Exception as e:
                  st.error(f"Error generating response: {str(e)}")
                  st.info("If you see this error, kindly contact the developer.")

