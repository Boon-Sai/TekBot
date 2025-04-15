import streamlit as st
from sentence_transformers import SentenceTransformer
import pandas as pd
import chromadb
from chromadb.utils import embedding_functions

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "df" not in st.session_state:
    st.session_state.df = None
if "chroma_collection" not in st.session_state:
    st.session_state.chroma_collection = None
if "model" not in st.session_state:
    st.session_state.model = None

# Load BERT model
@st.cache_resource
def load_model():
    return SentenceTransformer('sentence-transformers/bert-base-nli-mean-tokens')

@st.cache_resource
def initialize_chroma():
    # Create ChromaDB client and collection
    chroma_client = chromadb.Client()
    
    # Use Sentence Transformer for embeddings
    st_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="bert-base-nli-mean-tokens"
    )
    
    return chroma_client.create_collection(
        name="company_info",
        embedding_function=st_ef
    )

st.title("🧠 Company Information Chatbot (ChromaDB-powered)")

# Sidebar for CSV upload
with st.sidebar:
    st.header("📁 Upload CSV")
    uploaded_file = st.file_uploader(
        "Upload your CSV file (columns: text,intent,response)", 
        type=["csv"],
        key="file_uploader",
        label_visibility="collapsed"
    )
    
    if uploaded_file and st.session_state.df is None:
        st.session_state.df = pd.read_csv(uploaded_file)
        st.success("CSV file loaded successfully!")
        
        # Show preview of the data
        st.subheader("Data Preview")
        st.dataframe(st.session_state.df.head())
        
        # Initialize ChromaDB collection
        with st.spinner("Setting up ChromaDB and generating embeddings..."):
            st.session_state.model = load_model()
            st.session_state.chroma_collection = initialize_chroma()
            
            # Add data to ChromaDB
            documents = st.session_state.df['text'].tolist()
            metadatas = [{"intent": row.intent, "response": row.response} 
                         for _, row in st.session_state.df.iterrows()]
            ids = [str(i) for i in range(len(st.session_state.df))]
            
            st.session_state.chroma_collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            
            st.success(f"Loaded {len(documents)} documents into ChromaDB!")

# Main area for chat interface
if st.session_state.df is not None:
    # Display chat messages from history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "intent" in message:
                st.caption(f"Intent: {message['intent']}")
    
    # Accept user input
    if prompt := st.chat_input("Ask your question about the company..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message immediately
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Query ChromaDB
        results = st.session_state.chroma_collection.query(
            query_texts=[prompt],
            n_results=1
        )
        
        # Get the best match
        best_match = results['metadatas'][0][0]
        response = best_match["response"]
        intent = best_match["intent"]
        distance = results['distances'][0][0]
        
        # Display assistant response
        with st.chat_message("assistant"):
            st.markdown(response)
            st.caption(f"Intent: {intent}")
            st.caption(f"Similarity score: {1 - distance:.2f}")  # Convert distance to similarity
        
        # Add assistant response to chat history
        st.session_state.messages.append({
            "role": "assistant", 
            "content": response,
            "intent": intent
        })
else:
    st.info("Please upload a CSV file to start chatting")

# Add some styling to make the chat area take full width
st.markdown("""
    <style>
        .main .block-container {
            max-width: 80%;
        }
        .sidebar .sidebar-content {
            width: 20% !important;
        }
    </style>
""", unsafe_allow_html=True)