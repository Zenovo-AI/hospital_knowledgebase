# from pathlib import Path
# import time
# import streamlit as st
# from ingress import ingress_file_doc


# def process_files_and_links(files, web_links):
#     with st.spinner("Processing..."):
#         for uploaded_file in files:
#             process_file(uploaded_file, web_links)  # ✅ Call function directly
#     st.session_state["files_processed"] = True

# def process_file(uploaded_file, web_links):
#     try:
#         file_name = uploaded_file.name
#         st.session_state["file_name"] = file_name
#         # Use pathlib to define the file path
#         temp_dir = Path("./temp_files")
#         temp_dir.mkdir(parents=True, exist_ok=True)  # Ensure the directory exists
        
#         # Define the file path
#         file_path = temp_dir / file_name  # Concatenate the directory path and file name

#         # Save file locally for processing
#         with open(file_path, "wb") as f:
#             f.write(uploaded_file.getvalue())


#         # Call the function with correct arguments
#         try:
#             response = ingress_file_doc(file_name, file_path, web_links or [])
#             if "error" in response:
#                 st.error(f"File processing error: {response['error']}")
#             else:
#                 placeholder = st.empty()
#                 placeholder.success(f"File '{file_name}' processed successfully!")
#                 time.sleep(5)
#                 placeholder.empty()
#         except Exception as e:
#             st.error(f"Unexpected error: {e}")

#     except Exception as e:
#         st.error(f"Connection error: {e}")




from pathlib import Path
import time
import streamlit as st
from ingress import ingress_file_doc
from langchain.chains.qa_with_sources.retrieval import RetrievalQAWithSourcesChain
from langchain_openai import OpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS


# Define FAISS index storage path
FAISS_INDEX_PATH = Path("faiss_index")

# Load OpenAI Embeddings
embeddings = OpenAIEmbeddings()


def process_files_and_links(files, web_links):
    with st.spinner("Processing..."):
        # ✅ Process files
        for uploaded_file in files:
            process_file(uploaded_file)  

        # ✅ Process web links
        if web_links:
            process_web_links(web_links)

    st.session_state["files_processed"] = True

def process_file(uploaded_file):
    try:
        file_name = uploaded_file.name
        st.session_state["file_name"] = file_name

        temp_dir = Path("./temp_files")
        temp_dir.mkdir(parents=True, exist_ok=True)

        file_path = temp_dir / file_name
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getvalue())

        response = ingress_file_doc(file_name=file_name, file_path=file_path)
        if "error" in response:
            st.error(f"File processing error: {response['error']}")
        else:
            st.success(f"✅ File '{file_name}' processed successfully!")

    except Exception as e:
        st.error(f"❌ Connection error: {e}")

def process_web_links(web_links):
    """Processes web links separately."""
    try:
        response = ingress_file_doc(web_links=web_links)
        if "error" in response:
            st.error(f"Web link processing error: {response['error']}")
        else:
            st.success(f"✅ Web links processed successfully!")

    except Exception as e:
        st.error(f"❌ Connection error: {e}")

# Function to create or load FAISS index
def load_or_create_faiss_index(documents):
    
    # Ensure the directory exists
    FAISS_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    if FAISS_INDEX_PATH.exists():
        placeholder = st.empty()
        placeholder.write("✅ FAISS index found. Loading...")
        time.sleep(5)
        placeholder.empty()
        return FAISS.load_local(str(FAISS_INDEX_PATH), embeddings, allow_dangerous_deserialization=True)
    st.write("⚠️ FAISS index not found.")
    
    
    if not documents:
        print("⚠️ No documents to index. Skipping FAISS initialization.")
        return None
    
    texts = [doc.page_content for doc in documents]
    metadatas = [{"source": doc.metadata.get("source", "Unknown")} for doc in documents]
    
    if not texts:
        print("⚠️ No valid text found in documents. Skipping FAISS initialization.")
        return None
    
    vector_store = FAISS.from_texts(texts, embeddings, metadatas=metadatas)
    vector_store.save_local(str(FAISS_INDEX_PATH))
    return vector_store

# Load or initialize FAISS index
documents = []  # Populate this list dynamically\if "vector_store" not in st.session_state:
st.session_state["vector_store"] = load_or_create_faiss_index(documents)

vector_store = st.session_state["vector_store"]
if vector_store:
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 50})
else:
    retriever = None
    print("📢 No vector store created. Waiting for document upload.")

# Load LLM
llm = OpenAI(temperature=0.7)
if retriever:
    chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=retriever)
else:
    print("🚨 No retriever available! Waiting for document upload.")
    chain = None

def run_qa_chain(query):
    qa_results = chain.invoke({"question": query})
    return qa_results

def retrieve_answers(query):
    print(f"Retrieving answers for: {query}")
    try:
        response = run_qa_chain(query)
        print(f"Chain Response: {response}")

        if not isinstance(response, dict):
            return {"answer": "⚠️ Unexpected response format.", "sources": "N/A"}

        return response
    except Exception as e:
        return {"answer": f"⚠️ Error: {str(e)}", "sources": "N/A"}

# Function to add new documents without overwriting
def add_documents_to_faiss(new_documents):
    if new_documents:
        vector_store.add_texts(
            texts=[doc.page_content for doc in new_documents],
            metadatas=[{"source": doc.metadata.get("source", "Unknown")} for doc in new_documents]
        )
        vector_store.save_local(str(FAISS_INDEX_PATH))  # Save FAISS index persistently
        st.success("New documents added successfully! ✅")

# Function to clear FAISS index
def clear_faiss_index():
    global vector_store
    if FAISS_INDEX_PATH.exists():
        import shutil
        shutil.rmtree(FAISS_INDEX_PATH)
    vector_store = FAISS(embeddings)
    st.session_state["vector_store"] = vector_store
    st.success("FAISS index cleared successfully!")