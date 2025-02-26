import logging
from pathlib import Path
import sqlite3
import time
import numpy as np
import streamlit as st
from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import openai_embed, gpt_4o_complete
from lightrag.utils import EmbeddingFunc
from db_helper import check_if_file_exists, check_working_directory, delete_file, initialize_database
from inference import process_files_and_links
from googleapiclient.discovery import build
from streamlit_js import st_js, st_js_blocking
from google_auth_oauthlib.flow import Flow
import logging


def initialize_session_state():
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "initialized" not in st.session_state:
        initialize_database()
        st.session_state.initialized = True


def embedding_func(texts: list[str]) -> np.ndarray:
    embeddings = openai_embed(
        texts,
        model="text-embedding-3-large",
        api_key=st.secrets["OPENAI_API_KEY"],
        base_url=None
    )
    if embeddings is None:
        logging.error("Received empty embeddings from API.")
        return np.array([])
    return embeddings


class RAGFactory:
    _shared_embedding = EmbeddingFunc(
        embedding_dim=3072,
        max_token_size=8192,
        func=embedding_func
    )

    @classmethod
    def create_rag(cls, working_dir: str) -> LightRAG:
        """Create a LightRAG instance with shared configuration"""
        return LightRAG(
            working_dir=working_dir,
            addon_params={"insert_batch_size": 50},
            llm_model_func=gpt_4o_complete,
            embedding_func=cls._shared_embedding
        )


def generate_answer():
    """Generates an answer when the user enters a query and presses Enter."""
    query = st.session_state.query_input  # Get user query from session state
    if not query:
        return  # Do nothing if query is empty

    with st.spinner("Generating answer..."):
        try:
            working_dir = Path("./analysis_workspace")
            working_dir.mkdir(parents=True, exist_ok=True)
            rag = RAGFactory.create_rag(str(working_dir))
            response = rag.query(query, QueryParam(mode="hybrid"))

            # Store in chat history
            st.session_state.chat_history.append(("You", query))
            st.session_state.chat_history.append(("Bot", response))
        except Exception as e:
            st.error(f"Error retrieving response: {e}")

    # Reset query input to allow further queries
    st.session_state.query_input = ""
    
    
def process_web_links():
    """Trigger processing when web links are entered."""
    web_links = st.session_state["web_links"].strip()
    if web_links:  # Ensure input is not empty
        placeholder = st.empty()
        placeholder.write("🔄 Processing web links...")
        time.sleep(5)
        placeholder.empty()

        process_files_and_links([], web_links.split("\n"))  # Convert to list
        st.session_state["files_processed"] = True
        placeholder = st.empty()
        placeholder.write("✅ Web links processed!")
        time.sleep(5)
        placeholder.empty()
        
        
# client_config = st.sidebar.file_uploader("Upload your client secret JSON file", type=["json"])
# if client_config:
#     client_config = json.loads(client_config.read())
# redirect_uri = "http://localhost:8501"

# # Local Storage Functions


# # Function to retrieve data from local storage
# def ls_get(k, key=None):
#     if key is None:
#         key = f"ls_get_{k}"
#     return st_js_blocking(f"return JSON.parse(localStorage.getItem('{k}'));", key=key)

# # Function to set data in local storage
# def ls_set(k, v, key=None):
#     if key is None:
#         key = f"ls_set_{k}"
#     jdata = json.dumps(v, ensure_ascii=False)
#     st_js_blocking(f"localStorage.setItem('{k}', JSON.stringify({jdata}));", key=key)

# # Initialize session with user info if it exists in local storage
# def init_session():
#     key = "user_info_init_session"
#     if "user_info" not in st.session_state:
#         user_info = ls_get("user_info", key=key)
#         if user_info:
#             st.session_state["user_info"] = user_info


# def auth_flow():
#     st.write("Welcome to Health Policy App!")
#     auth_code = st.query_params.get("code")
#     flow = Flow.from_client_config(
#         client_config,
#         scopes=[
#             "https://www.googleapis.com/auth/youtube.force-ssl", 
#             "https://www.googleapis.com/auth/userinfo.profile", 
#             "https://www.googleapis.com/auth/userinfo.email", 
#             "openid"
#         ],
#         redirect_uri=redirect_uri,
#     )
#     if auth_code:
#         flow.fetch_token(code=auth_code)
#         credentials = flow.credentials
#         st.session_state["credentials"] = credentials  # Store credentials
#         st.write("Login Done")
#         user_info_service = build(
#             serviceName="oauth2",
#             version="v2",
#             credentials=credentials,
#         )
#         user_info = user_info_service.userinfo().get().execute()
#         assert user_info.get("email"), "Email not found in infos"
#         st.session_state["google_auth_code"] = auth_code
#         st.session_state["user_info"] = user_info
#         ls_set("user_info", user_info)
#     else:
#         authorization_url, state = flow.authorization_url(
#             access_type="offline",
#             include_granted_scopes="true",
#         )
#         st.link_button("Sign in with Google", authorization_url)


def main():
    logging.getLogger("root").setLevel(logging.CRITICAL)
    
    st.set_page_config(page_title="Hospital Policy Search", layout="wide")
    admin_password = st.secrets["ADMIN_PASSWORD"]

    # Sidebar: Admin Mode
    st.sidebar.title("Admin Panel")
    admin_mode = st.sidebar.checkbox("Enable Admin Mode")
    admin_authenticated = False

    if admin_mode:
        password = st.sidebar.text_input("Enter Admin Password", type="password")
        if password == admin_password:
            admin_authenticated = True
            st.sidebar.success("Admin authenticated")
        else:
            st.sidebar.error("Incorrect password!")

    st.title("Health Policy APP")
    st.write("Upload a document and ask questions based on structured knowledge retrieval.")

    initialize_session_state()
    
    if "files_processed" not in st.session_state:
        st.session_state["files_processed"] = False

    files = None  # Ensure files is always defined
    web_links = None  # Ensure web_links is always defined

    if admin_authenticated:
        st.sidebar.subheader("Upload New Documents")

        # File uploader widget
        files = st.sidebar.file_uploader("Upload documents", accept_multiple_files=True, type=["pdf", "txt"])

        # Store uploaded file name in session state
        if files:
            for file in files:
                st.session_state["file_name"] = file.name

        # Web links input
        web_links = st.sidebar.text_area("Enter web links (one per line)", key="web_links", on_change=process_web_links)

    # Process files and links if present
    if (files or web_links) and not st.session_state["files_processed"]:
        if files:  # Process files if available
            for file in files:
                file_name = file.name
                file_in_db = check_if_file_exists(file_name)
                dir_exists = check_working_directory(file_name)

                if file_in_db and dir_exists:
                    placeholder = st.empty()
                    placeholder.warning(f"⚠️ The file '{file_name}' has already been processed.")
                    time.sleep(5)
                    placeholder.empty()
                else:
                    placeholder = st.empty()
                    placeholder.write("🔄 Processing files and links...")
                    time.sleep(5)
                    placeholder.empty()

                    process_files_and_links(files, web_links)
                    st.session_state["files_processed"] = True

                    placeholder.write("✅ Files and links processed!")
                    time.sleep(5)
                    placeholder.empty()

        elif web_links:  # Process web links even if no files are uploaded
            placeholder = st.empty()
            placeholder.write("🔄 Processing web links...")
            time.sleep(5)
            placeholder.empty()

            process_files_and_links([], web_links.split("\n"))  # Convert to list
            st.session_state["files_processed"] = True

            placeholder.write("✅ Web links processed!")
            time.sleep(5)
            placeholder.empty()


    # Reset processing state and delete working directory
    if st.sidebar.button("Reset Processing", key="reset"):
        # Clear session state except for initialized state
        keys_to_keep = {"initialized"}
        for key in list(st.session_state.keys()):
            if key not in keys_to_keep:
                del st.session_state[key]

        # Reset processing flag
        st.session_state["files_processed"] = False

        # Define the working directory
        working_dir = Path("./analysis_workspace")

        # Delete the working directory if it exists
        if working_dir.exists() and working_dir.is_dir():
            import shutil
            shutil.rmtree(working_dir)
            st.sidebar.success("🔄 Processing reset! The working directory has been deleted.")
        else:
            st.sidebar.warning("⚠️ No working directory found to delete.")

    # Input field with automatic query execution on Enter
    st.text_input("Ask a question about the document:", key="query_input", on_change=generate_answer)

    # Display chat history
    for role, message in st.session_state.chat_history:
        with st.chat_message("user" if role == "You" else "assistant"):
            st.write(message)

    # Sidebar: Uploaded files display
    st.sidebar.write("### Uploaded Files")
    try:
        conn = sqlite3.connect("files.db", check_same_thread=False)
        cursor = conn.cursor()
        cursor.execute("SELECT file_name FROM documents;")  # Uses a single `documents` table
        uploaded_files_list = [file[0] for file in cursor.fetchall()]

        if uploaded_files_list:
            for file_name in uploaded_files_list:
                delete_key = f"delete_{file_name}"
                col1, col2 = st.sidebar.columns([3, 1])
                with col1:
                    st.sidebar.write(file_name)
                with col2:
                    if st.sidebar.button("Delete", key=delete_key):
                        try:
                            delete_file(file_name)
                            st.sidebar.success(f"✅ File '{file_name}' deleted successfully!")
                        except Exception as e:
                            st.error(f"❌ Failed to delete file '{file_name}': {e}")
        else:
            st.sidebar.info("ℹ️ No files uploaded.")
    except Exception as e:
        st.sidebar.error(f"❌ Failed to retrieve files: {e}")


if __name__ == "__main__":
    main()