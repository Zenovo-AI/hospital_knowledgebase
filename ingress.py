# import sqlite3
# import traceback
# import streamlit as st
# from pathlib import Path
# from db_helper import insert_file_metadata
# from document_processor import DocumentProcessor

# # Initialize document processor
# process_document = DocumentProcessor()

# def ingress_file_doc(file_name: str, file_path: str = None, web_links: list = None):
#     from app import RAGFactory
    
#     try:
#         # Connect to the database
#         conn = sqlite3.connect("files.db", check_same_thread=False)
#         cursor = conn.cursor()

#         # Check if the file already exists in the database
#         cursor.execute("SELECT file_name FROM documents WHERE file_name = ?", (file_name,))
#         if cursor.fetchone():
#             st.sidebar.warning(f"⚠️ File '{file_name}' has already been uploaded.")
#             return {"error": "File already exists."}

#         # Initialize text content list
#         text_content = []

#         # Process file content if file_path is provided
#         if file_path:
#             file_path_str = str(file_path)  # Convert Path object to string
#             if file_path_str.endswith(".pdf"):
#                 extracted_text = process_document.extract_text_and_tables_from_pdf(file_path_str)
#                 if extracted_text:
#                     text_content.append(extracted_text)
#             elif file_path_str.endswith(".txt"):
#                 text_content.append(process_document.extract_txt_content(file_path_str))
#             else:
#                 return {"error": "❌ Unsupported file format."}

#         # Process web links if provided
#         if web_links:
#             for link in web_links:
#                 web_content = process_document.process_webpage(link)
                
#                 if web_content:
#                     text_content.append(web_content)

#         # Ensure there is content to process
#         if not text_content:
#             return {"error": "No valid content extracted from file or web links."}

#         # Insert metadata into the database
#         for content in text_content:
#             insert_file_metadata(file_name, content)

#         # Create working directory
#         working_dir = Path("./analysis_workspace")
#         working_dir.mkdir(parents=True, exist_ok=True)  # Ensure directory exists

#         # Process data using RAGFactory
#         rag = RAGFactory.create_rag(str(working_dir))
#         rag.insert(text_content)

#         # Show success message
#         st.success(f"✅ File '{file_name}' processed successfully!")
#         return {"success": True}

#     except Exception as e:
#         traceback.print_exc()
#         return {"error": str(e)}

#     finally:
#         conn.close()




import sqlite3
import traceback
import streamlit as st
from pathlib import Path
from db_helper import insert_file_metadata
from document_processor import DocumentProcessor

# Initialize document processor
process_document = DocumentProcessor()

def ingress_file_doc(file_name: str = None, file_path: str = None, web_links: list = None):
    from app import RAGFactory

    try:
        conn = sqlite3.connect("files.db", check_same_thread=False)
        cursor = conn.cursor()

        text_content = []

        # ✅ If a file is uploaded, process it
        if file_path:
            cursor.execute("SELECT file_name FROM documents WHERE file_name = ?", (file_name,))
            if cursor.fetchone():
                st.sidebar.warning(f"⚠️ File '{file_name}' has already been uploaded.")
                return {"error": "File already exists."}

            file_path_str = str(file_path)
            if file_path_str.endswith(".pdf"):
                extracted_text = process_document.extract_text_and_tables_from_pdf(file_path_str)
                if extracted_text:
                    text_content.append(extracted_text)
            elif file_path_str.endswith(".txt"):
                text_content.append(process_document.extract_txt_content(file_path_str))
            else:
                return {"error": "❌ Unsupported file format."}

        # ✅ If web links are provided, scrape them
        if web_links:
            for link in web_links:
                cursor.execute("SELECT file_name FROM documents WHERE file_name = ?", (link,))
                if cursor.fetchone():
                    st.sidebar.warning(f"⚠️ Web link '{link}' has already been processed.")
                    continue  # Skip duplicate links

                web_content = process_document.process_webpage(link)
                if web_content:
                    text_content.append(web_content)
                else:
                    st.sidebar.error(f"❌ Failed to scrape content from {link}")

        # ✅ Ensure at least some content was extracted
        if not text_content:
            return {"error": "No valid content extracted from file or web links."}

        # ✅ Insert into the database
        for content in text_content:
            insert_file_metadata(file_name or "web_link", content)

        # ✅ Create working directory
        working_dir = Path("./analysis_workspace")
        working_dir.mkdir(parents=True, exist_ok=True)

        # ✅ Insert into LightRAG
        rag = RAGFactory.create_rag(str(working_dir))
        rag.insert(text_content)

        # ✅ Show success message
        st.success(f"✅ {'File' if file_name else 'Web links'} processed successfully!")
        return {"success": True}

    except Exception as e:
        traceback.print_exc()
        return {"error": str(e)}

    finally:
        conn.close()
