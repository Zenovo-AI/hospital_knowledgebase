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
