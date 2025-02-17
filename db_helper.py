import sqlite3
from pathlib import Path
from document_processor import DocumentProcessor

# Initialize document processor
process_document = DocumentProcessor()

# Initialize database with a single table
def initialize_database():
    conn = sqlite3.connect("files.db")
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY,
            file_name TEXT UNIQUE,
            file_content TEXT,
            upload_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)
    
    conn.commit()
    conn.close()


# Insert document metadata and content into the database
def insert_file_metadata(file_name, file_content):
    conn = sqlite3.connect("files.db")
    cursor = conn.cursor()
    try:
        cursor.execute("""
            INSERT INTO documents (file_name, file_content)
            VALUES (?, ?);
        """, (file_name, file_content))
        print(f"Inserted file content: {file_content[:100]}...")  # Print only the first 100 chars
        conn.commit()
    except sqlite3.IntegrityError:
        print(f"⚠️ File '{file_name}' already exists in the database.")
    except Exception as e:
        print(f"❌ Error inserting file metadata: {e}")
    finally:
        conn.close()


# Delete document by file name
def delete_file(file_name):
    conn = sqlite3.connect("files.db")
    cursor = conn.cursor()
    try:
        cursor.execute("DELETE FROM documents WHERE file_name = ?", (file_name,))
        conn.commit()
        print(f"✅ File '{file_name}' deleted from database.")
    except Exception as e:
        print(f"❌ Error deleting file: {e}")
    finally:
        conn.close()


# Check if a file already exists in the database
def check_if_file_exists(file_name):
    conn = sqlite3.connect("files.db")
    cursor = conn.cursor()
    
    cursor.execute("SELECT 1 FROM documents WHERE file_name = ?", (file_name,))
    result = cursor.fetchone()
    
    conn.close()
    
    return result is not None  # Returns True if file exists, otherwise False


# Check if the working directory exists for a processed file
def check_working_directory(file_name):
    working_dir = Path(f"./analysis_workspace/{file_name.split('.')[0]}")
    return working_dir.exists() and working_dir.is_dir()
