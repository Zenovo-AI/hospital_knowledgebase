import boto3
from botocore.exceptions import ClientError
from pathlib import Path
import streamlit as st

SPACE_NAME = "lightrag-bucket"
SPACE_REGION = "nyc3"  # change if yours differs
FOLDER_NAME = "hospital-policy-knowledge-base"

DO_ENDPOINT = f"https://{SPACE_REGION}.digitaloceanspaces.com"
ACCESS_KEY = st.secrets["DO_SPACES_ACCESS_KEY"]
SECRET_KEY = st.secrets["DO_SPACES_SECRET_KEY"]

session = boto3.session.Session()
client = session.client('s3',
    region_name=SPACE_REGION,
    endpoint_url=DO_ENDPOINT,
    aws_access_key_id=ACCESS_KEY,
    aws_secret_access_key=SECRET_KEY
)

# def list_files():
#     try:
#         response = client.list_objects_v2(Bucket=SPACE_NAME, Prefix=FOLDER_NAME + "/")
#         return [content["Key"] for content in response.get("Contents", [])]
#     except ClientError as e:
#         print("Error listing files:", e)
#         return []

# def delete_existing_files():
#     files = list_files()
#     for file_key in files:
#         client.delete_object(Bucket=SPACE_NAME, Key=file_key)

def list_files():
    """List all files inside the folder in the Space, excluding the folder key itself."""
    try:
        response = client.list_objects_v2(Bucket=SPACE_NAME, Prefix=FOLDER_NAME + "/")
        file_keys = []
        for content in response.get("Contents", []):
            key = content["Key"]
            # Skip if the key is just the folder itself (e.g., "knowledge-base/")
            if key != FOLDER_NAME + "/":
                file_keys.append(key)
        return file_keys
    except ClientError as e:
        print("Error listing files:", e)
        return []

def delete_existing_files():
    """Delete all files inside the folder, but not the folder itself."""
    files = list_files()
    for file_key in files:
        try:
            client.delete_object(Bucket=SPACE_NAME, Key=file_key)
            print(f"Deleted {file_key}")
        except ClientError as e:
            print(f"Error deleting {file_key}: {e}")


def file_exists(file_key: str) -> bool:
    try:
        client.head_object(Bucket=SPACE_NAME, Key=file_key)
        return True
    except ClientError as e:
        if e.response['Error']['Code'] == '404':
            return False
        else:
            print(f"Error checking if file exists: {e}")
            raise


# def upload_file(file_path: Path):
#     file_key = f"{FOLDER_NAME}/{file_path.name}"
#     with open(file_path, "rb") as f:
#         client.upload_fileobj(f, SPACE_NAME, file_key)

# Function to upload a file to the Space
def upload_file(file_path: Path):
    file_key = f"{FOLDER_NAME}/{file_path.name}"
    
    # Check if the file already exists in the Space
    if file_exists(file_key):
        print(f"{file_key} exists in the Space. It will be replaced.")
        # Optionally, delete the existing file before uploading new one
        client.delete_object(Bucket=SPACE_NAME, Key=file_key)

    # Upload the new file
    try:
        with open(file_path, "rb") as f:
            client.upload_fileobj(f, SPACE_NAME, file_key)
        print(f"Uploaded {file_key} to Space.")
    except ClientError as e:
        print(f"Error uploading file {file_key}: {e}")


# def download_all_files(download_dir: Path):
#     download_dir.mkdir(parents=True, exist_ok=True)
#     files = list_files()
#     for file_key in files:
#         file_name = file_key.split("/")[-1]
#         download_path = download_dir / file_name
#         client.download_file(SPACE_NAME, file_key, str(download_path))

def download_all_files(download_dir: Path): 
    download_dir.mkdir(parents=True, exist_ok=True)
    files = list_files()
    
    # Skip downloading if files already exist in local dir
    if any(download_dir.iterdir()):
        print(f"📁 Directory '{download_dir}' is not empty. Skipping download.")
        return
    
    for file_key in files:
        file_name = file_key.split("/")[-1]
        download_path = download_dir / file_name
        client.download_file(SPACE_NAME, file_key, str(download_path))
        print(f"⬇️ Downloaded {file_key} to {download_path}")



def delete_all():
    delete_existing_files()
