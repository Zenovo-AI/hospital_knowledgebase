from llama_index.core import SimpleDirectoryReader
from llama_index.core import VectorStoreIndex
import dotenv
import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext


dotenv.load_dotenv()

#Read the directory
def read_directory_and_index(storage_context, data_dir='data'):   
    documents = SimpleDirectoryReader(data_dir).load_data()
    index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)
    return index

def get_storage_context():
    # initialize client, setting path to save data
    db = chromadb.PersistentClient(path="chroma_db")
    #check if collection exists
    collections = db.list_collections()
    #change collections which is a list to a string
    collections = str(collections)
    if "chatbot" in collections:
        #delete collection
        db.delete_collection("chatbot")
        # create collection
        chroma_collection = db.get_or_create_collection("chatbot")
    else:
        # create collection
        chroma_collection = db.get_or_create_collection("chatbot")
    # assign chroma as the vector_store to the context
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    return storage_context

if __name__ == "__main__":
    storage_context = get_storage_context()
    index = read_directory_and_index(storage_context)
    print(index)
    print("Indexing complete")