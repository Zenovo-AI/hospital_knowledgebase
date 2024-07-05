import streamlit as st
from llama_index.core import VectorStoreIndex, ServiceContext, Document
from llama_index.llms.openai import OpenAI
import openai
from llama_index.core import SimpleDirectoryReader
from llama_index.core.memory import ChatMemoryBuffer
from pydantic import BaseModel
from llama_index.core.retrievers import VectorIndexRetriever
import re
import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext
    
memory = ChatMemoryBuffer.from_defaults(token_limit=2500)

st.set_page_config(page_title="Chat with HospitalAI", page_icon="", layout="centered", initial_sidebar_state="auto", menu_items=None)

openai.api_key = st.secrets.openai_key

st.title("HospitalAI")
         
if "messages" not in st.session_state.keys(): # Initialize the chat messages history
    st.session_state.messages = [
        {"role": "assistant", "content": "Ask me a Question regarding Hospital Policies."}
    ]

@st.cache_resource(show_spinner=False)
def load_data():
    with st.spinner(text="Loading Data..."):
         # initialize client, setting path to save data
        db = chromadb.PersistentClient(path="./chroma_db")
        # create collection
        chroma_collection = db.get_or_create_collection("chatbot")
        # assign chroma as the vector_store to the context
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        # load your index from stored vectors
        index = VectorStoreIndex.from_vector_store(
            vector_store, storage_context=storage_context
        )
        return index

index = load_data()


if "chat_engine" not in st.session_state.keys(): # Initialize the chat engine
        # st.session_state.chat_engine = index.as_chat_engine(chat_mode="condense_question", verbose=True)
        st.session_state.chat_engine = index.as_chat_engine(chat_mode="context", max_iterations=10,memory=memory, verbose=False, system_prompt="You are going to answer questions and make conversation regarding Hospital Policies. You can only make a conversation if the response is from the provided sources if none of the sources are relevant to the query, respond politely \"Please call the Risk Management Representative on call.\", user can ask question or just write a statement, you must respond.")

if prompt := st.chat_input("Your question"): # Prompt for user input and save to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

for message in st.session_state.messages: # Display the prior chat messages
    with st.chat_message(message["role"]):
        st.write(message["content"])

# If last message is not from assistant, generate a new response
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = st.session_state.chat_engine.chat(prompt)
            st.write(response.response)
            # numbers = re.findall(r'\[(\d+)\]', response.response)
            # numbers = set(numbers)  # Remove duplicates
            # for number in numbers:
            #     number = int(number) - 1
            #     node = response.source_nodes[number]
            #     text_fmt = node.node.get_content().strip().replace("\n", " ")[:1000]
            #     citation = f'<p style="color:blue;">Citation: {text_fmt}</p>'  # Make the citation italic and blue
            #     st.markdown(citation, unsafe_allow_html=True)
            for source in response.source_nodes:
                print(source)
                text_fmt = source.node.get_content().strip().replace("\n", " ")[:1000]
                citation = f'<p style="color:blue;">Citation: {text_fmt}</p>'  # Make the citation italic and blue
                st.markdown(citation, unsafe_allow_html=True)
            message = {"role": "assistant", "content": response.response}
            st.session_state.messages.append(message) # Add response to message history