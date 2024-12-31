# [2412]
#
# * A retrieval-centric interface for CCC-PA
#


import streamlit as st
import json
import sys, os
import datetime

from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.document_loaders import TextLoader, UnstructuredPDFLoader
from langchain.prompts import PromptTemplate
from langchain.text_splitter import CharacterTextSplitter
from langchain_chroma import Chroma
from langchain_google_vertexai import VertexAI, VertexAIEmbeddings
import vertexai
import chromadb
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

def upload_directory_to_gcs(local_directory, gcs_project_id,
                            gcs_bucket_name, gcs_directory):
    '''
    Function to upload a directory to Google Cloud Storage

    :param local_directory:
    :param bucket_name:
    :param gcs_directory:

    :return:
    '''

    # Initialize GCS client
    storage_client = storage.Client(project=gcs_project_id)
    bucket = storage_client.bucket(bucket_name=gcs_bucket_name)

    for root, _, files in os.walk(local_directory):
        for file_name in files:
            local_file_path = os.path.join(root, file_name)
            relative_path = os.path.relpath(local_file_path, local_directory)

            # Check if files should be stored in subdirectory of directly in bucket
            if gcs_directory == "":
                blob = bucket.blob(os.path.join(relative_path))
            else:
                blob = bucket.blob(os.path.join(gcs_directory, relative_path))

            # Upload
            blob.upload_from_filename(local_file_path)
            print(f"Uploaded {local_file_path} to gs://{gcs_bucket_name}/{gcs_directory}{relative_path}")


def download_directory_from_gcs(gcs_project_id, gcs_bucket_name,
                                gcs_directory, local_directory):
    '''
    Function to download a folder in Google Cloud Storage bucket to a local directory

    :param local_directory:
    :param bucket_name:
    :param gcs_directory:

    :return:
    '''

    # Initialize GCS client
    storage_client = storage.Client(project=gcs_project_id)
    bucket = storage_client.bucket(bucket_name=gcs_bucket_name)

    blobs = bucket.list_blobs(prefix=gcs_directory)

    for blob in blobs:
        if not blob.name.endswith("/"):  # Avoid directory blobs
            relative_path = os.path.relpath(blob.name, gcs_directory)
            local_file_path = os.path.join(local_directory, relative_path)
            os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
            blob.download_to_filename(local_file_path)
            print(f"Downloaded {blob.name} to {local_file_path}")

class BotCCCGlobals:

    """ BotCCCGlobals class holds globals for a chatbot based on a
    pre-trained model (phi3) and document retrieval from a vector
    database (ChromaDB). It includes a range of defaults, including
    an (optional) succinctness constraint, the default question, the
    file name for any saved transcripts, and the path & collection
    name for the vector database. """

    def __init__(self):

        self.transcript_name_base = "cccbot_transcript"
        self.transcript_path = "./local_transcripts/"
        self.transcript_gcs_bucket = "cccbot-transcripts"
        self.transcript_gcs_directory = ""
        # self.collection_name = "crawl_docs1"

        self.gcp_project_id = "eternal-bongo-435614-b9"
        self.gcp_location = "us-central1"
        self.gcs_embeddings_bucket_name = "ccc-chromadb-vai"
        self.gcs_embeddings_directory = ""
        self.embedding_model = "textembedding-gecko@003"
        self.embedding_num_batch = 5
        self.embeddings_local_path = "./local_chromadb/"

        self.llm_model = "gemini-1.5-pro"
        self.llm_model_max_output_tokens = 2048
        self.llm_model_temperature = 0.2
        self.llm_model_top_p = 0.8
        self.llm_model_top_k = 40
        self.llm_model_verbose = True

        self.retriever_search_type = "similarity"
        self.retriever_search_kwargs = {"k": 3}

        self.default_question = "What shall we discuss?"
        # self.be_succinct = ('Please provide only a one or two word answer' +
        #                     'Be as succinct as possible when answering. ')
        # self.prompt_template = """
        #                         You are a California Community College AI assistant.
        #                         You're tasked to answer the question given below,
        #                         but only based on the context provided.
        #                         context: {context}
        #                         question: {input}
        #                         If you cannot find an answer ask the user to rephrase the question.
        #                         answer:
        #                        """

        self.prompt_template = """
                                You are a California Community College AI assistant.
                                Use the following pieces of context to answer the question at the end. 
                                If you don't know the answer, just say that you don't know, 
                                don't try to make up an answer.

                               """
        self.chat_hist_memory_key = "chat_history"
        self.chat_hist_return_messages = True
        self.chat_hist_output_key = "answer"

        self.conv_chain_chain_type="stuff"
        self.conv_chain_chain_type_verbose=True
        self.conv_chain_chain_type_return_source_documents=True



########## Set up Chatbot

### Step 1: Establish parameters
bot_params = BotCCCGlobals()

### Step 2: Initialize Vertex AI
vertexai.init(project=bot_params.gcp_project_id,
              location=bot_params.gcp_location)

### Step 3: Copy Chromadb from GCS to a local directory

def setup_vectorstore():
    '''
    Function to set up vector store. This returns a vector store that
    can be set in the Streamlit session eliminating the automatic
    refresh that would happen with each new question

    '''

    # Download files from GCP
    download_directory_from_gcs(gcs_project_id=bot_params.gcp_project_id,
                                gcs_bucket_name=bot_params.gcs_embeddings_bucket_name,
                                gcs_directory="",
                                local_directory=bot_params.embeddings_local_path)

    # Load embeddings and persisted data
    embeddings = VertexAIEmbeddings(model_name=bot_params.embedding_model)

    # Load Chroma data from local persisted directory
    db = Chroma(persist_directory=bot_params.embeddings_local_path,
                embedding_function=embeddings)

    return db

def chat_chain(vectorstore):
    '''
    Function to set up chat_chain. This returns a conversational chain that
    can be set in the Streamlit session eliminating the automatic
    refresh that would happen with each new question.

    From searching, it seems that ConversationalRetrievalChain.from_llm might be
    deprecated so we may need to update the functionality used here. Also, it's not obvious
    how to provide a custom prompt.

    '''

    llm = VertexAI(model=bot_params.llm_model,
                   max_output_tokens=bot_params.llm_model_max_output_tokens,
                   temperature=bot_params.llm_model_temperature,
                   top_p=bot_params.llm_model_top_p,
                   top_k=bot_params.llm_model_top_k,
                   verbose=bot_params.llm_model_verbose,
                   )

    retriever = vectorstore.as_retriever()
    memory = ConversationBufferMemory(
        llm=llm,
        output_key=bot_params.chat_hist_output_key,
        memory_key=bot_params.chat_hist_memory_key,
        return_messages=bot_params.chat_hist_return_messages
    )

    # Not sure if this is used
    prompt = PromptTemplate.from_template(bot_params.prompt_template)

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        chain_type=bot_params.conv_chain_chain_type,
        memory=memory,
        verbose=bot_params.conv_chain_chain_type_verbose,
        return_source_documents=bot_params.conv_chain_chain_type_return_source_documents
    )

    return chain


########## Set up Streamlit
### Set up header
st.title("California Community Colleges Policy Assistant: Retrieval")
bot_summary = ("This an experimental chatbot employing Artificial Intelligence tools "
               "to help users easily improve their understanding of policy topics related "
               "to California's community colleges.  Its primary purpose is to demonstrate "
               "how policy advocacy can be supported through the use of AI technology. ")

invite = ("If you want to learn more or have thoughts about the application of this or similar "
               "tools or the underlying technology, please reach out to Steve or Nathan at "
               "info@numanticsolutions.com")

st.text(bot_summary)
st.text(invite)
st.divider()
st.text("""Example question : What are the responsibilities of the board members of a California community college?""")

st.divider()

with st.sidebar:

    # Add Company name
    st.title(":orange[Numantic Solutions]")

    # Add logo
    st.image("Numantic Solutions_Logomark_orange.png", width=200)

    st.title("Notes")
    sidebar_msg = ("This is a chatbot that is experimental and still in development. "
                   "Please reach out with feedback, suggestions and comments. Thank you")
    st.text(sidebar_msg)

    if st.button("Save Transcript"):
        print("save transcript")
        messages = []
        for i in range(len(st.session_state.messages)):
            messages.append(st.session_state.messages[i].copy())

        # Construct a filename
        now = datetime.datetime.now()
        tfilename = "{}_{}.json".format(bot_params.transcript_name_base,
                                        now.strftime("%Y%m%d_%H%M%S"))

        # Save locally
        with open(os.path.join(bot_params.transcript_path, tfilename), "w") as file:
            json.dump(messages, file)

        # Copy to GCS
        upload_directory_to_gcs(local_directory=bot_params.transcript_path,
                                gcs_project_id=bot_params.gcp_project_id,
                                gcs_bucket_name=bot_params.transcript_gcs_bucket,
                                gcs_directory=bot_params.transcript_gcs_directory)

########## Handle conversations in Streamlit

# Build session components if needed
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = setup_vectorstore()

if "conversationsal_chain" not in st.session_state:
    st.session_state.conversational_chain = chat_chain(st.session_state.vectorstore)

if "messages" not in st.session_state:
    st.session_state.messages = []

# displays the chat history when app is rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Input box for user's query
user_input = st.chat_input("Your message")

if user_input:
    # Display user's message
    with st.chat_message("user"):
        st.markdown(user_input)

    # Store user's query in the chat history
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Get the AI assistant's response
    response = st.session_state.conversational_chain({"question": user_input})
    assistant_response = response["answer"]

    # Store AI's response in the chat history
    st.session_state.messages.append({"role": "assistant", "content": assistant_response})

    # Display assistant's message
    with st.chat_message("assistant"):
        st.markdown(assistant_response)

# Option to clear chat history
# if st.button("Clear Chat"):
#     st.session_state.messages = []
#     memory.clear()
#     st.experimental_rerun()
