# [2412] n8
#
import os, sys

import pandas as pd


import langchain
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.document_loaders import TextLoader, UnstructuredPDFLoader
from langchain.prompts import PromptTemplate
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores.chroma import Chroma
from langchain_google_vertexai import VertexAI, VertexAIEmbeddings
import vertexai

from langchain_community.document_loaders import DataFrameLoader

# sys.path.insert(0, "../../embedding")
# from chunk_text import ChunkText

sys.path.insert(0, "../../utils")
import gcp_tools as gct
import authentication as auth

class EmbedDocuments:

    """Create a named database and embed chunks and corresponding metedata.

    The class constructor creates a new persistent ChromaDB database at a given
    path location and with a given collection name. If a database and collection
    already exist the constructor will fail.

    The 'embed()' method will use a specified model to vector embed chunks and
    corresponding metadatas. The 'meta_key' can be used to specify the key name
    and 'is_verbose' can toggle printing information during embedding. This method
    does not return anything.

    ():
        full_path (str)       : Full path specifying where the database is to be created.
        collection_name (str) : Name of the collection to be created in the database.

    embed():
        chunks [strs]     : The text chunks to be vector embedded in the database
        metas [strs]      : The metadata for each of the chunks to be embedded
        model (str)       : Embedding model to use, such as 'mxbai-embed-large'
        meta_key (str)    : Key to use for the metadats, such as 'url'
        is_verbose (bool) : Print updates during embedding

    """

    def __init__(self,
                 input_text_path,
                 embeddings_path,
                 collection_name,
                 gcs_bucket_name,
                 **kwargs):

        # Set project and location
        self.project_id = "eternal-bongo-435614-b9"
        self.location = "us-central1"

        # Get credentials
        creds = auth.ApiAuthentication(cred_source="dotenv")
        os.environ["GOOGLE_API_KEY"] = creds.apis_configs["GOOGLE_API_KEY"]

        # Set models
        self.embedding_model = "textembedding-gecko@003"
        self.embedding_num_batch = 5

        # Assign class values based on inputs
        self.input_text_path = input_text_path
        self.embeddings_path = embeddings_path
        self.collection_name = collection_name
        self.gcs_bucket_name = gcs_bucket_name

        # Text column
        self.text_col = "ptag_text"
        # Folder on GCS
        self.gcs_folder = ""

        # Update any keyword args
        self.__dict__.update(kwargs)

        # Initialize VertexAI
        vertexai.init(project=self.project_id, location=self.location)


    def get_input_filenames(self):
        '''
        Method to collect the filenames in the input_text_path - allowing for embedding multiple
        documents in a single run.
        :return:
        '''

        # Get csv files
        self.input_files = [f for f in os.listdir(self.input_text_path) if f.endswith(".csv")]


    def read_text_data(self):
        '''
        Method to read raw text csv files data into a dataframe.
        :return:
        '''

        # Read all files into a list of dataframes
        dfs = []
        for file in self.input_files:

            dfs.append(pd.read_csv(filepath_or_buffer=os.path.join(self.input_text_path, file)))

        # Create a single dataframe
        self.input_df = pd.concat(objs=dfs)

        # Ensure text column is string
        self.input_df[self.text_col] = self.input_df[self.text_col].astype(str)

        # Drop any rows with null in text column
        self.input_df = self.input_df.dropna(subset=[self.text_col])

        # Drop rows with identical text to others
        self.input_df = self.input_df.drop_duplicates(subset=[self.text_col])
        self.input_df = self.input_df.reset_index(drop=True)

        #####
        # We might want to reduce some redundant text by checking similarity of input texts
        # Need to figure out how to handle additional documents - do we add to the same collection?

    def chunk_input_text(self):
        '''
        Method to chunk the input text data into chunks.

        :return:
        '''

        self.split_docs = []

        # Create a loader and load documents
        loader = DataFrameLoader(self.input_df, page_content_column=self.text_col)

        text_splitter = CharacterTextSplitter(chunk_size=8192, chunk_overlap=128)
        self.split_docs.extend(text_splitter.split_documents(loader.load()))


    def embed(self):
        '''
        Create the embeddings

        '''

        # vertexai.init(project=self.project_id, location=self.location)

        # self.embeddings = VertexAIEmbeddings(model_name=self.embedding_model,
        #                                      batch_size=self.embedding_num_batch)

        self.embeddings = VertexAIEmbeddings(model_name=self.embedding_model)

        db = Chroma.from_documents(
            documents=self.split_docs, embedding=self.embeddings,
            persist_directory=self.embeddings_path, collection_name=self.collection_name)

    def copy_embeddings_to_gcs(self):
        '''
        Method to copy embeddings from a local store to GCS.
        :return:
        '''

        gct.upload_directory_to_gcs(local_directory=self.embeddings_path,
                                    gcs_project_id=self.project_id,
                                    gcs_bucket_name=self.gcs_bucket_name,
                                    gcs_directory=self.gcs_folder)

