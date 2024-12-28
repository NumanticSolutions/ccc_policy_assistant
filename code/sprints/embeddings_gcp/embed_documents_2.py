# [2412] n8
#
import os, sys

import pandas as pd

import ollama
import chromadb
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate

sys.path.insert(0, "../../embedding")
from chunk_text import ChunkText

sys.path.insert(0, "../../utils")
import gcp_tools as gct

class EmbedDocuments:

    """Create a named database and embed chunks and corresponding metedata.

    The class contsructor creates a new persistent ChromaDB database at a given
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

        # Control what th

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


        chunker = ChunkText()

        self.chunks, self.urls = chunker.chunk_dataframe(self.input_df)

        # Note how many chunks
        self.len_chunks = len(self.chunks)


    def embed(self,
              embed_model="mxbai-embed-large",
              meta_key="url",
              is_verbose="True"):
        '''
        Create the embeddings
        :param chunks:
        :param metas:
        :param embed_model:
        :param meta_key:
        :param is_verbose:
        :return:
        '''

        #### Step 1: Establish embedding object
        self.client = chromadb.PersistentClient(path=self.embeddings_path)

        # Check if collection name exists -
        #### Need to add more functionality on how to handle this case
        for c in self.client.list_collections():
            if c.name == self.collection_name:
                print("collection {} already exists; It will be overwritten".format(c.name))
                self.client.delete_collection(name=c.name)

        # Create a new collection
        self.collection = self.client.create_collection(name=self.collection_name)


        ##### Step 2: Create embeddings and save to a ChromaDB
        for i, chunk in enumerate(self.chunks):
            if is_verbose:
                if i % 10:
                    print(i, end=" ", flush=True)
            response = ollama.embeddings(model=embed_model,
                                         prompt=chunk)
            embedding = response["embedding"]
            self.collection.add(
                ids = [str(i)],
                embeddings = [embedding],
                documents = [chunk],
                metadatas = [{ meta_key : self.urls[i] }]
            )

    def copy_embeddings_to_gcs(self):
        '''
        Method to copy embeddings from a local store to GCS.
        :return:
        '''

        gct.upload_directory_to_gcs(local_directory=self.embeddings_path,
                                    gcs_bucket_name=self.gcs_bucket_name,
                                    gcs_directory=self.gcs_folder)

