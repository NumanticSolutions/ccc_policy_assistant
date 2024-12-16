# [2412] n8
#

import pandas as pd

import ollama
import chromadb
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate

from chunk_text import ChunkText


path_csv = "/Users/numantic/data/bulk/ccc/crawls/"
name_csv = "enwikipediaorg_2024Dec11_1.csv"

df = pd.read_csv(path_csv + name_csv)

chunker = ChunkText()

chunks, urls = chunker.chunk_dataframe(df)

print("len chunks : " + str(len(chunks)))


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

    def __init__(self, full_path, collection_name):
        self.full_path = full_path
        self.collection_name = collection_name

        self.client = chromadb.PersistentClient(path=self.full_path)

        self.collection = self.client.create_collection(name=collection_name)

    def embed(self, chunks, metas, 
              embed_model = "mxbai-embed-large", meta_key="url", is_verbose="True"):
        for i, chunk in enumerate(chunks):
            if is_verbose:
                if i % 10:
                    print(i, end=" ", flush=True)
            response = ollama.embeddings(model="mxbai-embed-large", prompt=chunk)
            embedding = response["embedding"]
            self.collection.add(
                ids = [str(i)],
                embeddings = [embedding],
                documents = [chunk],
                metadatas = [{ meta_key : metas[i] }]
            )
        
embedder = EmbedDocuments("/Users/numantic/projects/ccc/embedding_wikipedia/db", "docs")

embedder.embed(chunks, urls)

print("\ndocuments embedded\n")

