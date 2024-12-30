# [2412]
#

import ollama
import chromadb

from chromadb.config import Settings


print("query embeddings")

class QueryEmbeddings:

    """Get a previously created database and query for vector embedded neighbors.

    The class contsructor gets an existing persistent ChromaDB database at a given
    path location and with a given collection name. 

    The 'query_collection()' method will use a specified model to vector embed an 
    input prompt. The 'prompt_str' is the input prompt as a string.  The 'n_neighbors'
    determines the number of results to return from the query. The 'meta_key' can be 
    used to specify the key name for the returned metadatas. This method returns a list
    of documents and metadatas, both as strings.

    ():
        full_path (str)       : Full path specifying where the database is to be created.
        collection_name (str) : Name of the collection to be created in the database.

    query_collection():
        prompt_str (str)  : The input prompt as a string to query the document collection
        model (str)       : Embedding model to use, such as 'mxbai-embed-large'
        n_neighbors (int) :
        meta_key (str)    : Key to use for the metadats, such as 'url'

    """

    def __init__(self, path_db, collection_name):
        self.path_db = path_db
        self.collection_name = collection_name

        ### Changes to move embeddings to GCP
        self.client = chromadb.PersistentClient(path=self.path_db)

        self.collection = self.client.get_collection(name=collection_name)

        self.documnts = self.collection.get()

    def query_collection(self, prompt_str, 
                         embed_model="mxbai-embed-large", n_neighbors=3, meta_key="url"):
        embedded_prompt = ollama.embeddings(
            prompt=prompt_str,
            model=embed_model
        )

        results = self.collection.query(
            query_embeddings=[embedded_prompt["embedding"]],
            n_results=n_neighbors
        )       

        documents, metadatas = [], []
        for i in range(n_neighbors):
            documents.append(results['documents'][0][i])
            metadatas.append(str(results['metadatas'][0][i][meta_key]))

        return documents, metadatas


db_path = '/Users/numantic/projects/ccc/embedding_wikipedia/db'
collection_name = 'docs'

qe = QueryEmbeddings(db_path, collection_name)

prompt = "what college is designated a Center of Excellence in bioprocessing?"
# prompt = "what colleges have surf teams?"
# prompt = "Does MiraCosta College have a surf team?"

print("prompt :")
print(prompt + "\n")

documents, metadatas = qe.query_collection(prompt)

print("query results :")
for i, doc in enumerate(documents):
    print(str(i) + "\n" + doc)
    print(metadatas[i]+"\n")


