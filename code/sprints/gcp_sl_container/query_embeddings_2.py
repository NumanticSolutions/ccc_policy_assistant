# [2412]
#

import ollama
import chromadb

from chromadb.config import Settings



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
        db_path (str)       : Full path specifying where the location of the embeddings database. If
                                https is found in the string, it is assumed that the database is on GCS.
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
        ### Check if local or in the cloud
        if self.path_db.find("https") >= 0:

            # Create a Chroma client with the service URL and API token
            self.client = chromadb.HttpClient(host=self.path_db,
                                              port=443,
                                              ssl=True,
                                              settings=Settings(
                                                  chroma_client_auth_provider="chromadb.auth.token_authn.TokenAuthClientProvider",
                                                  chroma_client_auth_credentials="abcdefghijklmnopqrstuvwxyz",
                                                  anonymized_telemetry=False))
        else:

            # Embeddings database stored locally
            self.client = chromadb.PersistentClient(path=self.path_db)

        # Set up collection
        self.collection = self.client.get_collection(name=collection_name)

        # Get documents
        self.documents = self.collection.get()

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


# # db_path = '/Users/numantic/projects/ccc/embedding_wikipedia/db'
# db_path = ("/Users/stephengodfrey/OneDrive - numanticsolutions.com"
#                    "/Engagements/Projects/ccc_policy_assistant/data/embeddings")
# # collection_name = 'docs'
# collection_name = "crawl_docs1"
#
# qe = QueryEmbeddings(db_path, collection_name)
#
# prompt = "what college is designated a Center of Excellence in bioprocessing?"
# # prompt = "what colleges have surf teams?"
# # prompt = "Does MiraCosta College have a surf team?"
#
# print("prompt :")
# print(prompt + "\n")
#
# documents, metadatas = qe.query_collection(prompt)
#
# print("query results :")
# for i, doc in enumerate(documents):
#     print(str(i) + "\n" + doc)
#     print(metadatas[i]+"\n")


