############
# Experimental code creating a LangCHain RAG Bot class

import sys, os

from langchain_core.documents import Document
from typing_extensions import List, TypedDict
from langgraph.graph import START, StateGraph
from langchain import hub
from langchain_google_vertexai import VertexAI, VertexAIEmbeddings, ChatVertexAI
import vertexai
from langchain_chroma import Chroma
from langchain_core.tools import tool



sys.path.insert(0, "../../utils")
import authentication as auth



# Define state for application
class State(TypedDict):
    question: str
    context: List[Document]
    answer: str


class CCCPolicyAssistant:
    '''
    First draft of a CCC Policy Policy Assistant Chatbot using
    LangChain components

    Reference: https://python.langchain.com/docs/tutorials/rag/
    '''

    def __init__(self):

        self.transcript_name_base = "cccbot_transcript"
        self.transcript_path = "./local_transcripts/"
        self.transcript_gcs_bucket = "cccbot-transcripts"
        self.transcript_gcs_directory = ""
        self.chroma_collection_name = "crawl_docs1"

        self.gcp_project_id = "eternal-bongo-435614-b9"
        self.gcp_location = "us-central1"
        self.gcs_embeddings_bucket_name = "ccc-chromadb-vai"
        self.gcs_embeddings_directory = ""
        self.embedding_model = "textembedding-gecko@003"
        # self.embedding_model = "text-embedding-004"
        self.embedding_num_batch = 5
        # self.embeddings_local_path = "./local_chromadb/"
        self.embeddings_local_path = ("/Users/stephengodfrey/OneDrive - numanticsolutions.com"
                                      "/Engagements/Projects/ccc_policy_assistant/data/embeddings-vai")

        self.llm_model = "gemini-1.5-pro"
        # self.llm_model = "gemini-1.5-flash"
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

        self.prompt_template = ("You are a California Community College AI assistant. "
                                "Use the following pieces of context to answer the question at the end. "
                                "If you don't know the answer, just say that you don't know, "
                                "don't try to make up an answer. ")
        self.chat_hist_memory_key = "chat_history"
        self.chat_hist_return_messages = True
        self.chat_hist_output_key = "answer"

        self.conv_chain_chain_type="stuff"
        self.conv_chain_chain_type_verbose=True
        self.conv_chain_chain_type_return_source_documents=True

        self.doc_search_retrieval_k = 4

        ########## Set up Chatbot
        creds = auth.ApiAuthentication()

        # LangSmith
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_API_KEY"] = creds.apis_configs["LANGCHAIN_API_KEY"]

        ### Step 1: Initialize Vertex AI
        vertexai.init(project=self.gcp_project_id,
                      location=self.gcp_location)

        ### Step 2. Instantiate an LLM
        self.llm = ChatVertexAI(model=self.llm_model)

        ### Step 3. Establish an embeddings model
        self.embeddings = VertexAIEmbeddings(model=self.embedding_model)

        ### Step 4. Establish an embeddings model
        self.vector_store = self.setup_vectorstore()

        ### Step 5. Set up prompt
        self.prompt = hub.pull("rlm/rag-prompt")

        ### Step 5. Establish an embeddings model
        self.graph = self.setup_conversation_graph()

    def setup_vectorstore(self):
        '''
        Function to set up vector store. This returns a vector store that
        can be set in the Streamlit session eliminating the automatic
        refresh that would happen with each new question

        '''

        # Download files from GCP
        # gct.download_directory_from_gcs(gcs_project_id=self.gcp_project_id,
        #                                 gcs_bucket_name=self.gcs_embeddings_bucket_name,
        #                                 gcs_directory=self.gcs_embeddings_directory,
        #                                 local_directory=self.embeddings_local_path)

        # Load embeddings and persisted data
        embeddings = VertexAIEmbeddings(model_name=self.embedding_model)

        # Load Chroma data from local persisted directory
        db = Chroma(persist_directory=self.embeddings_local_path,
                    collection_name=self.chroma_collection_name,
                    embedding_function=embeddings)
        return db

    # Define application steps
    def retrieve(self, state: State):
        """Retrieve information related to a query."""
        retrieved_docs = self.vector_store.similarity_search(state["question"])
        return {"context": retrieved_docs}


    def generate(self, state: State):
        docs_content = "\n\n".join(doc.page_content for doc in state["context"])
        messages = self.prompt.invoke({"question": state["question"], "context": docs_content})
        response = self.llm.invoke(messages)
        return {"answer": response.content}

    def setup_conversation_graph(self):
        # Compile application and test
        graph_builder = StateGraph(State).add_sequence([self.retrieve, self.generate])
        graph_builder.add_edge(START, "retrieve")
        graph = graph_builder.compile()
        return graph
