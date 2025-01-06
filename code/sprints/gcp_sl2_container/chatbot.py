##########
# Code for the chatbot intended to be run using Streamlit

from langchain_core.tools import tool
from langgraph.graph import MessagesState, StateGraph
from langchain_core.messages import SystemMessage
from langgraph.prebuilt import ToolNode

from langchain_chroma import Chroma
from langchain_google_vertexai import VertexAI, VertexAIEmbeddings, ChatVertexAI

from langgraph.graph import END
from langgraph.prebuilt import ToolNode, tools_condition

import vertexai

import sys, os
sys.path.insert(0, "../../utils")
import gcp_tools as gct
import authentication as auth

class CCCPolicyAssistant:
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

        ### Step 5. Establish an embeddings model
        self.graph = self.setup_conversation_graph()


    def setup_vectorstore(self):
        '''
        Function to set up vector store. This returns a vector store that
        can be set in the Streamlit session eliminating the automatic
        refresh that would happen with each new question

        '''

        # Download files from GCP
        gct.download_directory_from_gcs(gcs_project_id=self.gcp_project_id,
                                        gcs_bucket_name=self.gcs_embeddings_bucket_name,
                                        gcs_directory="",
                                        local_directory=self.embeddings_local_path)

        # Load embeddings and persisted data
        embeddings = VertexAIEmbeddings(model_name=self.embedding_model)

        # Load Chroma data from local persisted directory
        db = Chroma(persist_directory=self.embeddings_local_path,
                    collection_name=self.chroma_collection_name,
                    embedding_function=embeddings)
        return db


    @tool(response_format="content_and_artifact")
    def retrieve(self, query: str):
        """Retrieve information related to a query."""
        retrieved_docs = self.vector_store.similarity_search(query,
                                                        k=self.doc_search_retrieval_k)

        serialized = "\n\n".join(
            (f"Source: {doc.metadata}\n" f"Content: {doc.page_content}")
            for doc in retrieved_docs
        )
        return serialized, retrieved_docs


    def query_or_respond(self, state: MessagesState):
        '''
        # Step 1: Generate an AIMessage that may include a tool-call to be sent.
        Generate tool call for retrieval or respond."
        '''
        llm_with_tools = self.llm.bind_tools([self.retrieve])
        response = llm_with_tools.invoke(state["messages"])
        # MessagesState appends messages to state instead of overwriting
        return {"messages": [response]}


    def generate(self, state: MessagesState):
        '''
        Generate answer.# Step 3: Generate a response using the retrieved content.
        '''
        # Get generated ToolMessages
        recent_tool_messages = []
        for message in reversed(state["messages"]):
            if message.type == "tool":
                recent_tool_messages.append(message)
            else:
                break
        tool_messages = recent_tool_messages[::-1]

        # Format into prompt
        docs_content = "\n\n".join(doc.content for doc in tool_messages)
        system_message_content = (f"{self.prompt_template}"
                                  "\n\n"
                                  f"{docs_content}")

        conversation_messages = [
            message
            for message in state["messages"]
            if message.type in ("human", "system")
            or (message.type == "ai" and not message.tool_calls)
        ]
        prompt = [SystemMessage(system_message_content)] + conversation_messages

        # Run
        response = self.llm.invoke(prompt)
        return {"messages": [response]}


    def setup_conversation_graph(self):
        '''
        Set up a Streamlit conversational graph

        :return:
        '''

        ## Step 12: Create graph_builder
        graph_builder = StateGraph(MessagesState)

        # Step 14: Execute the retrieval.
        tools = ToolNode([self.retrieve])

        # Step 16: Build graph
        graph_builder.add_node(self.query_or_respond)
        graph_builder.add_node(tools)
        graph_builder.add_node(self.generate)

        graph_builder.set_entry_point("query_or_respond")
        graph_builder.add_conditional_edges(
            "query_or_respond",
            tools_condition,
            {END: END, "tools": "tools"},
        )
        graph_builder.add_edge("tools", "generate")
        graph_builder.add_edge("generate", END)

        graph = graph_builder.compile()

        return graph
