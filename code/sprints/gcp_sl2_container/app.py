# [2412]
#
# * A retrieval-centric interface for CCC-PA
#


import streamlit as st
import json
import sys, os
import re
import datetime

from langchain_core.tools import tool
from langgraph.graph import MessagesState, StateGraph
from langchain_core.messages import SystemMessage
from langgraph.prebuilt import ToolNode

from langchain_chroma import Chroma
from langchain_google_vertexai import VertexAI, VertexAIEmbeddings, ChatVertexAI

from langgraph.graph import END
from langgraph.prebuilt import ToolNode, tools_condition

import vertexai

sys.path.insert(0, "../../utils")
import gcp_tools as gct
import authentication as auth

# from google.cloud import storage

import chatbot as cb

#########################################
# class BotCCCGlobals:
#     """ BotCCCGlobals class holds globals for a chatbot based on a
#     pre-trained model (phi3) and document retrieval from a vector
#     database (ChromaDB). It includes a range of defaults, including
#     an (optional) succinctness constraint, the default question, the
#     file name for any saved transcripts, and the path & collection
#     name for the vector database. """
#
#     def __init__(self):
#
#         self.transcript_name_base = "cccbot_transcript"
#         self.transcript_path = "./local_transcripts/"
#         self.transcript_gcs_bucket = "cccbot-transcripts"
#         self.transcript_gcs_directory = ""
#         self.chroma_collection_name = "crawl_docs1"
#
#         self.gcp_project_id = "eternal-bongo-435614-b9"
#         self.gcp_location = "us-central1"
#         self.gcs_embeddings_bucket_name = "ccc-chromadb-vai"
#         self.gcs_embeddings_directory = ""
#         self.embedding_model = "textembedding-gecko@003"
#         self.embedding_num_batch = 5
#         self.embeddings_local_path = "./local_chromadb/"
#         # self.embeddings_local_path = ("/Users/stephengodfrey/OneDrive - numanticsolutions.com"
#         #                               "/Engagements/Projects/ccc_policy_assistant/data/embeddings-vai")
#
#         self.llm_model = "gemini-1.5-pro"
#         self.llm_model_max_output_tokens = 2048
#         self.llm_model_temperature = 1.0   #old value 0.2
#         self.llm_model_top_p = 0.8
#         self.llm_model_top_k = 40
#         self.llm_model_verbose = True
#
#         self.retriever_search_type = "similarity"
#         self.retriever_search_kwargs = {"k": 3}
#         self.doc_search_retrieval_k = 4
#
#         self.default_question = "What shall we discuss?"
#         # self.be_succinct = ('Please provide only a one or two word answer' +
#         #                     'Be as succinct as possible when answering. ')
#         # self.prompt_template = """
#         #                         You are a California Community College AI assistant.
#         #                         You're tasked to answer the question given below,
#         #                         but only based on the context provided.
#         #                         context: {context}
#         #                         question: {input}
#         #                         If you cannot find an answer ask the user to rephrase the question.
#         #                         answer:
#         #                        """
#
#         self.prompt_template = ("You are a California Community College AI assistant. "
#                                 "Use the following pieces of context to answer the question at the end. "
#                                 "If you don't know the answer, just say that you don't know, "
#                                 "don't try to make up an answer. ")
#         self.chat_hist_memory_key = "chat_history"
#         self.chat_hist_return_messages = True
#         self.chat_hist_output_key = "answer"
#
#         self.conv_chain_chain_type="stuff"
#         self.conv_chain_chain_type_verbose=True
#         self.conv_chain_chain_type_return_source_documents=True
#
#
#
#
# def setup_vectorstore():
#     '''
#     Function to set up vector store. This returns a vector store that
#     can be set in the Streamlit session eliminating the automatic
#     refresh that would happen with each new question
#
#     '''
#
#     # Download files from GCP
#     gct.download_directory_from_gcs(gcs_project_id=bot_params.gcp_project_id,
#                                     gcs_bucket_name=bot_params.gcs_embeddings_bucket_name,
#                                     gcs_directory="",
#                                     local_directory=bot_params.embeddings_local_path)
#
#     # Load embeddings and persisted data
#     embeddings = VertexAIEmbeddings(model_name=bot_params.embedding_model)
#
#     # Load Chroma data from local persisted directory
#     db = Chroma(persist_directory=bot_params.embeddings_local_path,
#                 collection_name=bot_params.chroma_collection_name,
#                 embedding_function=embeddings)
#     return db
#
# @tool(response_format="content_and_artifact")
# def retrieve(query: str):
#     """Retrieve information related to a query."""
#     retrieved_docs = vector_store.similarity_search(query,
#                                                     k=bot_params.doc_search_retrieval_k)
#     serialized = "\n\n".join(
#         (f"Source: {doc.metadata}\n" f"Content: {doc.page_content}")
#         for doc in retrieved_docs
#     )
#     return serialized, retrieved_docs
#
# # Step 1: Generate an AIMessage that may include a tool-call to be sent.
# def query_or_respond(state: MessagesState):
#     """Generate tool call for retrieval or respond."""
#     llm_with_tools = llm.bind_tools([retrieve])
#     response = llm_with_tools.invoke(state["messages"])
#     # MessagesState appends messages to state instead of overwriting
#     return {"messages": [response]}
#
# # Step 3: Generate a response using the retrieved content.
# def generate(state: MessagesState):
#     """Generate answer."""
#     # Get generated ToolMessages
#     recent_tool_messages = []
#     for message in reversed(state["messages"]):
#         if message.type == "tool":
#             recent_tool_messages.append(message)
#         else:
#             break
#     tool_messages = recent_tool_messages[::-1]
#
#     # Format into prompt
#     docs_content = "\n\n".join(doc.content for doc in tool_messages)
#     system_message_content = (f"{bot_params.prompt_template}"
#                               "\n\n"
#                               f"{docs_content}")
#
#     conversation_messages = [
#         message
#         for message in state["messages"]
#         if message.type in ("human", "system")
#         or (message.type == "ai" and not message.tool_calls)
#     ]
#     prompt = [SystemMessage(system_message_content)] + conversation_messages
#
#     # Run
#     response = llm.invoke(prompt)
#     return {"messages": [response]}
#
# def setup_conversation_graph():
#     '''
#     Set up a Streamlit conversational graph
#
#     :return:
#     '''
#
#     ## Step 12: Create graph_builder
#     graph_builder = StateGraph(MessagesState)
#
#     # Step 14: Execute the retrieval.
#     tools = ToolNode([retrieve])
#
#     # Step 16: Build graph
#     graph_builder.add_node(query_or_respond)
#     graph_builder.add_node(tools)
#     graph_builder.add_node(generate)
#
#     graph_builder.set_entry_point("query_or_respond")
#     graph_builder.add_conditional_edges(
#         "query_or_respond",
#         tools_condition,
#         {END: END, "tools": "tools"},
#     )
#     graph_builder.add_edge("tools", "generate")
#     graph_builder.add_edge("generate", END)
#
#     graph = graph_builder.compile()
#
#     return graph
#
# ############################################
#
# ########## Set up Chatbot
#
# creds = auth.ApiAuthentication()
#
# # LangSmith
# os.environ["LANGCHAIN_TRACING_V2"] = "true"
# os.environ["LANGCHAIN_API_KEY"] = creds.apis_configs["LANGCHAIN_API_KEY"]
#
# ### Step 1: Establish parameters
# bot_params = BotCCCGlobals()
#
# ### Step 2: Initialize Vertex AI
# vertexai.init(project=bot_params.gcp_project_id,
#               location=bot_params.gcp_location)
#
# ### Step 3. Instantiate an LLM
# # llm = ChatVertexAI(model="gemini-1.5-flash")
# llm = ChatVertexAI(model=bot_params.llm_model)
#
# ### Step 4. Establish an embeddings model
# # embeddings = VertexAIEmbeddings(model="text-embedding-004")
# embeddings = VertexAIEmbeddings(model=bot_params.embedding_model)

# ### Step 5: Establish vector store
# vector_store = cb.setup_vectorstore()
#
# ### Step 6: Set up conversational graph
# graph = cb.setup_conversation_graph()


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

# if "vectorstore" not in st.session_state:
#     st.session_state.vectorstore = setup_vectorstore()

# if "conversational_chain" not in st.session_state:
#     st.session_state.conversational_chain = setup_conversation_graph()

#####################################
# vector_store = setup_vectorstore()
# graph = setup_conversation_graph()

bot = cb.CCCPolicyAssistant()
graph = bot.graph

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
    # response = st.session_state.conversational_chain({"question": user_input})
    for step in graph.stream(
            {"messages": [{"role": "user", "content": user_input}]},
            stream_mode="values",
    ):
        # step["messages"][-1].pretty_print()
        # st.session_state.messages.append({"role": "assistant", "content": step["messages"][-1]})

        try:
            #####= Find the source URLs
            pat = r"{'url': '.+}"
            for msg in step["messages"][-2]:
                if msg[0] == "content":
                    # Find all URLs in the source AI responses
                    source_urls = re.findall(pat, msg[1])
                    # Remove unwanted characters
                    source_urls = [s.replace("{'url': '","").replace("'}","") for s in source_urls]
                    # Remove duplicates
                    source_urls = set(source_urls)

            ############ Get the AI response
            for msg in step["messages"][-1]:
                if msg[0] == "content":
                    ai_response = msg[1]

        except:
            pass

    res_msg = ("{} \nThese resources might be helpful:\n {}")
    assistant_response = res_msg.format(ai_response,  source_urls)

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
