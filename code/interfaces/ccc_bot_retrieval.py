# [2412] 
#
# * A retrieval-centric interface for CCC-PA
#

from langchain_core.prompts import PromptTemplate
from langchain_ollama import OllamaLLM

import streamlit as st
import numpy as np

import json
import sys

sys.path.append('../embedding')
from query_embeddings import QueryEmbeddings

st.title("CCC Bot : Retrieval")


class BotCCCGlobals:

    """ BotCCCGlobals class holds globals for a chatbot based on a 
    pre-trained model (phi3) and document retrieval from a vector
    database (ChromaDB). It includes a range of defaults, including
    an (optional) succinctness constraint, the default question, the
    file name for any saved transcripts, and the path & collection 
    name for the vector database. """

    def __init__(self):
        self.default_question = "What shall we discuss?"
        self.be_succinct = ('Please provide only a one or two word answer' +
                            'Be as succinct as possible when answering. ')
        self.model = "phi3"
        self.transcript_name = "ccc_bot-retrieval_transcript"
        self.db_path = '/Users/numantic/projects/ccc/embedding_wikipedia/db'
        self.collection_name = 'docs'

bot = BotCCCGlobals()

qe = QueryEmbeddings(bot.db_path, bot.collection_name)


with st.sidebar:
    st.title("Retrieval")
    request_succinct = st.checkbox("Request succinct")
    st.text('The Request Succinct option appends this text to the prompt : "' +
             bot.be_succinct + '"')
    if st.button("Save Transcript"):
        print("save transcript")
        messages = []
        for i in range(len(st.session_state.messages)):
            message = st.session_state.messages[i].copy()
            j = int(i/2)
            message["succinct"] = st.session_state.succinct_hist[j]
            print(message)
            messages.append(message)
        with open(bot.transcript_name + '.json', 'w') as file:
            json.dump(messages, file)


st.text("""Example question : What are the responsibilities of the 
        board members of a California community college?""")
st.text("Note : make sure the Ollama client is running while using this application.")

st.divider()


llm = OllamaLLM(model=bot.model)

template = """
Question: {question}
Answer: 
"""
prompt = PromptTemplate(template=template, input_variables=["question"])

chain = prompt | llm


if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.succinct_hist = []

# displays the chat history when app is rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# if prompt := st.chat_input(initial_question):
if prompt := st.chat_input(bot.default_question):

    st.chat_message("user").markdown(prompt)

    st.session_state.messages.append({"role": "user", "content": prompt})

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        documents, metadatas = qe.query_collection(prompt)

        if request_succinct:
            prompt += " " + bot.be_succinct
            print(prompt)
            st.session_state.succinct_hist.append("True")
        else:
            print(prompt)
            st.session_state.succinct_hist.append("False")
        result = chain.invoke(prompt)

        for i, doc in enumerate(documents):
            result += "\n\n --- \n\n"
            result += documents[i]
            result += metadatas[i]

        st.markdown(result)

    st.session_state.messages.append({"role": "assistant", "content": result})


