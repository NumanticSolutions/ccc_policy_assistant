# [2412] n8
#

from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate
from langchain_ollama import OllamaLLM
from transformers import AutoTokenizer
import ollama

import streamlit as st
import numpy as np

st.title("CCC Bot")


with st.sidebar:
    request_succinct = st.checkbox("Request succinct")
    st.text('The Request Succinct option appends this text to the prompt : "Please provide only a one or two word answer. Be as succinct as possible when answering."')
    if st.button("Save Transcript"):
        print("save transcript")
        f1 = open("ccc_bot-transcript.txt", "w")
        for message in st.session_state.messages:
            f1.write(message["role"] + " : " + message["content"] + "\n")
            f1.write("--------------------------------------------------------\n\n")
        f1.close()


st.text("Example question : What is the California community college with the largest student enrollment?")
st.text("Note : make sure the Ollama client is running while using this application.")

st.divider()


llm = OllamaLLM(model="phi3")

template = """
Question: {question}
Answer: 
"""
prompt = PromptTemplate(template=template, input_variables=["question"])

chain = prompt | llm


# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("What shall we discuss?"):

    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)

    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        if request_succinct:
            prompt += " Please provide only a one or two word answer. Be as succinct as possible when answering. "
        result = chain.invoke(prompt)

        st.markdown(result)

    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": result})


# [ ] add succinct setting to transcript
# [ ] json formatted transcript
# [ ] data collection on impact of succinct option (word counts with and without)
# [ ] other options for the sidebar / chat session
#
