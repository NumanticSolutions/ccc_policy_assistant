# [2412]
#
# * A retrieval-centric interface for CCC-PA
#


import streamlit as st
import json
import sys, os
import re
import datetime

# sys.path.insert(0, "../../utils")
# import gcp_tools as gct
# import authentication as auth

# import rag_bot_1 as rb1
from rag_bot_1 import CCCPolicyAssistant

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


########## Handle conversations in Streamlit
# Build session components if needed
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "bot" not in st.session_state:
    # st.session_state["bot"] = rb1.CCCPolicyAssistant()
    st.session_state["bot"] = CCCPolicyAssistant()
    # st.write(st.session_state["bot"])

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
    response = st.session_state["bot"].graph.invoke({"question": user_input})
    # response = bot.graph.invoke({"question": user_input})

    # Extract metadata links
    context_urls = []
    for doc in response["context"]:
        if "url" in doc.metadata.keys():
            context_urls.append(doc.metadata["url"])
    context_urls = list(set(context_urls))

    # Converted URLs to a Markdown list
    s = "\n"
    for i in context_urls:
        s += "- " + i + "\n"

    # Extract the AI response and add URls for context
    ai_response = "{} These references might be useful: {}".format(response["answer"], s)

    # Store AI's response in the chat history
    st.session_state.messages.append({"role": "assistant", "content": ai_response})

    # Display assistant's message
    with st.chat_message("assistant"):
        st.markdown(ai_response)

# Option to clear chat history
# if st.button("Clear Chat"):
#     st.session_state.messages = []
#     memory.clear()
#     st.experimental_rerun()
