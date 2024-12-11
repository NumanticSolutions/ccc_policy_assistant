# [2412]
#

#
# * https://ollama.com/blog/embedding-models
#

import ollama
import chromadb

from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate


print("embedding - 02 :")

documents = []
with open("ccc_enrollments_to_narratives.txt", "r") as file:
    for line in file:
        if len(line) > 0:
            documents.append(line)


# create an in-memory database for testing, ie not persistent or stored locally
#
client = chromadb.Client()

# this will create a persistent client in the specified directory
#
# client = chromadb.PersistentClient(path="/Users/numantic/sprints/mixed/embedding_01/db")


collection = client.create_collection(name="docs")

print("embedding documents :")

# store each document in a vector embedding database
for i, d in enumerate(documents):
    if i % 10:
        print(i, end=" ", flush=True)
    response = ollama.embeddings(model="mxbai-embed-large", prompt=d)
    embedding = response["embedding"]
    collection.add(
        ids=[str(i)],
        embeddings=[embedding],
        documents=[d]
    )

print()

# example prompts
#
prompt = "What is the enrollment of Fresno City College?"

print("prompt :")
print(prompt + "\n")

# generate an embedding for the prompt and retrieve the most relevant doc
response = ollama.embeddings(
  prompt=prompt,
  model="mxbai-embed-large"
)
results = collection.query(
  query_embeddings=[response["embedding"]],
  n_results=3
  # n_results=1
)

data = results['documents'][0][0]
data += results['documents'][0][1]
data += results['documents'][0][2]

print("most similar document :")
print(data + "\n")


llm = OllamaLLM(model="phi3")

template = """
Question: {question}
Answer: 
"""
prompt_template = PromptTemplate(template=template, input_variables=["question"])

chain = prompt_template | llm
augmented_prompt = " Using this data : " + data + ", answer this question : " + prompt

print("augmented prompt :")
print(augmented_prompt + "\n")

result = chain.invoke(augmented_prompt)

print("augmented result :")
print(result + "\n")


result = chain.invoke(prompt)

print("phi3 baseline result :")
print(result + "\n")

