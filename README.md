## Design and Implementation of LangChain Expression Language (LCEL) Expressions

### AIM:
To design and implement a LangChain Expression Language (LCEL) expression that utilizes at least two prompt parameters and three key components (prompt, model, and output parser), and to evaluate its functionality by analyzing relevant examples of its application in real-world scenarios.

### PROBLEM STATEMENT:

### DESIGN STEPS:
#### STEP 1:
Import required libraries and add the API key for OPENAI.

### STEP 2:
Choose the components: a prompt, a model, and an output parser.

### STEP 3:
Create a simple or complex chain using these components.

#### STEP 4:
Provide input values and run the chain.

### STEP 5: 
Print the output.

### PROGRAM:

SIMPLE CHAIN
```
import os
import openai

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file
openai.api_key = os.environ['OPENAI_API_KEY']

from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.schema.output_parser import StrOutputParser

prompt = ChatPromptTemplate.from_template(
    "write {number} poems about {topic}"
)
model = ChatOpenAI()
output_parser = StrOutputParser()

chain = prompt | model | output_parser
result = chain.invoke({"number": "2", "topic": "nature"})
result
```

COMPLEX CHAIN
```
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import DocArrayInMemorySearch
vectorstore = DocArrayInMemorySearch.from_texts(
    ["John studies engineering", "John works three part-time jobs", "John watches TV shows in his free time"],
    embedding=OpenAIEmbeddings()
)
retriever = vectorstore.as_retriever()
retriever.get_relevant_documents("what does John study?")
retriever.get_relevant_documents("How many part-time jobs does John work at?")
template = """Answer the question based only on the following context:
{context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)
from langchain.schema.runnable import RunnableMap

chain = RunnableMap({
    "context": lambda x: retriever.get_relevant_documents(x["question"]),
    "question": lambda x: x["question"]
}) | prompt | model | output_parser

chain.invoke({"question": "How many part-time jobs does John handle?"})

chain.invoke({"question": "What is John's hobby?"})
```
### OUTPUT:
SIMPLE CHAIN

![image](https://github.com/user-attachments/assets/141df776-19da-4f0b-b3c1-b70252d9cfbd)

COMPLEX CHAIN

![image](https://github.com/user-attachments/assets/d460f0d8-adb4-411d-8aa0-483bdd85bba7)

### RESULT:
The implemented LCEL expression takes at least two prompt parameters, processes them using a model, and formats the output with a parser, demonstrating its effectiveness through real-world examples.
