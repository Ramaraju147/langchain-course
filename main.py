import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from operator import itemgetter

load_dotenv()

# Setup
embeddings = OpenAIEmbeddings()
vectorstore = PineconeVectorStore(
    pinecone_api_key=os.getenv("PINECONE_API_KEY"),
    index_name=os.getenv("INDEX_NAME"),
    embedding=embeddings
)
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.9)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

def format_docs(docs):
    return "\n\n".join([doc.page_content for doc in docs])

# Template
template = """
Use the following pieces of context to answer the question at the end. 
If you don't know the answer, say you don't know, don't try to make up an answer.

Context: {context}

Question: {question}

Answer:"""

# Create the prompt template
prompt = ChatPromptTemplate.from_template(template)

# Create chain
def create_chain():
    return (
        RunnablePassthrough.assign(
            context=itemgetter("question") | retriever | format_docs
        )
        | prompt
        | llm
        | StrOutputParser()
    )

if __name__ == '__main__':
    print("Asking question...")
    
    question = "What is the pinecone vector store?"
    chain = create_chain()
    answer = chain.invoke({"question": question})
    
    print(f"\nAnswer:\n{answer}")