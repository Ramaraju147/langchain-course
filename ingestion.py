import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore



load_dotenv()

if __name__ == '__main__':
    print("Ingesting...")

    loader = TextLoader('./mediumblog1.txt')
    documents = loader.load()

    text_splitter = CharacterTextSplitter(separator="\n",
                                        chunk_size=1000,
                                        chunk_overlap=200)
    
    embeddings = OpenAIEmbeddings()
    texts = text_splitter.split_documents(documents)

    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    index_name = os.getenv("INDEX_NAME")
    vectorstore = PineconeVectorStore.from_texts(
        [t.page_content for t in texts],
        embeddings,
        pinecone_api_key=pinecone_api_key,
        index_name=index_name
)



