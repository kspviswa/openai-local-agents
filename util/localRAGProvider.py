import os
from typing import List
from langchain_ollama import OllamaEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

class LocalRAGProvider:
    def __init__(self, model_name='llama3.2', chunk_size=1000, chunk_overlap=200):
        self.model_name = model_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.embeddings = OllamaEmbeddings(model=self.model_name)
        self.vector_store = InMemoryVectorStore(self.embeddings)
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
        self.retriever = self.vector_store.as_retriever(search_kwargs={"k": 5})

    def add_documents(self, documents):
        docs : List[Document] = []
        for doc in documents:
            docs.append(Document(page_content=doc))
        chunks = self.text_splitter.split_documents(docs)
        self.vector_store.add_documents(chunks)
    
    def load_documents(self, path):
        documents = []
        if os.path.isdir(path):
            for file in os.listdir(path):
                with open(os.path.join(path, file), 'r') as f:
                    documents.append(f.read())
        elif os.path.isfile(path):
            with open(path, 'r') as f:
                documents = [f.read()]
        else:
            raise ValueError(f"Path {path} is neither a file nor a directory.")            
        self.add_documents(documents)
    
    def query(self, query_text, k=1):
        raw_results = self.vector_store.similarity_search(query_text, k=k)
        results = []
        for result in raw_results:
            results.append(result.page_content)
        return "\n".join(results)
    

if __name__ == "__main__":
    local_rag = LocalRAGProvider(model_name="nomic-embed-text:latest", chunk_size=1000, chunk_overlap=200)
    local_rag.load_documents("./kb/")
    print(local_rag.query("How do I stream the output using pydantic AI?"))