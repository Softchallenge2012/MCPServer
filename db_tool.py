
import sys, os

import openai
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest
from langchain_chroma import Chroma
# from models.llm import Prompt

import uuid
import pandas as pd
import asyncio
from typing import Any, Iterable, List, Optional, Tuple
from itertools import islice
import numpy as np
from enum import Enum

import tiktoken
from langchain_openai import OpenAIEmbeddings, ChatOpenAI, OpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


class Prompt(Enum):
    QA = """You are a medical assistant. Provide a concise (approx. 100 words) answer to the Question based ONLY on the provided Context. 
        Use an unbiased, professional, and journalistic tone. 
        If the context does not contain enough information to answer the question, state: "I could not find a relevant answer."
        Do not use any outside knowledge.

        Context:
        {context}

        Question:
        {question}
        """
    CAL = """Create an executable Python script to answer the question below using only the information from the provided Context.
        - Mark variables as False unless explicitly stated as True in the context.
        - The script should be self-contained and print the final answer.
        - Think step by step in comments.
        - DO NOT INCLUDE ANY INTRODUCTORY OR CONCLUDING TEXT.
        - YOUR RESPONSE MUST START AND END WITH PYTHON CODE.
        - DO NOT Use markdown code markers (```) unless it is absolutely necessary (even then, try to avoid them).
        - JUST THE RAW PYTHON CODE.

        Context:
        {context}

        Question:
        {question}
        """


class ChromaDB:
    def __init__(self, db_dir = 'data/chromadb', model_name='gpt-4o', embed_name='vertex-embedding'):
        self.db_dir = db_dir
        self.model_name = model_name
        self.embed_name = embed_name

    def create_db(self, db_dir = 'data/chromadb'):
        # Initializing an empty Chroma database or loading from disk
        try:
            if os.path.exists(db_dir):
                import shutil
                shutil.rmtree(db_dir)
            embed = OpenAIEmbeddings(
                model=self.embed_name,
                api_key="sk-1234",
                base_url="http://0.0.0.0:4000",
            )
            self.chroma_db = Chroma(persist_directory=db_dir, embedding_function=embed, collection_metadata={"hnsw:space": "cosine"})
        except Exception as e:
            print(f"Error creating Chroma DB: {e}")
            self.chroma_db = None

    def add_texts(self, texts: Iterable[str], metadatas: Optional[List[str]] = None) -> List[str]:

        # Split texts into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
        
        all_chunks = []
        all_urls = []
        
        for i, text in enumerate(texts):
            chunks = text_splitter.split_text(text)
            all_chunks.extend(chunks)
            if metadatas:
                all_urls.extend([metadatas[i]] * len(chunks))
            else:
                all_urls.extend([None] * len(chunks))
                
        df_document = pd.DataFrame({'text': all_chunks, 'url': all_urls})
        
        docs = [Document(page_content=row['text'], metadata={"url": row['url']}) for _, row in df_document.iterrows()]
        self.chroma_db.add_documents(docs)

        ids = self.chroma_db.get()['ids']

        return ids   

    def search(self, text: str = "", model=None):
        llm = model or ChatOpenAI(
            model=self.model_name,
            api_key="sk-1234",
            base_url="http://0.0.0.0:4000",
            temperature=0
        )
        retriever = self.chroma_db.as_retriever()
        prompt = ChatPromptTemplate.from_template(
            "Answer based on the context:\n\n{context}\n\nQuestion: {question}"
        )
        def format_docs(docs):
            return "\n\n".join(d.page_content for d in docs)

        chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
        answer = chain.invoke(text)
        source_docs = retriever.invoke(text)
        url = source_docs[0].metadata.get("url") if source_docs else None
        return {"prompt": text, "response": answer, "url": url}

    def get_embedding(self,text: str) -> List[float]:
        """Get the embedding for a given text."""
        # 1. Initialize the client
        # client = openai.OpenAI(api_key=os.environ['OPENAI_API_KEY'])
        client = openai.OpenAI(
            api_key="sk-1234",         # Match the master_key in config_vertex.yaml
            base_url="http://0.0.0.0:4000" # Your LiteLLM Proxy URL
        )
        # 2. Call the embeddings endpoint
        response = client.embeddings.create(
            model=self.embed_name, 
            input=text,
            encoding_format="float"
        )

        # 3. Access the data using dot notation (not brackets)
        vector = response.data[0].embedding
        return vector

class QdrantDB:
    def __init__(self, model_name, host=None, url=None, api_key=None, port=None, encoding_name="cl100k_base", context_length=1000):
        self.model_name = model_name

        self.encoding_name = encoding_name
        self.context_length = context_length
        
        if host and port:        
            self.database = QdrantClient(host=host, port=port)
        elif url and api_key:
            self.database = QdrantClient(url=url, api_key=api_key)
        else:
            raise ValueError("Please provide either host and port or url and api_key")

    def create_collection(self, collection_name: str, size = 768, distance = rest.Distance.COSINE):
        self.collection_name = collection_name
        self.database.create_collection(collection_name=collection_name, 
                                        vectors_config=rest.VectorParams(size=size, distance=distance))

    def delete_collection(self, collection_name: str):
        self.database.delete_collection(collection_name=collection_name)

    def add_texts(self, texts: Iterable[str], metadatas: Optional[List[str]] = None) -> List[str]:
        embeddings = []
        urls = []
        text_chunk = []
        for index, text in enumerate(texts):
            chunks, embd = self.get_embedding(text)
            urls.append(metadatas[index] if metadatas is not None else None)
            embeddings.append(embd)
            
            text_tokens = []
            for chunk in chunks:
                text_tokens.extend(chunk)
            text_chunk.append(text_tokens)

        ids = [uuid.uuid4().hex for _ in embeddings]
        self.database.upsert(collection_name=self.collection_name,
                                points=rest.Batch(ids=ids, 
                                vectors=embeddings,
                                payloads=self.build_payloads(text_chunk, urls)))

        return ids 

    def search(self, query: str, top_k: int=1, threshold: float=0.10, is_calculation: bool=False, generate_func=None):

        _, embedding = self.get_embedding(query)
        results_points = self.database.query_points(
            collection_name=self.collection_name,
            query=embedding,
            with_payload=True,
            limit=top_k,
        ).points
        results = [(self.get_docs_from_payload(result), result.score) for result in results_points]
        context = ""
        urls = set()
        confidence = 0
        for result in results:
            (text, url), score = result
            context += text
            if url:
                urls.add(url.split("?search")[0])
            confidence += score
        if len(results) > 0:
            confidence /= len(results)

        if confidence < threshold: # lambda threshold
            return {"prompt": query, "response": "I cannot answer query reliably. Please try again.", "url":urls ,"score": confidence}
        else:
            if generate_func is None:
                generate_func = globals().get("generate")
            if generate_func is None:
                return {"prompt": query, "response": context, "url": urls, "score": confidence}
            response = generate_func(prompt=query, context=context, meta = ', '.join(urls), promptStyle=Prompt.CAL if is_calculation else Prompt.QA)
            return {"prompt": query, "response": response, "url":urls ,"score": confidence}


    def get_docs_from_payload(self, vector: Any) -> Tuple[str, str]:
        return vector.payload.get("document"), vector.payload.get("url")


    def build_payloads(self, texts: List[List[float]], metadatas: Optional[List[str]]) -> List[dict]:
        """
        Build payloads for Qdrant
        :param texts: List of texts
        :param metadatas: List of metadata
        """
        payloads = []
        for i, text in enumerate(texts):
            text = self.get_tokenizer().decode(text)
            payloads.append({"document": text, "url": metadatas[i] if metadatas is not None else None})
        return payloads

    def _batched(self, iterable: Iterable, n: int = 1000):
        """Batch data into tuples of length n. The last batch may be shorter."""
        if n < 1:
            raise ValueError('n must be at least one')
        it = iter(iterable)
        while (batch := tuple(islice(it, n))):
            yield batch 

    def _chunked_tokens(self, text: str, chunk_length: int = 1000):
        """Chunk a text into chunks of tokens of length chunk_length."""
        encoding = tiktoken.get_encoding(self.encoding_name)
        tokens = encoding.encode(text)
        chunks_iterator = self._batched(tokens, chunk_length)
        yield from chunks_iterator

    def get_tokenizer(self, tokenizer_name: str="cl100k_base") -> tiktoken.Encoding:
        return tiktoken.get_encoding(tokenizer_name)

    def _get_embedding(self,text: str) -> List[float]:
        """Get the embedding for a given text."""
        # 1. Initialize the client
        # client = openai.OpenAI(api_key=os.environ['OPENAI_API_KEY'])
        client = openai.OpenAI(
            api_key="sk-1234",         # Match the master_key in config_vertex.yaml
            base_url="http://0.0.0.0:4000" # Your LiteLLM Proxy URL
        )
        # 2. Call the embeddings endpoint
        response = client.embeddings.create(
            model=self.model_name, 
            input=text,
            encoding_format="float"
        )

        # 3. Access the data using dot notation (not brackets)
        vector = response.data[0].embedding
        return vector
    def get_embedding(self,text: str) -> List[float]:
        """Get the embedding for a given text. If the text is too long, it will be chunked into smaller pieces."""
        text = text.replace("\n", " ")
        chunk_embeddings = []
        chunks = []
        chunk_lens = []
        encoding = tiktoken.get_encoding(self.encoding_name)
        for chunk in self._chunked_tokens(text, chunk_length=self.context_length):
            decoded_chunk = encoding.decode(chunk)
            if not decoded_chunk.strip():
                continue
            chunks.append(chunk)
            chunk_embeddings.append(self._get_embedding(decoded_chunk))
            chunk_lens.append(len(chunk))

        # if self.average:
        chunk_embeddings = np.average(chunk_embeddings, axis=0, weights=chunk_lens)
        chunk_embeddings = chunk_embeddings / np.linalg.norm(chunk_embeddings)  # normalizes length to 1
        chunk_embeddings = chunk_embeddings.tolist()
        return chunks, chunk_embeddings

if __name__ == "__main__":
    print("test qdrant database")
    url=os.environ['QDRANT_SERVER_URL']
    api_key=os.environ['QDRANT_API_KEY']
    
    model_name = 'vertex-embedding' # 'text-embedding-ada-002'
    
    qdb = QdrantDB(model_name, url=url, api_key=api_key)
    # Create database collection
    COLLECTION_NAME = "almanac"
    # 1. Delete the collection (handles error if it doesn't exist)
    qdb.delete_collection(collection_name=COLLECTION_NAME)

    # 2. Create the collection from scratch
    qdb.create_collection(collection_name=COLLECTION_NAME)

    texts = ["Hello world", "How are you?"]
    metadatas = ["https://www.google.com", "https://www.bing.com"]
    qdb.add_texts(texts, metadatas)
    results = qdb.search(query="Hello world", top_k=1)
    print(results)

    print("test chroma database")    
    directory_cdb = 'data/chromadb'
    llm = ChatOpenAI(
        model="gpt-4o",                # The model identifier set in your LiteLLM config
        api_key="sk-1234",                    # Your LiteLLM virtual key
        base_url="http://0.0.0.0:4000",       # The LiteLLM Proxy URL
        temperature=0
    )
    cdb = ChromaDB()

    cdb.create_db(directory_cdb)
    cdb.add_texts(texts, metadatas)
    result = cdb.search(model=llm, text="Hello world")
    print(result)
    


  