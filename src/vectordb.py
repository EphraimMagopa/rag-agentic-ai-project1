import os
import chromadb
import uuid
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer


class VectorDB:
    """
    A simple vector database wrapper using ChromaDB with HuggingFace embeddings.
    """

    def __init__(self, collection_name: str = None, embedding_model: str = None):
        """
        Initialize the vector database.

        Args:
            collection_name: Name of the ChromaDB collection
            embedding_model: HuggingFace model name for embeddings
        """
        self.collection_name = collection_name or os.getenv(
            "CHROMA_COLLECTION_NAME", "rag_documents"
        )
        self.embedding_model_name = embedding_model or os.getenv(
            "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
        )

        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(path="./chroma_db")

        # Load embedding model
        print(f"Loading embedding model: {self.embedding_model_name}")
        self.embedding_model = SentenceTransformer(self.embedding_model_name)

        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"description": "RAG document collection"},
        )

        print(f"Vector database initialized with collection: {self.collection_name}")

    def chunk_text(self, text: str, chunk_size: int = 500) -> List[str]:
        """
        Split text into smaller chunks for better retrieval using RecursiveCharacterTextSplitter.

        Args:
            text: Input text to chunk
            chunk_size: Approximate number of characters per chunk

        Returns:
            List of text chunks
        """
        # Done: Implement text chunking logic and used option 2
        # You have several options for chunking text - choose one or experiment with multiple:
        #
        # OPTION 1: Simple word-based splitting
        #   - Split text by spaces and group words into chunks of ~chunk_size characters
        #   - Keep track of current chunk length and start new chunks when needed
        #
        # OPTION 2: Use LangChain's RecursiveCharacterTextSplitter
        #   - from langchain_text_splitters import RecursiveCharacterTextSplitter
        #   - Automatically handles sentence boundaries and preserves context better
        #
        # OPTION 3: Semantic splitting (advanced)
        #   - Split by sentences using nltk or spacy
        #   - Group semantically related sentences together
        #   - Consider paragraph boundaries and document structure
        #
        # Feel free to try different approaches and see what works best!

        chunks = []
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size = chunk_size,
            chunk_overlap = 50, # overlap to maintain context across chunks
            length_function = len
        )
        chunks = text_splitter.split_text(text)

        return chunks

    def add_documents(self, documents: List[Dict[str, Any]]) -> None:
        """
        Process documents and Add them to the vector database.

        Args:
            documents: List of documents with 'content' and optional 'metadata'
        """
        # Done: Implement document ingestion logic
        # HINT: Loop through each document in the documents list
        # HINT: Extract 'content' and 'metadata' from each document dict
        # HINT: Use self.chunk_text() to split each document into chunks
        # HINT: Create unique IDs for each chunk (e.g., "doc_0_chunk_0")
        # HINT: Use self.embedding_model.encode() to create embeddings for all chunks
        # HINT: Store the embeddings, documents, metadata, and IDs in your vector database
        # HINT: Print progress messages to inform the user

        print(f"Processing {len(documents)} documents...")
        
        all_chunks = []
        all_metadatas = []
        all_ids = []
        
        for doc in documents:
            content = doc.get("content", "")
            metadata = doc.get("metadata", {})
            
            # Split content into chunks using the chunk_text method
            chunks = self.chunk_text(content) # This returns a list of text chunks
            
            for i, chunk in enumerate(chunks):
                # Generate a unique ID for each chunk
                chunk_id = str(uuid.uuid4()) # Unique ID per chunk
                all_ids.append(chunk_id) # Append 1 per chunk
                
                # Store chunk metadata, including parend document information and chunk index
                chunk_metadata = {
                    **metadata,
                    "chunk_index": i,
                    "chunk_text": chunk[:50] # Preview first 50 characters just for convenience
                }
                all_metadatas.append(chunk_metadata) # Append 1 per chunk
                
                all_chunks.append(chunk) # Append 1 per chunk
            
        # Create embeddings for all chunks
        embeddings = self.embedding_model.encode(all_chunks, convert_to_numpy=True)
        
        print(f"Adding to DB: {len(all_ids)} ids, {len(all_metadatas)} metadatas, {len(all_chunks)} chunks")
        
        # Add chunks and eembeddings to the ChromaDB collection
        self.collection.add(
            documents=all_chunks,
            metadatas=all_metadatas,
            ids=all_ids,
            embeddings=embeddings
        )
        
        print("Documents added to vector database")

    def search(self, query: str, n_results: int = 5) -> Dict[str, Any]:
        """
        Search for similar documents in the vector database.

        Args:
            query: Search query
            n_results: Number of results to return

        Returns:
            Dictionary containing search results with keys: 'documents', 'metadatas', 'distances', 'ids'
        """
        # Done: Implement similarity search logic
        # HINT: Use self.embedding_model.encode([query]) to create query embedding
        # HINT: Convert the embedding to appropriate format for your vector database
        # HINT: Use your vector database's search/query method with the query embedding and n_results
        # HINT: Return a dictionary with keys: 'documents', 'metadatas', 'distances', 'ids'
        # HINT: Handle the case where results might be empty

        # Create embedding for the query
        query_embedding = self.embedding_model.encode([query], convert_to_numpy=True)
        
        # Perform vector similarity search in ChromaDB
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=n_results,
            include=["documents", "metadatas", "distances"]
        )
        
        # Extract fields from results
        documents = results["documents"][0] if "documents" in results else []
        metadatas = results["metadatas"][0] if "metadatas" in results else []
        distances = results["distances"][0] if "distances" in results else []
        ids = results["ids"][0] if "ids" in results else []
        
        return {
            "documents": documents,
            "metadatas": metadatas,
            "distances": distances,
            "ids": ids,
        }
