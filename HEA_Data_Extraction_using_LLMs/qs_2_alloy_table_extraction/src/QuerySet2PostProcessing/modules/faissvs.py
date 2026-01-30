import pandas as pd
import numpy as np
import faiss
from openai import OpenAI
import pickle
import uuid
import os
from dotenv import load_dotenv

class FAISSVectorStore:
    """FAISS-based vector store for RAG with open-source LLMs"""
    
    def __init__(self, model_name='text-embedding-3-small'):
        """
        Initialize the vector store
        
        Parameters:
        model_name: str - Name of the sentence-transformers model
        """
        load_dotenv()
        self.client = OpenAI(organization='org-aYBgUPhs8cYqZUEa6Fk86Z0W')
        self.model_name = model_name

        dummy_embedding = self.client.embeddings.create(input=["dummy text"], model=self.model_name) #Dummy text is used to get the dimension of the embeddings
        dummy_embedding = np.array(dummy_embedding.data[0].embedding)
        self.dimension = len(dummy_embedding)
        
        # Initialize FAISS index (using L2 distance)
        self.index = faiss.IndexFlatL2(self.dimension)
        
        # Storage for documents and metadata
        self.documents = []
        self.metadata = []
        self.ids = []
        self.extra_columns = []

        print(f"Initialized FAISSVectorStore with {model_name}, dimension={self.dimension}")
        
    def format_document_for_vectorization(self, row: pd.Series) -> str:
        """Format the document text for vectorization (only searchable fields)"""

        alloys = row['Alloy'].split(" | ")
        alloys = "\n".join(alloys)
    
        title = row['Title']
        abstract = row['Simplified Abstract'] if 'Simplified Abstract' in row.keys() else row['Abstract']
        extracted_section = row['Simplified Paragraph'] if 'Simplified Paragraph' in row.keys() else row['Paragraph']
        text = f"ENTRIES TO MAP\n{alloys}\n\nTITLE\n{title}\n\nABSTRACT\n{abstract}\n\nEXTRACTED TEXT SECTION\n{extracted_section}\n\n"

        return text
    
    
    def add_dataframe(self, df: pd.DataFrame, 
                      extra_columns: list[str] | None = None) -> list[str]:
        """
        Add dataframe entries to the vector store
        
        Parameters:
        df: DataFrame with columns ['Alloy', 'Abstract', 'Introduction'] 
            and optional additional columns
        extra_columns: List of column names to store but not vectorize
        
        Returns:
        List of document IDs
        """
        new_documents = []
        new_metadata = []
        new_ids = []
        new_extra_columns = []
        
        for idx, row in df.iterrows():
            # Format the document for vectorization (only searchable fields)
            doc_text = self.format_document_for_vectorization(row)
            new_documents.append(doc_text)
            
            # Create metadata
            new_metadata.append({
                'alloy': str(row['Alloy']),
                'source_index': idx
            })
            
            # Store extra columns if specified
            extra_data = {}
            if extra_columns:
                for col in extra_columns:
                    if col in df.columns:
                        extra_data[col] = row[col]
            new_extra_columns.append(extra_data)
            
            # Generate unique ID
            doc_id = str(uuid.uuid4())
            new_ids.append(doc_id)
        
        # Generate embeddings
        print("Generating embeddings...")
        embeddings = self.client.embeddings.create(input=new_documents, model=self.model_name)
        embeddings = np.array([e.embedding for e in embeddings.data]).astype('float32')   
        # Normalize embeddings for better retrieval
        faiss.normalize_L2(embeddings)
        
        # Add to FAISS index
        self.index.add(embeddings)
        
        # Store documents and metadata
        self.documents.extend(new_documents)
        self.metadata.extend(new_metadata)
        self.ids.extend(new_ids)
        self.extra_columns.extend(new_extra_columns)
        
        print(f"Added {len(new_documents)} documents to vector store")
        print(f"Total documents in store: {len(self.documents)}")
        
        return new_ids
    
    def query(self, query_text: str, k: int = 2) -> dict:
        """
        Query the vector store
        
        Parameters:
        query_text: str - The query text
        k: int - Number of results to return
        
        Returns:
        dict with documents, metadata, distances, ids, and extra_columns
        """
        if self.index.ntotal == 0:
            raise ValueError("Vector store is empty. Add documents first.")
        
        # Generate query embedding
        query_embedding = self.client.embeddings.create(input=[query_text], model=self.model_name)
        query_embedding = np.array([e.embedding for e in query_embedding.data]).astype('float32')
        
        # Normalize query embedding
        faiss.normalize_L2(query_embedding)
        
        # Search
        k = min(k, self.index.ntotal)  # Don't request more than available
        distances, indices = self.index.search(query_embedding, k)
        
        # Prepare results
        results = {
            'documents': [self.documents[idx] for idx in indices[0]],
            'metadata': [self.metadata[idx] for idx in indices[0]],
            'distances': distances[0].tolist(),
            'ids': [self.ids[idx] for idx in indices[0]],
            'extra_columns': [self.extra_columns[idx] for idx in indices[0]]
        }
        
        return results
    
    def get_context_for_llm(self, query_text: pd.Series, k: int = 2, 
                           include_extra: bool = True) -> str:
        """
        Get formatted context for LLM inference including extra columns
        
        Parameters:
        query_text: str - The query text
        k: int - Number of documents to retrieve
        include_extra: bool - Whether to include extra columns in context
        
        Returns:
        str - Formatted context string ready for LLM
        """
        query_text = self.format_document_for_vectorization(query_text)
        results = self.query(query_text, k)
        
        context_parts = ["FEW SHOT EXAMPLES"]
        for i, (doc, metadata, extra) in enumerate(
            zip(results['documents'], results['metadata'], results['extra_columns']), 1
        ):
            # Start with the base document
            doc_text = doc
            
            # Add extra columns if requested and available
            if include_extra and extra:

                doc_text += f"REASONING\n{extra['Reasoning']}\n\n"
                doc_text += f"OUTPUT\n{extra['Ideal Output']}"
            
            context_parts.append(f"=== Example {i} ===\n{doc_text}\n")
        
        context = "\n".join(context_parts)
        return context
    
    def save(self, save_dir: str = '/home/user/alloy_comp_cleaning_rbcdsai/output/vector_store'):
        """
        Save the FAISS index and all metadata to disk
        
        Parameters:
        save_dir: str - Directory to save the vector store
        """
        # Create directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)
        
        index_path = os.path.join(save_dir, 'faiss_index.bin')
        metadata_path = os.path.join(save_dir, 'metadata.pkl')
        
        # Save FAISS index
        faiss.write_index(self.index, index_path)
        
        # Save all metadata, documents, IDs, and extra columns
        with open(metadata_path, 'wb') as f:
            pickle.dump({
                'documents': self.documents,
                'metadata': self.metadata,
                'ids': self.ids,
                'extra_columns': self.extra_columns,
                'embedding_model_name': self.model_name
            }, f)
        
        print(f"✓ Saved vector store to {save_dir}/")
        print(f"  - Index: {index_path}")
        print(f"  - Metadata: {metadata_path}")
        print(f"  - Total documents: {len(self.documents)}")
    
    def load(self, save_dir: str = '/home/user/alloy_comp_cleaning_rbcdsai/output/vector_store'):
        """
        Load the FAISS index and metadata from disk
        
        Parameters:
        save_dir: str - Directory containing the saved vector store
        """
        index_path = os.path.join(save_dir, 'faiss_index.bin')
        metadata_path = os.path.join(save_dir, 'metadata.pkl')
        
        if not os.path.exists(index_path) or not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Vector store not found in {save_dir}/")
        
        # Load FAISS index
        self.index = faiss.read_index(index_path)
        
        # Load metadata, documents, IDs, and extra columns
        with open(metadata_path, 'rb') as f:
            data = pickle.load(f)
            self.documents = data['documents']
            self.metadata = data['metadata']
            self.ids = data['ids']
            self.extra_columns = data.get('extra_columns', [])
        
        print(f"✓ Loaded vector store from {save_dir}/")
        print(f"  - Total documents: {len(self.documents)}")
    
    @classmethod
    def load_existing(cls, save_dir: str = '/home/user/alloy_comp_cleaning_rbcdsai/output/vector_store', 
                      model_name: str = 'text-embedding-3-small'):
        """
        Class method to load an existing vector store
        
        Parameters:
        save_dir: str - Directory containing the saved vector store
        model_name: str - Name of the embedding model to use
        
        Returns:
        FAISSVectorStore instance
        """
        store = cls(model_name=model_name)
        store.load(save_dir)
        return store
