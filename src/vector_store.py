"""
Semantic Search and Vector Store for IRAM

This module implements semantic search capabilities using sentence-transformers
embeddings and a vector store for efficient similarity searching over collected data.
"""

import os
import json
import asyncio
import faiss
import numpy as np
from typing import Dict, List, Any, Optional
from pathlib import Path

from .utils import get_logger
from .analysis_module import ContentAnalyzer # For embeddings

logger = get_logger(__name__)


class VectorStore:
    """Manages a FAISS-based vector store for semantic search."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the vector store."""
        self.config = config or {}
        self.index_file = self.config.get("vector_index_file", "data/vector_index.faiss")
        self.metadata_file = self.config.get("vector_metadata_file", "data/vector_metadata.json")
        
        # Initialize components
        self.analyzer = ContentAnalyzer(config)
        self.embedding_dim = self._get_embedding_dim()
        self.index = None
        self.metadata: List[Dict[str, Any]] = []
        
        # Load existing index
        self._load_index()
        
        logger.info("Vector store initialized")
    
    def _get_embedding_dim(self) -> int:
        """Get embedding dimension from the model."""
        if self.analyzer and self.analyzer.sentence_model:
            return self.analyzer.sentence_model.get_sentence_embedding_dimension()
        return 384 # Default for all-MiniLM-L6-v2
    
    def _load_index(self):
        """Load FAISS index and metadata from files."""
        try:
            index_path = Path(self.index_file)
            metadata_path = Path(self.metadata_file)
            
            if index_path.exists() and metadata_path.exists():
                self.index = faiss.read_index(str(index_path))
                with open(metadata_path, 'r') as f:
                    self.metadata = json.load(f)
                logger.info(f"Loaded vector store with {self.index.ntotal} vectors")
            else:
                # Initialize new index
                if self.embedding_dim:
                    self.index = faiss.IndexFlatL2(self.embedding_dim)
                logger.info("Initialized new vector store")
        except Exception as e:
            logger.error(f"Failed to load vector store: {e}")
            if self.embedding_dim:
                self.index = faiss.IndexFlatL2(self.embedding_dim)

    def _save_index(self):
        """Save FAISS index and metadata to files."""
        try:
            os.makedirs(Path(self.index_file).parent, exist_ok=True)
            faiss.write_index(self.index, self.index_file)
            with open(self.metadata_file, 'w') as f:
                json.dump(self.metadata, f, indent=2)
            logger.info(f"Saved vector store with {self.index.ntotal} vectors")
        except Exception as e:
            logger.error(f"Failed to save vector store: {e}")

    async def add_documents(self, documents: List[Dict[str, Any]]):
        """Add documents to the vector store."""
        try:
            if not self.analyzer or not self.analyzer.sentence_model:
                logger.warning("Sentence model not available, cannot add documents")
                return

            texts_to_embed = []
            metadata_to_add = []

            for doc in documents:
                text = doc.get("text")
                if text:
                    texts_to_embed.append(text)
                    metadata_to_add.append(doc.get("metadata", {}))

            if not texts_to_embed:
                return

            # Generate embeddings
            embeddings = self.analyzer.sentence_model.encode(texts_to_embed, convert_to_tensor=False)
            
            # Add to index
            self.index.add(np.array(embeddings, dtype=np.float32))
            self.metadata.extend(metadata_to_add)
            
            # Save updates
            self._save_index()
            logger.info(f"Added {len(documents)} documents to vector store")

        except Exception as e:
            logger.error(f"Failed to add documents: {e}")

    async def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Perform semantic search."""
        try:
            if not self.index or self.index.ntotal == 0:
                return []

            # Embed query
            query_embedding = self.analyzer.sentence_model.encode([query], convert_to_tensor=False)
            
            # Search index
            distances, indices = self.index.search(np.array(query_embedding, dtype=np.float32), top_k)

            results = []
            for i, idx in enumerate(indices[0]):
                if idx < len(self.metadata):
                    results.append({
                        "metadata": self.metadata[idx],
                        "distance": distances[0][i],
                        "id": int(idx)
                    })
            
            return results

        except Exception as e:
            logger.error(f"Semantic search failed: {e}")
            return []

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector store."""
        return {
            "indexed_documents": self.index.ntotal if self.index else 0,
            "embedding_dimension": self.embedding_dim,
            "index_file": self.index_file,
            "metadata_file": self.metadata_file
        }

# Global vector store instance
_vector_store: Optional[VectorStore] = None

def get_vector_store() -> VectorStore:
    """Get global vector store instance."""
    global _vector_store
    if _vector_store is None:
        _vector_store = VectorStore()
    return _vector_store