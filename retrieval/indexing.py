from processing.data_processing import DataProcessing
from pathlib import Path
from sklearn.feature_extraction.text import CountVectorizer
from langchain.vectorstores import FAISS
from langchain_core.documents import Document
import numpy as np
import faiss
import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)
DATA_DIR = Path(__file__).absolute().parent.parent.joinpath('data')


class Indexing:
    def __init__(self):
        self.processing = DataProcessing(data_dir=DATA_DIR / "en" / "sampled_jobs.csv")
        self.processed_data = self.processing.process_profiles()
        # Taking only 10k of data to handle dense array issue as it takes too much memory
        self.processed_data = self.processed_data[:20000]
        self.count_vectorizer = None
        self.vectorized_documents = None
        self.docs_texts = None
        self.docs_documents = None
        self.extract_docs()
        print(len(self.processed_data), "processed_profiles")

    def extract_docs(self):
        """This function extract docs from processed data"""
        self.docs_texts = [profile["doc"] for profile in self.processed_data]

    def vectorize_data(self):
        """This function change documents to embeddings and return each document with it's corresponding vector."""
        self.count_vectorizer = CountVectorizer(analyzer="word")
        self.vectorized_documents = self.count_vectorizer.fit_transform(self.docs_texts)
        self.vectorized_documents = self.vectorized_documents.toarray().astype(np.float32)

    def store_docs_to_vector_db_and_get_index(self):
        # Create a FAISS index
        index = faiss.IndexFlatL2(self.vectorized_documents.shape[1])  # L2 distance
        index.add(self.vectorized_documents)
        logger.info("Data is indexed and ready for querying")
        return index


if __name__ == "__main__":
    index_en = Indexing()
    index_en.vectorize_data()
    index_en.store_docs_to_vector_db_and_get_index()
