import faiss
import numpy as np
from retrieval.indexing import Indexing
import logging
from generation.llm_loading import LLM

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)


class query:
    def __init__(self):
        self.indexer = Indexing()
        self.indexer.vectorize_data()
        self.vectorizer = self.indexer.count_vectorizer
        self.index = self.indexer.store_docs_to_vector_db_and_get_index()
        self.retrieved_docs = []

    def retrieve_profile_docs(self, query):
        """This function call indexer with query and return docs related fields concatenated and ready for llm
        and return context of appended retrieved docs

        Args:
            query (str): query string

        Returns:
            context: list of retrieved docs
         """
        query_vector = self.vectorizer.transform([query])
        query_vector = query_vector.toarray().astype(np.float32)
        _, neighbors = self.index.search(query_vector, k=2)

        for i in range(len(neighbors[0])):
            retrieved_doc = self.indexer.processed_data[neighbors[0][i]]
            concatenated_fields_in_doc = f"job title : {retrieved_doc['job_title']} \n" \
                                         f"description : {retrieved_doc['description']} \n" \
                                         f"requirements : {retrieved_doc['requirements']} \n" \
                                         f"career level : {retrieved_doc['career_level']} \n"
            self.retrieved_docs.append(concatenated_fields_in_doc)

        logging.info(f"Query : {query}, concatenated fields : {self.retrieved_docs}")

        context = "\nExtracted documents:\n"
        context += "".join([f"Document {str(i)}:::\n" + doc for i, doc in enumerate(self.retrieved_docs)])

        return context

    def create_rag_chat_template(self, tokenizer):
        """This function creates the template that will be passed to the llm for generating results

        Args:
            tokenizer (transformers.pipeline): transformers pipeline

        Returns:
            RAG_PROMPT_TEMPLATE: a chat template of system and user
        """
        prompt_in_chat_format = [
            {
                "role": "system",
                "content": """Using the information contained in the context,
        give me skills and requirements i should be good at to be well prepared for this job title.""",
            },
            {
                "role": "user",
                "content": """Context:
        {context}
        ---
        Now here is the job title you need to provide me some advices to be good at this job role.

        Job title: {title}""",
            },
        ]
        RAG_PROMPT_TEMPLATE = tokenizer.apply_chat_template(
            prompt_in_chat_format, tokenize=False, add_generation_prompt=True
        )
        print(RAG_PROMPT_TEMPLATE)
        return RAG_PROMPT_TEMPLATE

    def query_llm_with_retrieval_results(self, query):
        """This function fetch user query and results then pass to llm to generate llm response

        Args:
            query (str): query string

        Returns:
            generation_output: LLm generation output
            final_prompt: the prompt loaded with query and retrieved docs
        """
        llm_object = LLM()
        context = self.retrieve_profile_docs(query=query)
        RAG_PROMPT_TEMPLATE = self.create_rag_chat_template(tokenizer=llm_object.tokenizer)
        final_prompt = RAG_PROMPT_TEMPLATE.format(title=query, context=context)

        print(final_prompt)

        llm_reader = llm_object.instantiate_LLM_Reader()
        generation_output = llm_reader(final_prompt)[0]["generated_text"]
        generation_output = generation_output.replace(final_prompt, "")
        return generation_output, final_prompt


if __name__ == "__main__":
    query_object = query()
    query_object.query_llm_with_retrieval_results(query="Machine learning Engineer")
