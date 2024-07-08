from query.query_rag import query


class RelevanceEvaluate:
    def keyword_match_score(self, prompt, passage):
        """
        This function calculates the relevance score based on keyword matching.

        Args:
            prompt: The user query or prompt for the RAG system.
            passage: A retrieved passage from the knowledge base.

        Returns:
        A score representing the relevance of the passage to the prompt.
        """
        keywords = set(prompt.lower().split())
        passage_words = set(passage.lower().split())
        matches = len(keywords.intersection(passage_words))
        return matches / len(keywords)


if __name__ == '__main__':
    # Sample prompt and retrieved passage
    query_message = "Machine learning engineer"

    rag_response = query()
    retrieved_passage, prompt = rag_response.query_llm_with_retreival_results(query=query_message)
    # Calculate relevance score using keyword matching
    evaluation_object = RelevanceEvaluate()
    relevance_score = evaluation_object.keyword_match_score(prompt=prompt, passage=retrieved_passage)
    print(f"Relevance score (keyword match): {relevance_score}")
