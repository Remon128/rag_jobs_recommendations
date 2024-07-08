# rag_jobs_recommendations
This repository is building a RAG system to recommend advice and tips to user based on their queries for the related job.

## About the system:
- This Rag system help users to query job titles and get a recommendation advice on how to prepare for this role.
- The system is divided into some modules that integrate together to perform the rag functionality.
- `data` module is where we store the data we need to use for retrieval module.
- `processing` module has the data loading from files and processing to be prepared for indexing.
- `generation` module have all implementation related to llm loading and creating transformer pipline.
- `retrieval` module has the logic related to indexing the processed items and creating embeddings for data.
- `query` module has the integration logic for the rag system as we propagate the query from this module to retrieval module, and to create the chat prompt we send to llm.
- `evaluation` module is responsible for evaluating the rag system performance to evaluate the relevant generated output to the query template sent to llm , currently we apply one method for evaluation which is relevance response to the query prompt.


## Current results and limitations:
- The system scores around 35% for relevant tokens retrieved from the user prompt compared to llm generated result.
- We use a tiny llama version model in current implementation due to gpu and ram limitation on local machine, which is affecting the score dramatically.
- We used around 50% of the docs in the data due to ram limitation for dense vectors.

## Future improvements:
- Utilize larger llm and utilize gpu resources to improve text generation part.
- Apply more evaluation techniques using a query data set.
- Dockerize environment for llm and search engine.
- Apply RLHF technique to enhance system performance.


## How to start the system:
- The system can work on cpu , but inference will take some time around 20 mins.
- To start the system you can run in terminal `python query/query_rag.py --job "job title"` example
- `python query/query_rag.py --job "machine learning engineer"`

## Evaluate the system
- You can run this command `python evaluation/relevance_evaluate.py --job "Machine learning engineer""`
- NOTE:
- The system is working on 50% of jobs data due to RAM limitation , so expected some jobs to not be found during search.