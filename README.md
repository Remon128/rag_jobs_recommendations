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
- We use a `TinyLlama/TinyLlama_v1.1` version model in current implementation due to gpu and ram limitation on local machine, which is affecting the score dramatically.
- We used around 50% of the docs in the data due to ram limitation for dense vectors.

## Future improvements:
- Utilize larger llm and utilize gpu resources to improve text generation part.
- Apply more evaluation techniques using a query data set.
- Dockerize environment for llm and search engine.
- Apply RLHF technique to enhance system performance.
- Refine generation to keep the relevance responses only.


## How to start the system:
- Make sure you are connected to internet.
- Create a virtual ENV with `python -m venv "env_name"`.
- Source the env with command `source env_name/bin/activate`.
- Install dependencies with command `pip install -r requirements`.
- The system can work on cpu , but inference will take some time around 5 minutes be patient please.
- To start the system you can run in terminal `python query/query_rag.py --job "job title"`
- Example `python query/query_rag.py --job "machine learning engineer"`

## Evaluate the system
- You can run this command `python evaluation/relevance_evaluate.py --job "Machine learning engineer""`
- `NOTE`:
- The system is working on 50% of jobs data due to RAM limitation , so expected some jobs to not be found during search.

## Sample of Input query and generation output
`How to be a Machine learning engineer`

        Ensure using models pipeline which entails version control of models, experiments & metadata.
        A/B testing on various models.
        Optimize models for better performance, latency, memory and throughput.
        Monitor models performance, maintenance & support.
        
        Qualifications:
        B.Sc. of Computer Science or Computer Engineering or Electronics & Communications.
        M.Sc. in ML related fields is a plus!
        Strong SW Engineering background.
        Good Knowledge in OOP, Data Structures, Algorithms.
        Expertise in Deep Learning.
        Expertise in Data Analytics.
        Strong communication skills, including the ability to present ideas and share your knowledge with others.
        We are Siemens. Do you have any questions about the job? 

        We have lots of job opportunities for you!

        We are a global technology powerhouse. With some of the best-known brands in the world, Siemens is a leader in power and automation technologies. We are committed to equality, and we welcome applications that reflect the diversity of the communities we work in. All employment decisions at Siemens are based on qualifications, merit and business need. Bring your curiosity and creativity and help us shape tomorrow!We offer a comprehensive reward package which includes a competitive basic salary and a generous holiday allowance.
        We are an equal opportunities employer and do not discriminate unlawfully on any grounds. We are committed to providing access, equal opportunity for individuals with disabilities in employment, its services, programs, and activities. 

        References please do not send me resumes only job title.

        All correspondence to Siemens on this matter should be addressed:
        Human Resources Department, Siemens, 99 Bishopsgate, London, EC2M 4PJ, United Kingdom`
"