from processing.data_load import DataLoader
import spacy
from bs4 import BeautifulSoup
import numpy as np
import multiprocessing
import itertools
import logging
from time import time

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)


class DataProcessing:
    """This class for applying processing pipline on data"""

    def __init__(self, data_dir):
        self.data_loader = DataLoader(data_dir=data_dir)
        self.prepared_data = self.prepare_data(data=self.data_loader.jobs_data)
        self.nlp = spacy.load('en_core_web_md')
        self.result_queue = multiprocessing.Queue()
        self.n_processes = multiprocessing.cpu_count()

    def prepare_data(self, data):
        """This function remove null from df and clean df
        Args:
            data (Dataframe): df of raw data
        Returns:
            Dataframe: df of cleaned data
        """
        # replace Nan values with empty string
        data.fillna(value="", inplace=True)

        # transform data frame into list of dicts to process each attribute
        prepared_data = data.to_dict(orient='records')
        return prepared_data

    def standardize_data(self, profile):
        """This function is used to standardize the data and remove html tags
        Args:
            profile (Dataframe): df of raw data

        Returns:
            Dataframe: df of standardized data without html tags
        """
        # Parse the HTML
        description_soup = BeautifulSoup(profile["description"], "html.parser")
        requirements_soup = BeautifulSoup(profile["requirements"], "html.parser")

        # Extract all text from the HTML
        description_content = description_soup.get_text(separator="\n")
        requirements_content = requirements_soup.get_text(separator="\n")

        profile["description"] = description_content.strip()
        profile["requirements"] = requirements_content.strip()

        return profile

    def processing_util(self, attribute):
        """This util function used to stop words removal

        Args:
            attribute (str): a string of attribute

        Returns:
            processed_text (str): processed text
        """

        doc = self.nlp(attribute)
        # Convert to lowercase and remove stop words
        processed_text = [token.text.lower() for token in doc if not token.is_stop]
        # Join the tokens back into a string
        processed_text = " ".join(processed_text)

        return processed_text

    def remove_stopwords(self, profile):
        """This function is used to remove stopwords"""
        profile["job_title"] = self.processing_util(attribute=profile["job_title"])

        return profile

    def _process_profile(self, profile):
        """This function applying processing pipline on data
        - prepare data like fillna nan values
        - stop word removal
        - html headers removal
        """
        t_0 = time()
        self.standardize_data(profile=profile)
        self.remove_stopwords(profile=profile)

        logger.info(f"Processed one profile in {round(time() - t_0, 2)}s")
        profile["doc"] = profile['job_title']
        return profile

    def _process_profiles_chunk(self, chunk):
        profiles_chunk, idx = chunk
        # process items in the chunk
        processed_chunk = [self._process_profile(profile) for profile in profiles_chunk]

        # push to the shared queue
        self.result_queue.put((processed_chunk, idx))

    def process_profiles(self):
        logger.info("------------------------")
        logger.info(f"Started Processing")
        t_0 = time()

        profile_chunks = [
            (chunk, idx)
            for idx, chunk in enumerate(np.array_split(self.prepared_data, self.n_processes))
        ]

        # Spawn a process for every chunk.
        processes = [
            multiprocessing.Process(target=self._process_profiles_chunk, args=(chunk,))
            for chunk in profile_chunks
        ]

        # empty list for processes
        result = [[]] * len(processes)

        # Start processes
        for process in processes:
            process.start()

        # Fetch the result back in order.
        for _ in processes:
            processed_profile, idx = self.result_queue.get()
            result[idx] = processed_profile

        result = list(itertools.chain.from_iterable(result))

        # join processes
        for process in processes:
            process.join()

        logger.info("------------------------")
        logger.info(f"Processed profiles in {round(time() - t_0, 2)}s")

        return result
