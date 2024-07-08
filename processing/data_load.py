from pathlib import Path
import pandas as pd
from pandas import DataFrame
import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

DATA_DIR = Path(__file__).absolute().parent.parent.joinpath('data')


class DataLoader:
    """This class for loading and processing data to have a filtered documents"""
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.jobs_data = self.load_data()

    def load_data(self):
        """This function load data and return loaded data

        Returns:
            DataFrame: jobs data
        """
        logger.info(f"Loading data from files in {self.data_dir}")
        jobs_data = pd.read_csv(self.data_dir)
        logger.info(f"Loaded {len(jobs_data)} job entries")
        return jobs_data

