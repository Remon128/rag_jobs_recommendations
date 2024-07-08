from pathlib import Path
import pandas as pd
from pandas import DataFrame

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
        jobs_data = pd.read_csv(self.data_dir)
        return jobs_data

