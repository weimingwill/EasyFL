import logging
import os

from easyfl.datasets.shakespeare.utils.gen_all_data import generated_all_data
from easyfl.datasets.shakespeare.utils.preprocess_shakespeare import shakespeare_preprocess
from easyfl.datasets.utils.base_dataset import BaseDataset
from easyfl.datasets.utils.download import download_url, extract_archive, download_from_google_drive

logger = logging.getLogger(__name__)


class Shakespeare(BaseDataset):
    """Shakespeare dataset implementation. It gets Shakespeare dataset according to configurations.

    Attributes:
        base_folder (str): The base folder path of the datasets folder.
        raw_data_url (str): The url to get the `by_class` split shakespeare.
        write_url (str): The url to get the `by_write` split shakespeare.
    """

    def __init__(self,
                 root,
                 fraction,
                 split_type,
                 user,
                 iid_user_fraction=0.1,
                 train_test_split=0.9,
                 minsample=10,
                 num_class=80,
                 num_of_client=100,
                 class_per_client=2,
                 setting_folder=None,
                 seed=-1,
                 **kwargs):
        super(Shakespeare, self).__init__(root,
                                          "shakespeare",
                                          fraction,
                                          split_type,
                                          user,
                                          iid_user_fraction,
                                          train_test_split,
                                          minsample,
                                          num_class,
                                          num_of_client,
                                          class_per_client,
                                          setting_folder,
                                          seed)
        self.raw_data_url = "http://www.gutenberg.org/files/100/old/1994-01-100.zip"
        self.packaged_data_files = {
            "shakespeare_niid_100_10_1_0.05_0.1_sample_0.9.zip": "https://dl.dropboxusercontent.com/s/5qr9ozziy3yfzss/shakespeare_niid_100_10_1_0.05_0.1_sample_0.9.zip",
            "shakespeare_iid_100_10_1_0.05_0.1_sample_0.9.zip": "https://dl.dropboxusercontent.com/s/4p7osgjd2pecsi3/shakespeare_iid_100_10_1_0.05_0.1_sample_0.9.zip"
        }
        # Google drive ids.
        # self.packaged_data_files = {
        #     "shakespeare_niid_100_10_1_0.05_0.1_sample_0.9.zip": "1zvmNiUNu7r0h4t0jBhOJ204qyc61NvfJ",
        #     "shakespeare_iid_100_10_1_0.05_0.1_sample_0.9.zip": "1Lb8n1zDtrj2DX_QkjNnL6DH5IrnYFdsR"
        # }

    def download_packaged_dataset_and_extract(self, filename):
        file_path = download_url(self.packaged_data_files[filename], self.base_folder)
        extract_archive(file_path, remove_finished=True)

    def download_raw_file_and_extract(self):
        raw_data_folder = os.path.join(self.base_folder, "raw_data")
        if not os.path.exists(raw_data_folder):
            os.makedirs(raw_data_folder)
        elif os.listdir(raw_data_folder):
            logger.info("raw file exists")
            return
        raw_data_path = download_url(self.raw_data_url, raw_data_folder)
        extract_archive(raw_data_path, remove_finished=True)
        os.rename(os.path.join(raw_data_folder, "100.txt"), os.path.join(raw_data_folder, "raw_data.txt"))
        logger.info("raw file is downloaded")

    def preprocess(self):
        filename = os.path.join(self.base_folder, "raw_data", "raw_data.txt")
        raw_data_folder = os.path.join(self.base_folder, "raw_data")
        if not os.path.exists(raw_data_folder):
            os.makedirs(raw_data_folder)
        shakespeare_preprocess(filename, raw_data_folder)

    def convert_data_to_json(self):
        all_data_folder = os.path.join(self.base_folder, "all_data")
        if not os.path.exists(all_data_folder):
            os.makedirs(all_data_folder)
        if not os.listdir(all_data_folder):
            logger.info("converting data to .json format")
            generated_all_data(self.base_folder)
            logger.info("finished converting data to .json format")
