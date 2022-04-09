import logging
import os

from easyfl.datasets.femnist.preprocess.data_to_json import data_to_json
from easyfl.datasets.femnist.preprocess.get_file_dirs import get_file_dir
from easyfl.datasets.femnist.preprocess.get_hashes import get_hash
from easyfl.datasets.femnist.preprocess.group_by_writer import group_by_writer
from easyfl.datasets.femnist.preprocess.match_hashes import match_hash
from easyfl.datasets.utils.base_dataset import BaseDataset
from easyfl.datasets.utils.download import download_url, extract_archive, download_from_google_drive

logger = logging.getLogger(__name__)


class Femnist(BaseDataset):
    """FEMNIST dataset implementation. It gets FEMNIST dataset according to configurations.
     It stores the processed datasets locally.

    Attributes:
        base_folder (str): The base folder path of the datasets folder.
        class_url (str): The url to get the by_class split FEMNIST.
        write_url (str): The url to get the by_write split FEMNIST.
    """

    def __init__(self,
                 root,
                 fraction,
                 split_type,
                 user,
                 iid_user_fraction=0.1,
                 train_test_split=0.9,
                 minsample=10,
                 num_class=62,
                 num_of_client=100,
                 class_per_client=2,
                 setting_folder=None,
                 seed=-1,
                 **kwargs):
        super(Femnist, self).__init__(root,
                                      "femnist",
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
        self.class_url = "https://s3.amazonaws.com/nist-srd/SD19/by_class.zip"
        self.write_url = "https://s3.amazonaws.com/nist-srd/SD19/by_write.zip"
        self.packaged_data_files = {
            "femnist_niid_100_10_1_0.05_0.1_sample_0.9.zip": "https://dl.dropboxusercontent.com/s/oyhegd3c0pxa0tl/femnist_niid_100_10_1_0.05_0.1_sample_0.9.zip",
            "femnist_iid_100_10_1_0.05_0.1_sample_0.9.zip": "https://dl.dropboxusercontent.com/s/jcg0xrz5qrri4tv/femnist_iid_100_10_1_0.05_0.1_sample_0.9.zip"
        }
        # Google Drive ids
        # self.packaged_data_files = {
        #     "femnist_niid_100_10_1_0.05_0.1_sample_0.9.zip": "11vAxASl-af41iHpFqW2jixs1jOUZDXMS",
        #     "femnist_iid_100_10_1_0.05_0.1_sample_0.9.zip": "1U9Sn2ACbidwhhihdJdZPfK2YddPMr33k"
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
        class_path = download_url(self.class_url, raw_data_folder)
        write_path = download_url(self.write_url, raw_data_folder)
        extract_archive(class_path, remove_finished=True)
        extract_archive(write_path, remove_finished=True)
        logger.info("raw file is downloaded")

    def preprocess(self):
        intermediate_folder = os.path.join(self.base_folder, "intermediate")
        if not os.path.exists(intermediate_folder):
            os.makedirs(intermediate_folder)
        if not os.path.exists(intermediate_folder + "/class_file_dirs.pkl"):
            logger.info("extracting file directories of images")
            get_file_dir(self.base_folder)
            logger.info("finished extracting file directories of images")
        if not os.path.exists(intermediate_folder + "/class_file_hashes.pkl"):
            logger.info("calculating image hashes")
            get_hash(self.base_folder)
            logger.info("finished calculating image hashes")
        if not os.path.exists(intermediate_folder + "/write_with_class.pkl"):
            logger.info("assigning class labels to write images")
            match_hash(self.base_folder)
            logger.info("finished assigning class labels to write images")
        if not os.path.exists(intermediate_folder + "/images_by_writer.pkl"):
            logger.info("grouping images by writer")
            group_by_writer(self.base_folder)
            logger.info("finished grouping images by writer")

    def convert_data_to_json(self):
        all_data_folder = os.path.join(self.base_folder, "all_data")
        if not os.path.exists(all_data_folder):
            os.makedirs(all_data_folder)
        if not os.listdir(all_data_folder):
            logger.info("converting data to .json format")
            data_to_json(self.base_folder)
            logger.info("finished converting data to .json format")
