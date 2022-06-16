import hashlib
import io
import shutil
import zipfile
from abc import ABC, abstractmethod
from os import listdir, makedirs, path
from typing import Iterable

import requests

from conmo.conf import Directory


class Dataset(ABC):
    """
    Abstract base class for a Dataset.

    This class is an abstract class from which other subclasses inherit and must not be instanciated directly.
    """

    def __init__(self, name: str) -> None:
        """
        Main constructor of the class.

        Parameters
        ----------
        name : str
            The name given to the dataset.
        """
        self.name = name
        self.dataset_dir = path.join(Directory.DATA, self.name)

    @abstractmethod
    def fetch(self, out_dir: str) -> None:
        """
        Fetch data to feed the pipeline.

        Parameters
        ----------
        out_dir : str
            Directory where the dataset will be stored.
        """

    @abstractmethod
    def dataset_files(self) -> Iterable:
        """
        Iterable of files included in the dataset.
        """

    def show_start_message(self) -> None:
        """
        Show starting step info message.
        """
        print("\n+++ Dataset {} +++".format(self.name))

    def is_dataset_ready(self) -> bool:
        """
        Check if dataset has been already loaded/downloaded and parsed to package format.
        """

        # Check dataset folder exists
        if not path.exists(self.dataset_dir):
            return False

        # Check dataset folder has all dataset files
        dir_files = [path.join(self.dataset_dir, f) for f in listdir(
            self.dataset_dir) if path.isfile(path.join(self.dataset_dir, f))]
        dataset_files = self.dataset_files()

        if len(dataset_files) == 0:
            return False

        for dataset_file in dataset_files:
            if dataset_file not in dir_files:
                return False

        # All OK
        return True


class RemoteDataset(Dataset):
    """
    Abstract base class for a RemoteDataset (downloadable).
    """

    def __init__(self, url: str, file_format: str, checksum: str, checksum_format: str) -> None:
        super().__init__(self.__class__.__name__)
        self.url = url
        self.file_format = file_format
        self.checksum = checksum
        self.checksum_format = checksum_format

    @abstractmethod
    def parse_to_package(self, raw_dir: str) -> None:
        """
        Parse raw dataset to package format.

        Parameters
        ----------
        raw_dir:
            Directory where the dataset was downloaded from its source.
        """

    @abstractmethod
    def feed_pipeline(self, out_dir: str) -> None:
        """
        Copy selected data file to pipeline step folder.
        """

    def fetch(self, out_dir: str) -> None:
        """
        Fetch data to feed the pipeline.

        Parameters
        ----------
        out_dir : str
            Directory where the dataset will be stored.
        """
        self.show_start_message()

        # Check if dataset is already downloaded and parsed to package format
        if not self.is_dataset_ready():
            # Create download (raw) folder and download the dataset
            makedirs(self.dataset_dir, exist_ok=True)
            raw_dir = path.join(self.dataset_dir, "raw")
            self.download(raw_dir)

            # Parse downloaded files to package format
            print("Parsing downloaded files to package format")
            self.parse_to_package(raw_dir)

            # Remove raw files to save disk space
            shutil.rmtree(raw_dir, ignore_errors=True)

        # Copy data to pipeline step directory
        self.feed_pipeline(out_dir)

    def download(self, out_dir: str) -> None:
        """
        Download a Dataset from a remote URL.
        """
        print("Downloading data from " + self.url)
        r = requests.get(self.url, stream=True)
        if not r.ok:
            raise ConnectionError(
                "An error occurred downloading {}.".format(self.name))
        if not self.check_checksum(r):
            raise RuntimeError(
                "{} has a checksum differing from expected, file may be corrupted.").format(self.name)
        self.extract_data(r, out_dir)

    def check_checksum(self, response: object) -> bool:
        """
        Checks if the checksum of the downloaded file corresponds to the one provided in the class.
        For security e integrity issues. Currently only the md5 algorithm is integrated.

        Parameters
        ----------
        response: Object
            Response object returned by the get method of the Requests library.

        Returns
        -------
        Boolean variable indicating whether the comparison of the hash with the checksum was successful or not.
        """
        if self.checksum_format == 'md5':
            # MD5 checksum
            md5 = hashlib.md5(response.content).hexdigest()
            if md5 == self.checksum:
                return True
            else:
                return False

    def extract_data(self, response: object, out_dir: str) -> None:
        """
        Extracts the contents of a compressed file in zip format.

        Parameters
        ----------
        response: Object
            Response object returned by the get method of the Requests library.
        out_dir: str
            Directory were the zip file will be unzziped.

        """
        if self.file_format == 'zip':
            # ZIP file format
            z = zipfile.ZipFile(io.BytesIO(response.content))
            z.extractall(out_dir)


class LocalDataset(Dataset):
    # Habrá dos funciones: una para generar el dataframe en la carpeta data y otro para feedear el pipeline a partir del nombre y la carpeta ya creada, para no tener que pasar siempre un dataframe como parametro
    """
    Abstract base class for a LocalDataset (loadable).
    """

    def __init__(self, path: str) -> None:
        super().__init__(self.__class__.__name__)
        self.path = path


    @abstractmethod
    def load(self, raw_dir: str) -> None:
        """
        Parse raw dataset to package format.

        Parameters
        ----------
        raw_dir:
            Directory where the dataset was originally stored.
        """

    @abstractmethod
    def feed_pipeline(self, out_dir: str) -> None:
        """
        Copy selected data file to pipeline step folder.
        
        Parameters
        ----------
        out_dir:
            Directory where the dataset was originally stored.
        """

    def fetch(self, out_dir: str) -> None:
        """
        Fetch data to feed the pipeline.

        Parameters
        ----------
        out_dir : str
            Directory where the dataset will be stored.
        """
        self.show_start_message()

        # Check if dataset is already in Conmo's datasets folder
        if not self.is_dataset_ready():
            # Create download (raw) folder 
            makedirs(self.dataset_dir, exist_ok=True)
            raw_dir = path.join(self.dataset_dir, "raw")

            # Parse local files to package format
            print("Parsing local files to package format")
            self.load()

            # Remove raw files to save disk space
            shutil.rmtree(raw_dir, ignore_errors=True)

        # Copy data to pipeline step directory
        self.feed_pipeline(out_dir)
