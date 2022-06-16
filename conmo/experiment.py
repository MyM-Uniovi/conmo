from abc import ABC
from datetime import datetime
from os import makedirs, path
from typing import Iterable, Optional

from conmo.algorithms.algorithm import Algorithm
from conmo.conf import Directory
from conmo.datasets.dataset import Dataset
from conmo.metrics.metric import Metric
from conmo.preprocesses.preprocess import Preprocess
from conmo.splitters.splitter import Splitter


class Pipeline(ABC):
    DATASET_FOLDER = "01_Dataset"
    SPLITTER_FOLDER = "02_Splitter"
    PREPROCESSES_FOLDER = "03_Preprocesses"
    ALGORITHMS_FOLDER = "04_Algorithms"
    METRICS_FOLDER = "05_Metrics"

    def __init__(self, dataset: Dataset, splitter: Optional[Splitter], preprocesses: Optional[Iterable[Preprocess]], algorithms: Iterable[Algorithm], metrics: Iterable[Metric]) -> None:
        self.dataset = dataset
        self.splitter = splitter
        self.preprocesses = preprocesses
        self.algorithms = algorithms
        self.metrics = metrics

    def run(self, pipe_dir: str, pipe_num: int, pipes: int) -> None:
        """
        Contains all the logic for the execution of a particular pipeline, creating intermediate 
        directories for data passing and executing the relevant methods for each step.

        Parameters
        ----------
        pipe_dir: str
            Name of the current pipeline directory.
        pipe_num: int
            Index of the current pipelines.
        pipes: int
            Total number of pipelines in the current experiment.
        """
        print("\n**** START PIPELINE {:02}/{:02} ****".format(pipe_num, pipes))
        self.generate_dirs(pipe_dir)
        in_dir = None
        out_dir = None

        # Dataset
        out_dir = path.join(pipe_dir, self.DATASET_FOLDER)
        self.dataset.fetch(out_dir)

        # Splitter
        if self.splitter != None:
            in_dir = out_dir
            out_dir = path.join(pipe_dir, self.SPLITTER_FOLDER)
            self.splitter.split(in_dir, out_dir)

        # Preprocesses
        if self.preprocesses != None:
            for idx, preprocess in enumerate(self.preprocesses):
                in_dir = out_dir
                out_dir = path.join(pipe_dir, self.PREPROCESSES_FOLDER, "{:02}_{}".format(
                    idx+1, preprocess.__class__.__name__))
                preprocess.apply(in_dir, out_dir)

        # Algorithms
        in_dir = out_dir
        out_dir = path.join(pipe_dir, self.ALGORITHMS_FOLDER)
        algs = []
        for idx, algorithm in enumerate(self.algorithms):
            algs.append(algorithm.execute(idx+1, in_dir, out_dir))

        # Metrics
        last_preprocess_dir = in_dir
        algorithms_dir = out_dir
        metrics_dir = path.join(pipe_dir, self.METRICS_FOLDER)
        for idx, metric in enumerate(self.metrics):
            metric.calculate(idx+1, algs, last_preprocess_dir,
                             algorithms_dir, metrics_dir)

        print("\n**** END PIPELINE {:02}/{:02} ****\n".format(pipe_num, pipes))

    def generate_dirs(self, pipe_dir: str) -> None:
        """
        Auxiliary method to generate directories for each of the steps in the current pipeline.

        Parameters
        ----------
        pipe_dir: str
            Name of the pipe directory.

        """
        # Generate first level of directories
        makedirs(path.join(pipe_dir, self.DATASET_FOLDER))
        if self.splitter != None:
            makedirs(path.join(pipe_dir, self.SPLITTER_FOLDER))
        if self.preprocesses != None and len(self.preprocesses) > 0:
            makedirs(path.join(pipe_dir, self.PREPROCESSES_FOLDER))
        makedirs(path.join(pipe_dir, self.ALGORITHMS_FOLDER))
        makedirs(path.join(pipe_dir, self.METRICS_FOLDER))

        # Preprocesses: second level
        if self.preprocesses != None and len(self.preprocesses) > 0:
            for idx, preprocess in enumerate(self.preprocesses):
                makedirs(path.join(pipe_dir, self.PREPROCESSES_FOLDER,
                                "{:02}_{}".format(idx+1, preprocess.__class__.__name__)))


class Experiment(ABC):

    def __init__(self, pipelines: Iterable[Pipeline], analytics: Iterable, name=datetime.now().strftime('%Y_%m_%d-%H_%M_%S')):
        self.pipelines = pipelines
        self.analytics = analytics
        self.name = name

    def launch(self):
        """
        Launchs the current experiment.
        """
        print("\n##### EXPERIMENT {} #####".format(self.name))
        pipes_dirs = self.generate_dirs()

        # Pipelines
        n_pipes = len(self.pipelines)
        for idx, pipeline in enumerate(self.pipelines):
            pipeline.run(pipes_dirs[idx], idx+1, n_pipes)

        # Analytics
        # TODO

    def generate_dirs(self) -> Iterable[str]:
        """
        Generates directories both for the experiment itself and for the pipelines it contains.

        Returns
        -------
        pipes_dirs: Iterable[str]
            Array containing the names of the directories of the pipes of the current experiment.
        """
        # Experiment dir
        exp_dir = path.join(Directory.EXPERIMENTS, self.name)
        makedirs(exp_dir, exist_ok=True)

        # Pipelines dirs
        pipes_dirs = []
        for idx, pipeline in enumerate(self.pipelines):
            pipe_dir = path.join(exp_dir, "{:02}_Pipeline".format(idx+1))
            makedirs(pipe_dir, exist_ok=True)
            pipes_dirs.append(pipe_dir)

        return pipes_dirs
