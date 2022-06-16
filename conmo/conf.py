from os import path


class Directory:
    DATA = path.join(path.expanduser("~"), "conmo", "data")
    EXPERIMENTS = path.join(path.expanduser("~"), "conmo", "experiments")


class File:
    DATA = "data.gz"
    LABELS = "labels.gz"


class Index:
    FOLD = "fold"
    SET = "set"
    SET_TRAIN = "train"
    SET_TEST = "test"
    SEQUENCE = "sequence"
    TIME = "time"


class Label:
    ANOMALY = "anomaly"
    RUL = "rul"
    BATTERIES_DEG_TYPES = ["LLI", "LAMPE", "LAMNE"]


class Testing:
    RANDOM_SEED = 22


class RandomSeed:
    RANDOM_SEED = 22