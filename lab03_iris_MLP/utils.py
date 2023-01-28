import pandas as pd
from numpy.random import shuffle


def load_iris_dataset(path:str, training_set_ratio:float=0.8) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Loads Iris dataset, encodes the labels with one-hot encoding
    and divides into trainins and test subsets according to a given ratio.

    Args:
        path (str): path to the dataset file
        training_set_ratio (float, optional): What part of the whole dataset
        is going to be the training subset. Defaults to 0.8.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: Two subsets - 1:training subset; 2:test subset
    """
    df = pd.read_csv(path, header=None)
    __encode_labels(df)
    return __divide_dataset(df)


def __encode_labels(dataset:pd.DataFrame):
    for i in range(len(dataset)):
        label = dataset.iloc[i][4]
        vector = (int(label=="Iris-setosa"), int(label=="Iris-versicolor"), int(label=="Iris-virginica"))
        dataset.iat[i,4] = vector


def __divide_dataset(dataset:pd.DataFrame, training_set_ratio:float=0.8):
    training_records_count = int(len(dataset) * training_set_ratio)
    dataset = dataset.sample(frac=1).reset_index(drop=True)  # shuffling the dataset
    
    training_set = dataset[:training_records_count]
    testing_set = dataset[training_records_count:].reset_index(drop=True)
    return training_set, testing_set


if __name__ == "__main__":
    trs, tes = load_iris_dataset('iris.data')
    print(tes)