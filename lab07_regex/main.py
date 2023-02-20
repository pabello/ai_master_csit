import pandas as pd
import numpy as np

from random import choices, shuffle, randint, seed
from sklearn.model_selection import train_test_split

from torch.utils.data import DataLoader, TensorDataset
from torch import nn, tensor, float32
from torchmetrics.classification import BinaryAccuracy

import torch.optim as optimizers
import plotly.express as px



EPOCHS = 100
SET_SIZE = 10000
TEST_SIZE = 0.2
CODE_TABLE = {"a": [1,0,0,0], "b": [0,1,0,0], "c":[0,0,1,0], "d": [0,0,0,1]}

def generate_dataset(regex:str):
    charset = ["a", "b", "c", "d"]
    
    dataset = []
    labels = []
    contain_count = 0
    not_contain_count = 0
    
    while not_contain_count < SET_SIZE//2:
        dataset.append(choices(charset, k=15))
        if regex in ''.join(dataset[-1]):
            contain_count += 1
            labels.append(1)
        else:
            not_contain_count += 1
            labels.append(0)
    
    regex_as_list = list(regex)
    while contain_count < SET_SIZE//2:
        split = randint(0, 10)
        left = choices(charset, k=split)
        right = choices(charset, k=10-split)
        dataset.append(left + regex_as_list + right)
        contain_count += 1
        labels.append(1)
    
    return dataset, labels


def one_hot_encode_data(dataset:list[str]) -> list[list[int]]:
    for i in range(len(dataset)):
        for j in range(len(dataset[i])):
            dataset[i][j] = CODE_TABLE[dataset[i][j]]


def shuffle_data_and_labels(dataset_input:list[list[int]], labels_input:list[int]) -> tuple[list[list[int]], list[int]]:
    all_data = [[dataset_input[i], labels_input[i]] for i in range(len(dataset_input))]
    shuffle(all_data)
    dataset = [data[:-1] for data in all_data]
    labels = [data[-1] for data in all_data]
    return dataset, labels


if __name__ == "__main__":
    regex = 'acabd'
    dataset, labels = generate_dataset(regex)
    
    one_hot_encode_data(dataset)
    dataset, labels = shuffle_data_and_labels(dataset, labels)
    train_data, test_data = train_test_split(dataset, test_size=TEST_SIZE)
    train_labels, test_labels = train_test_split(labels, test_size=TEST_SIZE)

    inputs = tensor(np.transpose(train_data, (0, 1, 3, 2)), dtype=float32).squeeze(1)
    labels = tensor(train_labels, dtype=float32).unsqueeze(1)
    tensor_dataset = TensorDataset(inputs, labels)
    data_loader = DataLoader(tensor_dataset, batch_size=SET_SIZE, shuffle=True)
    
    test_data = tensor(np.transpose(test_data, (0, 1, 3, 2)), dtype=float32).squeeze(1)
    test_labels = tensor(test_labels, dtype=float32).unsqueeze(1)

    # 1 Convolutional + 1 Dense
    model = nn.Sequential(
        nn.Conv1d(in_channels=4, out_channels=1, kernel_size=5),
        nn.Flatten(),
        nn.Linear(in_features=11, out_features=1),
        nn.Sigmoid()
    )
    
    # # 2 Conv + MaxPool + 2 Conv + 1 Dense + ReLU + 1 Dense + Sigm
    # model = nn.Sequential(
    #     nn.Conv1d(in_channels=4, out_channels=15, kernel_size=5),
    #     nn.Conv1d(in_channels=15, out_channels=25, kernel_size=1),
    #     nn.MaxPool1d(kernel_size=2, padding=1),
    #     nn.Conv1d(in_channels=25, out_channels=15, kernel_size=1),
    #     nn.Conv1d(in_channels=15, out_channels=1, kernel_size=3),
    #     nn.Linear(in_features=4, out_features=20),
    #     nn.ReLU(),
    #     nn.Linear(in_features=4, out_features=1),
    #     nn.Sigmoid()
    # )
    
    optimizer = optimizers.Adam(params=model.parameters(), lr=0.001)
    # optimizer = optimizers.SGD(params=model.parameters(), lr=0.5)
    loss_function = nn.BCEWithLogitsLoss()
    accuracy_function = BinaryAccuracy()
    for epoch in range(EPOCHS):
        for  inputs_batch, labels_batch in data_loader:
            optimizer.zero_grad()  # clear the optimizer state
            accuracy_function.zero_grad()
            
            outputs = model(inputs_batch)#.squeeze(2)
            loss = loss_function(outputs, labels_batch)
            loss.backward()  # backpropagation
            optimizer.step()  # updates the weights
            
            accuracy = accuracy_function(outputs, labels_batch)
        if not (epoch+1) % 5:
            print('Epoch [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}'.format(epoch+1, EPOCHS, loss.item(), accuracy.item()))
    
    print()
    test_outputs = model(test_data)#.squeeze(2)
    test_loss = loss_function(test_outputs, test_labels)
    test_accuracy = accuracy_function(test_outputs, test_labels)
    print('Testing the model | Loss: {:.4f}, Accuracy: {:.2f}'.format(test_loss.item(), test_accuracy.item()))
    print()
    
    kernels = list(model.children())[0].weight.cpu().detach().clone()
    print(kernels)
    fig = px.imshow(kernels[0], text_auto=True)
    fig.update_coloraxes(showscale=True, colorscale="Greys")
    fig.show()