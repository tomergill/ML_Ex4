import torch as tr
from torch import nn
from torch import optim
from time import time
from torchvision.datasets import FashionMNIST
from random import shuffle
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import transforms
from collections import OrderedDict
from sys import argv
from matplotlib import pyplot as plt


def loss_and_accuracy_on(net, criterion, dev_loader, return_results=False):
    """
    Predicts and computes average loss per example and accuracy
    :param net: Neural Net object
    :param criterion: Loss function module
    :param dev_loader: Data loader to predict on
    :param return_results: If true returns the predictions as a list
    :return: avg. loss, accuracy [, results]
    """
    total_loss = good = 0.0
    if return_results:
        results = []

    net.eval()  # evaluation mode (for dropout)
    for x, y in dev_loader:
        x = x.reshape((-1, 28 * 28))  # shape from batch X 28 X 28 to batch X 784
        out = net.forward(x)
        total_loss += criterion(out, y).item()
        pred = tr.argmax(out, dim=1)
        good += (pred == y).sum().item()
        if return_results:
            results.extend(pred.tolist())

    size = len(dev_loader.sampler)
    if return_results:
        return total_loss / size, good / size, results
    return total_loss / size, good / size


def train_on(net, criterion, optimizer, train_loader, dev_loader, epochs=10):
    """
    Trains on train data and computes avg. loss and accuracy for train and dev each epoch and prints them
    :param net: Neural Net object
    :param criterion: Loss function module
    :param optimizer: Pytorch Optimizer
    :param train_loader: Data loader for train set
    :param dev_loader: Data loader for validation set
    :param epochs: Number of epochs to train net on
    :return: list of avg. train loss at each epoch, list of avg. validation loss at each epoch
    """
    print "+-------+------------+-----------+----------+---------+------------+"
    print "| epoch | train loss | train acc | dev loss | dev acc | epoch time |"
    print "+-------+------------+-----------+----------+----------------------+"

    all_train_loss, all_dev_loss = [], []
    for i in xrange(epochs):
        total = total_loss = good = 0.0
        start = time()
        net.train()  # training mode
        for x, y in train_loader:
            x = x.reshape((-1 , 28*28))  # shape from batch X 28 X 28 to batch X 784

            optimizer.zero_grad()
            out = net.forward(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total += train_loader.batch_size
            correct = (tr.argmax(out, dim=1) == y).sum().item()
            good += correct

        dev_loss, dev_acc = loss_and_accuracy_on(net, criterion, dev_loader)
        size = len(train_loader.sampler)
        train_loss = total_loss / size

        print "| {:^5} | {:010f} | {:8.4f}% | {:7f} | {:6.3f}% | {:08f}s |".format(
            i, train_loss, good / size * 100.0, dev_loss,
               dev_acc * 100.0, time() - start)

        all_train_loss.append(train_loss)
        all_dev_loss.append(dev_loss)

    print "+-------+------------+-----------+----------+---------+------------+\n"
    return all_train_loss, all_dev_loss


def split_train(train_set, train_part=0.8, batch_size=1):
    """
    Splits the data set into 2 data loader using SubsetRandomSampler
    by counting how many examples should be for each class
    :param train_set: Dataset to split
    :param train_part: Fraction representing train loader part (e.g 0.8 = train is 80% from dataset)
    :param batch_size: Batch size for loaders
    :return: train dataloader, validation dataloader
    """
    size = len(train_set)
    train_size = int(size * train_part)
    class_size = int(train_size / 10.0)
    indices = range(size)
    shuffle(indices)

    train_indices = []
    counters = {i: 0 for i in xrange(10)}
    for i in indices:
        if counters[int(train_set[i][1])] < class_size:
            train_indices.append(i)
            counters[int(train_set[i][1])] += 1
    dev_indices = [i for i in indices if i not in train_indices]
    # train_indices, dev_indices = indices[:train_size], indices[train_size:]
    train_sampler, dev_sampler = SubsetRandomSampler(train_indices), SubsetRandomSampler(dev_indices)
    return DataLoader(train_set, batch_size=batch_size, num_workers=2, sampler=train_sampler), \
           DataLoader(train_set, batch_size=batch_size, num_workers=2, sampler=dev_sampler)


def main():
    """
    Main function, used to create net and read data. trains the net and tests it
    :return: None
    """
    use_dropout = "--d" in argv or "--dropout" in argv
    use_batch_norm = "--bn" in argv or "--batch_norm" in argv

    # parameters
    input_size = 28 * 28
    hidden1_size = 100
    hidden2_size = 50
    output_size = 10

    dropout_prob = 0.1
    learning_rate = 0.001
    batch = 500

    # print parameters
    print "#" * 68
    print "My Parameters:"
    print "*\tBatch Size: {}".format(batch)
    print "*\tLearning Rate (eta): {}".format(learning_rate)
    print "*\tDropout: {}".format("On" if use_dropout else "Off")
    if use_dropout:
        print "*\tDropout Probability: {}".format(dropout_prob)
    print "*\tBatch Normalization: {}".format("On" if use_batch_norm else "Off")
    print "#" * 68 + "\n"

    # load data
    train_set = FashionMNIST("./data", train=True, download=True, transform=transforms.ToTensor())
    train_loader, dev_loader = split_train(train_set, batch_size=batch)
    test_set = FashionMNIST("./data", train=False, download=True, transform=transforms.ToTensor())
    test_loader = DataLoader(test_set, batch_size=batch, shuffle=True, num_workers=2)

    # Neural net
    modules = [("fc1", nn.Linear(input_size, hidden1_size))]  # First Hidden Layer
    if use_batch_norm:
        modules.append(("bn1", nn.BatchNorm1d(hidden1_size)))
    modules.append(("r1", nn.ReLU()))
    if use_dropout:
        modules.append(("d1", nn.Dropout(dropout_prob)))
    modules.append(("fc2", nn.Linear(hidden1_size, hidden2_size)))  # Second Hidden Layer
    if use_batch_norm:
        modules.append(("b2", nn.BatchNorm1d(hidden2_size)))
    modules.append(("r2", nn.ReLU()))
    if use_dropout:
        modules.append(("d2", nn.Dropout(dropout_prob)))
    modules.append(("fc3", nn.Linear(hidden2_size, output_size)))  # Output Layer
    modules.append(("soft", nn.LogSoftmax(dim=1)))

    net = nn.Sequential(OrderedDict(modules))

    # net = nn.Sequential(
    #     # First Hidden Layer:
    #     nn.Linear(input_size, hidden1_size), nn.BatchNorm1d(hidden1_size), nn.ReLU(), nn.Dropout(dropout_prob),
    #     # Second Hidden Layer:
    #     nn.Linear(hidden1_size, hidden2_size), nn.BatchNorm1d(hidden2_size), nn.ReLU(), nn.Dropout(dropout_prob),
    #     # output layer
    #     nn.Linear(hidden2_size, output_size), nn.LogSoftmax(dim=1)
    # )
    criterion = nn.NLLLoss(size_average=False)
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)

    # train
    train_loss, dev_loss = train_on(net, criterion, optimizer, train_loader, dev_loader)

    # plot graph
    plt.plot(range(10), train_loss, "r", label="Training Set")
    plt.plot(range(10), dev_loss, "b--", label="Validation Set")
    plt.axis([0, 9, 0.0, 0.7])
    plt.xlabel("Epoch Number")
    plt.ylabel("Average Loss per Example")
    plt.legend()
    plt.show()

    # test
    test_loss, test_acc, test_pred = loss_and_accuracy_on(net, criterion, test_loader, return_results=True)
    print "\nLoss on test is {}, and accuracy is {}%".format(test_loss, test_acc * 100)

    # write to test.pred or a verion of it
    name = "test"
    if use_dropout:
        name = "dropout_" + name
    if use_batch_norm:
        name = "batch_norm_" + name
    name += ".pred"
    with open(name, "w") as f:
        f.writelines(map(lambda x: str(int(x)) + "\n", test_pred))


if __name__ == '__main__':
    main()
