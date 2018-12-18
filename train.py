# -*- coding: utf-8 -*-

from data_loader import dataset
from torch.utils.data import DataLoader
from torch import optim
from torch import nn
from torch.autograd import Variable
import torch
from config import Config
from model import PeleeNet
from tensorboardX import SummaryWriter


def train():
    # Initialize the PeleeNet
    net = PeleeNet.PeleeNet(conf)

    # Set best loss and best acc
    best_loss = float('inf')
    best_acc = 0.0

    # Use GPU
    if conf.USE_CUDA:
        print("==> Using CUDA to train")
        net.cuda()
        # https://www.pytorchtutorial.com/when-should-we-set-cudnn-benchmark-to-true/
        torch.backends.cudnn.benchmark = True

    # Set learning rate and set hyper-parameter
    optimizer = optim.Adam(net.parameters(), lr=conf.LEARNING_RATE, weight_decay=1e-4)
    # Set loss
    criterion = nn.CrossEntropyLoss()
    # Set net as training mode
    net.train()

    for epoch in range(conf.NUM_EPOCHS):
        print("############## Training epoch {}#############".format(epoch))
        for batch, (X, Y) in enumerate(imgLoader):
            # transform to cuda
            if conf.USE_CUDA:
                X = X.cuda()
                Y = Y.cuda()
            # transform to Variabele
            X = Variable(X)
            Y = Variable(Y)

            # Set gradients as zero
            optimizer.zero_grad()
            # Forward
            pred = net(X)
            # Compute loss
            loss = criterion(pred, Y)
            # Backward
            loss.backward()
            # Update parameters
            optimizer.step()

            # Save best loss and best acc
            if best_loss > float(loss):
                best_loss = float(loss)
            # Add loss to summary
            writer.add_scalar("scalar/loss", float(loss), epoch*conf.BATCH_SIZE+batch)

            print("==> Training epoch {}, batch {}, loss {:.4f}, best loss {:.4f}".
                  format(epoch, batch, float(loss), best_loss))


if __name__ == "__main__":
    conf = Config()

    print("==> Load data")
    imgLoader = DataLoader(dataset(path=conf.RAW_TRAIN_DATA),
                           batch_size=conf.BATCH_SIZE,
                           shuffle=True,
                           num_workers=2)
    # Initialize SummaryWriter
    writer = SummaryWriter(log_dir=conf.SOURCE_DIR_PATH["SUMMARY_DIR"])

    # Begin to train
    print("==> Begin to train")
    train()

    # close the summary writer
    writer.close()
