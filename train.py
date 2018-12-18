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
import numpy as np
import os
from sklearn.metrics import accuracy_score


def get_right_wrong_num(pred, Y):
    '''
    This function is used to compute the number of right samples
    and wrong samples, then return the numbers.
    :param pred: The pred array shape (batch_size, num_class)
    :param Y: The label index array shape (batch_size)
    :return: (right_num, wrong_num)
    '''
    assert pred.shape[0] == Y.shape[0]
    batch_size = pred.shape[0]
    pred_Y = np.argmax(pred, axis=1)  # Get the pred index

    # Get the right number and wrong number
    right = accuracy_score(Y, pred_Y, normalize=False)
    wrong = batch_size - right
    return right, wrong

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

    for epoch in range(conf.NUM_EPOCHS):
        print("############## Training epoch {}#############".format(epoch))
        for batch, (X, Y) in enumerate(imgLoader):
            # Set net as training mode
            net.train()

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

            # If batch % conf.VALPERBATCH == 0, validate
            if (batch + 1) % (conf.VALPERBATCH) == 0:
                # Set evaluation mode
                net.eval()

                right = 0
                wrong = 0
                for tX, tY in evalLoader:
                    if conf.USE_CUDA:
                        tX = tX.cuda()
                    tX = Variable(tX)
                    preds = net(tX)  # Get the prediction result
                    if conf.USE_CUDA:
                        preds = preds.data.cpu().numpy() # Transform preds to numpy array
                    else:
                        preds = preds.data.numpy()
                    # Get the number of right number and wrong number
                    tr, tw = get_right_wrong_num(preds, tY)
                    right += tr
                    wrong += tw
                acc = float(right)/(right + wrong)
                if acc > best_acc:
                    best_acc = acc
                    # Save model
                    torch.save(net, os.path.join(conf.SOURCE_DIR_PATH["MODEL_DIR"], "Bestacc_%s.mdl"%str(best_acc)))
                # Add acc to summary
                writer.add_scalar("scalar/test_acc", acc, epoch*conf.BATCH_SIZE+batch)
                print("*****Acc {:.4f}, best acc {:.4f}".format(acc, best_acc))

            # Add loss to summary
            writer.add_scalar("scalar/loss", float(loss), epoch*conf.BATCH_SIZE+batch)

            print("==> Training epoch {}, batch {}, loss {:.4f}, best loss {:.4f}, ".
                  format(epoch, batch, float(loss),best_loss))


if __name__ == "__main__":
    conf = Config()

    print("==> Load data")
    imgLoader = DataLoader(dataset(path=conf.RAW_TRAIN_DATA),
                           batch_size=conf.BATCH_SIZE,
                           shuffle=True,
                           num_workers=2)
    evalLoader = DataLoader(dataset(path=conf.RAW_TEST_DATA),
                            batch_size=100,
                            shuffle=False,
                            drop_last=False)
    # Initialize SummaryWriter
    writer = SummaryWriter(log_dir=conf.SOURCE_DIR_PATH["SUMMARY_DIR"])

    # Begin to train
    print("==> Begin to train")
    train()

    # close the summary writer
    writer.close()
