import numpy as np
import torch
from sklearn.metrics import f1_score, accuracy_score, multilabel_confusion_matrix
from torcheval.metrics import BinaryAccuracy, BinaryF1Score
from sklearn import preprocessing
import sys

def R2(pred, true):
    mean_true = true.mean() # the mean of the true values
    ss_residual = ((true - pred) ** 2).sum() # the sum of squares of the residuals
    ss_total = ((true - mean_true) ** 2).sum() # the total sum of squares
    r2 = 1 - (ss_residual / ss_total) # the R^2 score
    return r2

def RSE(pred, true):
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))

def CORR(pred, true):
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
    d += 1e-12
    return 0.01*(u / d).mean(-1)


def MAE(pred, true):
    return np.mean(np.abs(pred - true))


def MSE(pred, true):
    return np.mean((pred - true) ** 2)


def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))


def MAPE(pred, true):
    return np.mean(np.abs((pred - true) / true))


def MSPE(pred, true):
    return np.mean(np.square((pred - true) / true))

def discretize_prices(prices, low_threshold, high_threshold):
    labels = []
    daily_pct_change = np.diff(prices) / prices[:-1]
    for change in daily_pct_change:
        if change < low_threshold:
            labels.append(0)  # low
        else:
            #change > high_threshold:
            labels.append(1)  # high
    return torch.tensor(labels)

def ACCURACY(pred, true):
    torch.set_printoptions(profile="full")
##    print(len(true.flatten()))
##    print(len(pred.flatten()))
##    true = np.array(true)
##    pred = np.array(pred.flatten())
    #nsamples, nx, ny = true.shape
    #min_max_scaler = preprocessing.MinMaxScaler()
    #true = min_max_scaler.fit_transform(true.reshape((nsamples,nx*ny)))
    #pred = min_max_scaler.fit_transform(pred.reshape((nsamples,nx*ny)))
    #print(true)
    #print(pred)
    #true = discretize_prices(true.flatten(), 0, 0)
    #pred = discretize_prices(pred.flatten(), 0, 0)
    #return accuracy_score(true.numpy().astype(int), pred.numpy().astype(int))
    # np.set_printoptions(threshold=sys.maxsize)
    
    # Reshape true to (440, 56)
    # true = true.reshape(true.shape[0], -1)
    
    
    # Take the class with the highest predicted probability
    #print(pred)
    #print(true)
    #pred = np.argmax(pred, axis=-1)
    #true = np.argmax(true, axis=-1)
    true_binary = np.where(true < 0, 0, 1)
    pred_binary = np.where(pred < 0.5, 0, 1)
    #print(true_binary)
    #print(pred_binary)
    true_binary = true_binary.reshape(-1)
    pred_binary = pred_binary.reshape(-1)
    print(f'{len(true_binary)}, {len(pred_binary)}')
    acc = accuracy_score(true_binary, pred_binary)
    return acc

def F1(pred, true):
##    true = np.array(true.flatten())
##    pred = np.array(pred.flatten())
    #true = true.reshape(true.shape[0], -1)
    
    # Take the class with the highest predicted probability
    #pred = np.argmax(pred, axis=-1)
    #true = np.argmax(true, axis=-1)
    true_binary = np.where(true < 0, 0, 1)
    pred_binary = np.where(pred < 0.5, 0, 1)
    true_binary = true_binary.reshape(-1)
    pred_binary = pred_binary.reshape(-1)
    f1 = f1_score(true_binary, pred_binary, average='weighted')
    return f1
##    nsamples, nx, ny = true.shape
##    min_max_scaler = preprocessing.MinMaxScaler()
##    true = min_max_scaler.fit_transform(true.reshape((nsamples,nx*ny)))
##    pred = min_max_scaler.fit_transform(pred.reshape((nsamples,nx*ny)))
##    true = discretize_prices(true.flatten(), 0, 0)
##    pred = discretize_prices(pred.flatten(), 0, 0)
##    return f1_score(true.numpy().astype(int), pred.numpy().astype(int), average='weighted')

def metric(pred, true):
    #print(f"Shape of true before calling ACCURACY: {true.shape}")
    #print(f"Shape of pred before calling ACCURACY: {pred.shape}")
    r2 = R2(pred, true)
    rse = RSE(pred, true)
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)
    accuracy = ACCURACY(pred, true)
    f1 = F1(pred, true)

    return mae, mse, rmse, mape, mspe, rse, accuracy, f1
