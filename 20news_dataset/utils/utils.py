import torch
import numpy as np


def accuracy(output, labels, test=False):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    if test:
        return correct/len(labels), preds.eq(labels)
    else:
        return correct / len(labels)

def mig_loss_function(output1, output2, p, bs, device):

    I = torch.FloatTensor(np.eye(bs), ).to(device)
    E = torch.FloatTensor(np.ones((bs, bs))).to(device)
    normalize_1 = bs
    normalize_2 = bs * bs- bs
    new_output = output1 / p
    m =  (new_output @ output2.transpose(1,0))
    noise = torch.rand(1).to(device)*0.0001
    m1 = torch.log(m * I + I * noise + E - I)
    m2 = m * (E-I)


    return -(m1.sum() + bs) / normalize_1 + m2.sum() / normalize_2

def enu_modalities(m_num):
    p = []
    for i in range(1,m_num+1):
        for j in range(i+1,m_num+1):
            p += [[i,j]]

    return p