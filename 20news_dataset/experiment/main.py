import argparse
import sys
import torch.nn.functional as F
import torch

sys.path.append('..')
sys.path.append('.')
from utils.utils import mig_loss_function,enu_modalities
from utils.vat import VATLoss

parser = argparse.ArgumentParser(description='20Text')
parser.add_argument('--device', type=int)
parser.add_argument('--epoch', type=int,default=500,help="Number of epochs to train.")
parser.add_argument('--modalities', type=int, default=3, help='Number of modalities')
parser.add_argument('--split', type=int, default=4000, help='Number of feature split')
parser.add_argument('--labeled', type=int, default=500, help='Number of labeled data')
parser.add_argument('--thresh', type=float, default=0.9, help='Threshold')
parser.add_argument('--mig', type=int, default=0, help='MIG or not')
parser.add_argument('--vat', type=int, default=0, help='VAT or not')
parser.add_argument('--alpha', type=float, default=1.0, help='VAT or not')
parser.add_argument('--xi', type=float, default=0.01, help='VAT perturbation( best hyper is default)')


args = parser.parse_args()

from dataset.dataprepare import *
from models.model import *




class Config(object):

    pass

def test(epoch,device):

    acc = np.zeros((args.modalities+1),dtype=np.float32)


    for idx, (data_dict,label) in enumerate(dataloader['test']):

        label = label.long().to(device)

        output_total = 0.0
        for i in range(1, 1 + args.modalities):

            output = Net[i](data_dict[i].to(device))
            pred = torch.max(output, 1)[1]
            acc[i-1] += pred.eq(label.data).cpu().sum()

            output_total += output


        pred = torch.max(output_total, 1)[1]
        acc[args.modalities] += pred.eq(label.data).cpu().sum()


    acc /= float(len(dataloader['test'].dataset))

    print("-- Test Epoch:{}, Acc 1 = {:.2%}, Acc 2 = {:.2%}, Acc 3 = {:.2%}, Acc = {:.2%}".format(epoch, acc[0],acc[1],acc[2],acc[3]))


def VAT(epoch,device,alpha):

    train_loss = 0.0
    vat_loss = VATLoss(xi=0.1, eps=1.0, ip=1)

    for idx, (data_dict, label) in enumerate(dataloader['vat']):

        loss = 0.0
        for i in range(1,1 + args.modalities):
            Optimizer[i].zero_grad()

        for i in range(1, 1 + args.modalities):

            loss += alpha * vat_loss(Net[i], data_dict[i].to(device))

        loss.backward()

        for i in range(1, 1 + args.modalities):
            Optimizer[i].step()

        train_loss += loss.data

    print(">> VAT Train Epoch:{}".format(epoch))



def MIG_train(epoch, device,alpha):

    train_loss = 0.0
    p = torch.FloatTensor([0.25,0.25,0.25,0.25]).to(device)

    acc = np.zeros((args.modalities),dtype=np.float32)
    pairs = enu_modalities(args.modalities)

    for data_dict, label in dataloader['mig']:
        for idx,(i,j) in enumerate(pairs):

            ldata, rdata = data_dict[i].to(device), data_dict[j].to(device)
            label = label.long().to(device)

            Optimizer[i].zero_grad()
            Optimizer[j].zero_grad()
            output_i = torch.exp(Net[i](ldata))
            output_j = torch.exp(Net[j](rdata))

            loss = alpha *  mig_loss_function(output_i,output_j,p,len(label),device=device)
            loss.backward()

            Optimizer[i].step()
            Optimizer[j].step()

            train_loss += loss.data
            pred = torch.max(output_i * output_j, 1)[1]
            acc[idx] += pred.eq(label.data).cpu().sum()


    acc /= float(len(dataloader['mig'].dataset))

    print(">> MIG Train Epoch:{}, Acc 1= {:.2%}, Acc 2 = {:.2%}, Acc 3 = {:.2%}".format(epoch, acc[0], acc[1], acc[2]))


def train(epoch,device):

    for t in range(epoch):

        acc = np.zeros((args.modalities), dtype=np.float32)
        train_loss = 0.0

        for idx, (data_dict,label) in enumerate(dataloader['train']):

            label = label.long().to(device)

            for i in range(1, 1 + args.modalities):

                Optimizer[i].zero_grad()
                output = Net[i](data_dict[i].to(device))
                loss = F.nll_loss(output, label)
                loss.backward()
                Optimizer[i].step()

                pred = torch.max(output, 1)[1]
                acc[i-1] += pred.eq(label.data).cpu().sum()
                train_loss += loss.data


        acc /= float(len(dataloader['train'].dataset))

        #train_loss = float(train_loss)/ float(len(train_loader.dataset))
        print("S Train Epoch:{}, Acc 1= {:.2%}, Acc 2 = {:.2%}, Acc 3 = {:.2%}".format(t, acc[0],acc[1], acc[2]))

        if args.vat:
            VAT(t,device,args.alpha)

        elif args.mig and acc[0] > args.thresh and acc[1] > args.thresh and acc[2] > args.thresh:
            MIG_train(t, device, args.alpha)

        test(t,device)



if __name__ == '__main__':


    train(args.epoch,args.device)
