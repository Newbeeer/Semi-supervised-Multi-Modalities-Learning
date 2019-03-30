from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer

import numpy as np
import argparse

parser = argparse.ArgumentParser(description='20Text')
parser.add_argument('--device', type=int)
parser.add_argument('--epoch', type=int,default=500,help="Number of epochs to train.")
parser.add_argument('--modalities', type=int, default=3, help='Number of modalities')
parser.add_argument('--split', type=int, default=4000, help='Number of feature split')
parser.add_argument('--labeled', type=int, default=500, help='Number of labeled data')
parser.add_argument('--thresh', type=float, default=0.9, help='Threshold')
parser.add_argument('--mig', type=int, default=0, help='MIG or not')
parser.add_argument('--vat', type=int, default=0, help='VAT or not')
parser.add_argument('--lambda2', type=float, default=10.0, help='VAT or not')
parser.add_argument('--lambda1', type=float, default=1.0, help='VAT perturbation( best hyper is default)')
args = parser.parse_args()

categories = ['alt.atheism', 'talk.religion.misc','comp.graphics', 'sci.space']

newsgroups_train = fetch_20newsgroups(subset='train',remove=('headers', 'footers', 'quotes'),categories=categories)
newsgroups_test = fetch_20newsgroups(subset='test',remove=('headers', 'footers', 'quotes'),
categories=categories)
vectorizer = TfidfVectorizer()

vectors_train = np.array(vectorizer.fit_transform(newsgroups_train.data).todense(),dtype=np.float32)
vectors_train_target = np.array(newsgroups_train.target)
vectors_test = np.array(vectorizer.transform(newsgroups_test.data).todense(),dtype=np.float32)
vectors_test_target = np.array(newsgroups_test.target,)
print("Shape of training set: ", vectors_train.shape, "Shape of testing set: ",vectors_test.shape)


v = args.modalities
N = vectors_train.shape[0]
lambda1 = args.lambda1
lambda2 = args.lambda2
maxIterPVCC = 150
epsilon = 1e-6
SS = 1e-1 #step size
class_num = 4
numL = 500 # number of the labeled datapoint
numC = N #TODO inc
H = np.zeros((v,N,N))
C = np.zeros((v,N,N))
s = np.zeros((v,N,N)) #view-specific similarity matrix S_i
M = np.zeros((N,N)) #global similarity matrix M
A = dict()
B = dict()
#Dimention of data point
data_dim = np.zeros((v),dtype=np.int32)
#Complete modalities in every modalities
eta = np.zeros((v))
eta += N #Complete dataset for now TODO inc

X = np.zeros((v,N,args.split))
X_test = np.zeros((v,vectors_test.shape[0],args.split))

for i in range(v):
    X[i] = vectors_train[:, i * args.split : (i + 1)*args.split]
    X_test[i] = vectors_test[:, i * args.split: (i + 1) * args.split]
    data_dim[i] = args.split


for i in range(v):
    C[i] = np.eye(N) - np.ones((N,N)) / eta[i] #TODO inc
    B[i] = X[i].T.dot(C[i].T)
    A[i] = B[i].dot(B[i].T) + eta[i] * lambda1 * np.eye(data_dim[i])

for i in range(v):
    tmpX = np.zeros((N,data_dim[i]))
    for j in range(N):
        n = np.sqrt( (X[i,j] ** 2).sum() )
        if n !=0 :

            tmpX[j] = X[i,j] / n   #normalize per row
        else:
            tmpX[j]  += 0
    s[i] = tmpX.dot(tmpX.T)
    M = M + s[i]
M = M/v
f = np.zeros((N,class_num))
for i in range(numL):
    f[i][vectors_train_target[i]] = 1

'''
for i in range(numL,N):

    nearest = np.argmin(M[i,:numL])
    f[i][vectors_train_target[nearest]] = 1
'''
#TODO unlabeled part initilization


sumH = np.zeros((N,N))
aMb = dict()
for i in range(v):
    aMb[i] = np.linalg.inv(A[i]).dot(B[i])
    H[i] = C[i].dot(C[i].T).dot(aMb[i].T).dot(lambda1/2 * aMb[i] + 1/(2 * eta[i])*B[i].dot(B[i].T).dot(aMb[i]) - 1/eta[i]*B[i]) + 1/(2*eta[i])*C[i].T.dot(C[i])
    #TODO inc
    sumH += H[i]

f_old = f.copy()
for i in range(maxIterPVCC):
    print('Optimization step %d'%i)
    grad_1 = sumH.dot(f)
    M_approx = f.dot(f.T)
    grad_2 = np.zeros((N,class_num))
    for j in range(v):
        dif = M_approx - s[j]
        n = np.sqrt( (dif ** 2).sum() )
        if n == 0:
            proj = 0
        else:
            proj = dif / n #TODO inc
        grad_2 += proj.dot(f)

    grad_f = grad_1 + grad_2 * lambda2
    f -= SS * grad_f
    f = f.clip(0,1)
    for i in range(numL):
        f[i] *= 0
        f[i][vectors_train_target[i]] = 1

    pred = np.argmax(f,axis=1)
    acc = (pred == vectors_train_target).sum()
    acc = float(acc) / float(vectors_train_target.shape[0])
    print(f)
    print("Training Accuracy: {:.2%}".format(acc))

W = dict()
b = np.zeros((v,class_num))
for i in range(v):

    W[i] = aMb[i].dot(C[i]).dot(f)

    b[i] = ((f - X[i].dot(W[i])).T.dot(np.ones((N,1)))/eta[i]).reshape(b[i].shape)

test_predict = np.zeros((vectors_test.shape[0],class_num))
for i in range(v):
    test_predict += X_test[i].dot(W[i]) + b[i]


test_predict = np.argmax(test_predict,axis=1)
print(test_predict.shape,test_predict)
acc = (test_predict ==  vectors_test_target).sum()
acc = float(acc) / float(vectors_test_target.shape[0])
print("Accuracy: {:.2%}".format(acc))