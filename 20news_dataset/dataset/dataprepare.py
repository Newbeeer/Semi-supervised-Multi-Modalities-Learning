from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer

import numpy as np
import torch
import torch.utils.data
from experiment.main import args


categories = ['alt.atheism', 'talk.religion.misc','comp.graphics', 'sci.space']

newsgroups_train = fetch_20newsgroups(subset='train',remove=('headers', 'footers', 'quotes'),categories=categories)
newsgroups_test = fetch_20newsgroups(subset='test',remove=('headers', 'footers', 'quotes'),
categories=categories)
vectorizer = TfidfVectorizer()
vectors_train = np.array(vectorizer.fit_transform(newsgroups_train.data).todense(),dtype=np.float32)
print("Mean of Training set:",np.mean(vectors_train),np.max(vectors_train))
vectors_train_target = np.array(newsgroups_train.target)
vectors_test = np.array(vectorizer.transform(newsgroups_test.data).todense(),dtype=np.float32)
vectors_test_target = np.array(newsgroups_test.target,)

print(vectors_train.shape,vectors_train_target.shape,vectors_test.shape,vectors_test_target.shape)


class Text(torch.utils.data.Dataset):

    def __init__(self, data, label, train, labeled = True):

        self.m = args.modalities
        if not labeled:

            if train:
                self.data = self.cons_dict(data[args.labeled: ], args.split, self.m)
                self.label = label[args.labeled:]
            else:
                self.data = self.cons_dict(data, args.split, self.m)
                self.label = label
        else:
            if train:
                self.data = self.cons_dict(data[: args.labeled], args.split, self.m)
                self.label = label[: args.labeled]
            else:
                self.data = self.cons_dict(data, args.split,self.m)
                self.label = label

        self.train = train
        self.labeled = labeled


    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):

        label = self.label[index]
        data_dict = {i: self.data[i][index] for i in range(1, self.m + 1)}

        return data_dict, label

    def cons_dict(self,data,split,m):

        t_dict = dict()
        for i in range(m):
            t_dict[i+1] = data[:, i*split: (i+1)*split]

        return t_dict

class Text_MIG(torch.utils.data.Dataset):

    def __init__(self, data, label):


        self.m = args.modalities
        self.data_dict = self.cons_dict(data,args.split,self.m)
        self.label = label

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):

        label = self.label[index]
        data_dict = {i:self.data_dict[i][index] for i in range(1,self.m+1)}

        return data_dict, label

    def cons_dict(self,data,split,m):

        t_dict = dict()
        for i in range(m):
            t_dict[i+1] = data[:, i*split: (i+1)*split]
        return t_dict



T_train = Text(data = vectors_train,label = vectors_train_target,train= True)
T_test = Text(data = vectors_test,label = vectors_test_target,train= False)
U_train = Text(data = vectors_train,label = vectors_train_target,train= True,labeled=False)
U_MIG = Text_MIG(data = vectors_train,label = vectors_train_target)


dataloader = {'train':torch.utils.data.DataLoader(dataset = T_train, batch_size = 64),
'test':torch.utils.data.DataLoader(dataset = T_test, batch_size = 128),
'mig':torch.utils.data.DataLoader(dataset = U_MIG, batch_size = 64),
'vat':torch.utils.data.DataLoader(dataset = U_train, batch_size = 64)}