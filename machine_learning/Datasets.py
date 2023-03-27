import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
class MyDataSets():
    def __init__(self,features,labels,foldn,splitn,isshuffle):
        self.features=np.asarray(features)
        self.labels=np.asarray(labels)
        self.foldn=foldn
        self.splitn=splitn
        self.isshuffle=isshuffle
 

    def combine_data(self):
        features=np.concatenate(self.features,axis=0)
        if len(self.labels[0].shape)==1:
            labels=np.concatenate(self.labels)
            labels=self.labels[:,np.newaxis]
        elif len(self.labels[0].shape)==2:
            labels=np.concatenate(self.labels,axis=0)
        return features, labels  

    def shuffle(self,features,labels):
        index=np.arange(features.shape[0])
        np.random.shuffle(index)
        features=features[index]
        labels=labels[index]
        return features, labels

    def sort(self,features,labels):
        index=np.argsort(labels,axis=0)
        index=np.squeeze(index)
        features=features[index]
        labels=labels[index]
        return features, labels


    def split(self):
        if self.foldn==1:
            all_features=np.copy(self.features)
            all_labels=np.copy(self.labels)
            all_val_features=np.flipud(self.features)
            all_val_labels=np.flipud(self.labels)
            for i in range(len(all_features)):
                train_features=all_features[i]
                train_labels=all_labels[i]
                val_features=all_val_features[i]
                val_labels=all_val_labels[i]
                yield {'train_features':train_features, 'train_labels':train_labels, 'val_features':val_features, 'val_labels':val_labels}

        elif self.foldn>1:
            features,labels=self.combine_data()
            if self.isshuffle:
                features,labels=self.shuffle(features,labels)
            else:
                features,labels=self.sort(features,labels)
            all_features=[]
            all_labels=[]
            for i in range(self.foldn):
                all_features.append(features[i:features.shape[0]:self.foldn])
                all_labels.append(labels[i:features.shape[0]:self.foldn])
            all_features=np.array(all_features)
            all_labels=np.array(all_labels)
            if self.splitn==2:
                for i in range(self.foldn):
                    train_features=np.delete(all_features,i,0)
                    train_labels=np.delete(all_labels,i,0)
                    train_features=np.concatenate(train_features,axis=0)
                    train_labels=np.concatenate(train_labels,axis=0)
                    val_features=all_features[i]
                    val_labels=all_labels[i]
                    yield {'train_features':train_features, 'train_labels':train_labels, 'val_features':val_features, 'val_labels':val_labels}
            elif self.splitn==3:
                for i in range(self.foldn):
                    tmp_features=np.delete(all_features,i,0)
                    tmp_labels=np.delete(all_labels,i,0)
                    test_features=all_features[i]
                    test_labels=all_labels[i]
                    all_train_features=[]
                    all_val_features=[]
                    all_train_labels=[]
                    all_val_labels=[]
                    for j in range(self.foldn-1):
                        train_features=np.delete(tmp_features,j,0)
                        train_labels=np.delete(tmp_labels,j,0)
                        train_features=np.concatenate(train_features,axis=0)
                        train_labels=np.concatenate(train_labels,axis=0)
                        val_features=tmp_features[j]
                        val_labels=tmp_labels[j]
                        all_train_features.append(train_features)
                        all_train_labels.append(train_labels)
                        all_val_features.append(val_features)
                        all_val_labels.append(val_labels)
                    yield {'all_train_features':all_train_features, 'all_train_labels':all_train_labels, 'all_val_features':all_val_features, 'all_val_labels':all_val_labels, 'test_features':test_features,'test_labels':test_labels}

    def nest_split(self, inner_foldn, is_random=False):
            if self.foldn<=1 or self.splitn<=2:
                print("Number of folds of outer CV must >1 and number of groups must be 3")
            else:
                features,labels=self.combine_data()
                if self.isshuffle:
                    features,labels=self.shuffle(features,labels)
                else:
                    features,labels=self.sort(features,labels)
                all_features=[]
                all_labels=[]
                # for i in range(self.foldn):
                #     all_features.append(features[i:features.shape[0]:self.foldn])
                #     all_labels.append(labels[i:features.shape[0]:self.foldn])
                all_features=np.array(all_features)
                all_labels=np.array(all_labels)
                for i in range(self.foldn):
                    all_index=np.arange(features.shape[0])
                    test_index=np.arange(i,features.shape[0],self.foldn)
                    out_train_index=np.delete(all_index,test_index)
                    test_features=features[test_index]
                    test_labels=labels[test_index]
                    out_train_features=features[out_train_index]
                    out_train_labels=labels[out_train_index]
                    if is_random:
                        rand_index=np.arange(out_train_features.shape[0])
                        np.random.shuffle(rand_index)
                        out_train_labels=out_train_labels(rand_index)
                    all_train_features=[]
                    all_val_features=[]
                    all_train_labels=[]
                    all_val_labels=[]
                    for j in range(inner_foldn):
                        inner_all_index=np.arange(out_train_features.shape[0])
                        inner_test_index=np.arange(j,out_train_features.shape[0],inner_foldn)
                        inner_train_index=np.delete(inner_all_index,inner_test_index)
                        inner_train_features=out_train_features[inner_train_index]
                        inner_train_labels=out_train_labels[inner_train_index]
                        val_features=out_train_features[inner_test_index]
                        val_labels=out_train_labels[inner_test_index]
                        all_train_features.append(inner_train_features)
                        all_train_labels.append(inner_train_labels)
                        all_val_features.append(val_features)
                        all_val_labels.append(val_labels)
                    yield {'all_train_features':all_train_features, 'all_train_labels':all_train_labels, 'all_val_features':all_val_features,   'all_val_labels':all_val_labels, 'test_features':test_features,'test_labels':test_labels}


def numpy2loader(features,labels,shuffle):
    features=torch.from_numpy(features)
    labels=torch.from_numpy(labels)
    dataset=TensorDataset(features,labels)
    dataloader=DataLoader(dataset,batch_size=features.shape[0],shuffle=shuffle)
    return dataloader

                        






    