import torch
from torch.utils.data import DataLoader, Dataset
import Vocabulary
import Generate_train_test_data

file='train_set.csv'
data=open(file).readlines()
file='test_a.csv'
testdata=open(file).readlines()

data_words, labels = Generate_train_test_data.make_train(data)
new_words, new_labels = Generate_train_test_data.roll_lower(data_words, labels, 5)
"""划分训练集和测试集"""
train_words, test_words, train_labels, test_labels = Generate_train_test_data.train_test_split(new_words,
                                                                             new_labels, random_state=6, test_size=0.2)
test_data_words = Generate_train_test_data.make_test(testdata)

vocab = Vocabulary.Vocabulary(data_words+test_data_words, min_number=7)

#数据加载和提取
class GetDataset(Dataset):
    def __init__(self, data = None, label = None):
        self.data  = data
        self.label  = label

    def __getitem__(self, index):
        if self.label:
            need_data = self.data[index]
            need_label = int(self.label[index])
            return need_data, need_label
        else:
            need_data = self.data[index]
            return need_data
    
    def __len__(self):
        return len(self.data)

#数据转化
def collate_fn(batch):
    """batch 一个数据，带标签或者不带。
    """
    if len(batch[0])==2:
        content, label = list(zip(*batch))
        digitcontent = []
        for sentence in content:
            digitcontent.append(vocab.transform(sentence, 200))
        content = torch.tensor(digitcontent)
        label = torch.torch.LongTensor(label)
        return content, label
    else:
        content = tuple(batch)
        digitcontent = []
        for sentence in content:
            digitcontent.append(vocab.transform(sentence, 200))
        content = torch.tensor(digitcontent)
        return content

def get_dataloader(batch_size, data = None, label = None, shuffle = True):
    """batch: 一次批量大小; data: 数据; label: 标签; shuffle: 是否打乱。
    """
    dataset = GetDataset(data, label)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn = collate_fn)
    return data_loader