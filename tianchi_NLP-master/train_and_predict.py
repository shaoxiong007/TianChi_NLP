import torch
import model
from torch import optim
import Generate_train_test_data
import Vocabulary
import Generate_train_test_data
import pandas as pd
import DataBatch
from tqdm import tqdm
import numpy as np

trainfile='./train_set.csv'
data=open(trainfile).readlines()
testfile='./test_a.csv'
testdata=open(testfile).readlines()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device1 = torch.device('cpu')

data_words, labels = Generate_train_test_data.make_train(data)
new_words, new_labels = Generate_train_test_data.roll_lower(data_words, labels, 5)
"""划分训练集和测试集"""
train_words, test_words, train_labels, test_labels = Generate_train_test_data.train_test_split(new_words,
                                                                             new_labels, random_state=6, test_size=0.2)
test_data_words = Generate_train_test_data.make_test(testdata)

model = model.mymodel(d_model=40, d_k=30, d_v=40, n_heads=8, d_ff=50, n_layers=1).to(device)
optimizer = optim.AdamW(model.parameters(), lr = 0.002)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.2)

Generate_train_test_data.do_train(scheduler, optimizer, model, train_words, train_labels, 200, 16, 80)

label_all = []
dataloader = DataBatch.get_dataloader(200, data = test_data_words, shuffle = False)
for data in tqdm(dataloader):
    data = data.to(device)
    prob = model(data) #表示模型的预测输出
    prob = prob.to(device1)
    prob = prob.detach().numpy() #转成numpy
    label_all.extend(np.argmax(prob,axis=1)) #求每一行的最大值索引

#保存数据
name = ['label']
test = pd.DataFrame(columns = name, data = label_all)
test.to_csv('./test_label.csv', index = False)