import re
from tqdm import tqdm
import torch
import DataBatch
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device1 = torch.device('cpu')

def roll_lower(data, label, min_number):
  newdata = []
  newlabel = []
  for i, sen in enumerate(data):
    if len(sen) > min_number:
      newdata.append(sen)
      newlabel.append(label[i])
  return newdata, newlabel

"""
整合数据集"""
def make_train(data):
    """
    output： data_words, labels
    """
    data_words=[]
    labels=[]
    for i in tqdm(range(1,len(data)), desc='数据整合', ncols=80):
        label_words=re.sub('"',' ',data[i][:310]).strip().split('\t')
        if len(label_words)==2:
            labels.append(label_words[0])
            data_words.append(label_words[-1].split(' '))
    return data_words, labels

"""
整合测试数据集"""
def make_test(testdata):
    """
    output:
    testdata_words
    """
    test_data_words=[]
    for i in tqdm(range(1,len(testdata)), desc='数据整合', ncols=80):
        words=testdata[i][:400].strip()
        test_data_words.append(words.split(' '))
    return test_data_words

def train(optimizer, net, data, labels, batch_size, time):
    i=0
    for input,target in tqdm(DataBatch.get_dataloader(batch_size, data, labels)):
        i+=1
        input = input.to(device)
        target = target.to(device)
        optimizer.zero_grad()
        pre_target = net(input)
        loss = F.nll_loss(pre_target,target)
        loss.backward()
        optimizer.step()
        if i%time == 0:
            print('当前损失值:',loss.item())
  
def do_train(scheduler, optimizer, net, data, labels, batch_size, epochs, timel):  
    for epoch in range(epochs):
        train(optimizer, net, data, labels, batch_size, timel)
        if epoch % 2 == 0:
            scheduler.step()

def test(net, data, labels, batch_size):
    test_loss = 0
    correct = 0
    net.eval()
    dataloader = DataBatch.get_dataloader(batch_size, data, labels)
    with torch.no_grad():
        for input,target in tqdm(dataloader,leave=True,ncols=100,desc='计算进度：'):
            input = input.to(device)
            target = target.to(device)
            output = net(input)
            test_loss += F.nll_loss(output,target,reduction='sum')
            pred = torch.max(output,dim = -1,keepdim = False)[-1]
            correct += pred.eq(target).sum()
        test_loss = test_loss/len(dataloader.dataset)
        correct = correct/len(dataloader.dataset)
        print('平均损失：',test_loss.item(),'\n准确率：',correct.item()*100,'%')