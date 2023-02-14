import requests
from tqdm import tqdm
import os
import warnings


warnings.filterwarnings('ignore')


def download_file(url, folder):
    print("------","Start download with urllib")
    name=url.split("/")[-1]
    resp = requests.get(url,stream=True)
    content_size = int(resp.headers['Content-Length']) / 1024  # 确定整个安装包的大小
    #下载到上一级目录
    #path = os.path.abspath(os.path.dirname(os.getcwd())) + "/news_recommend/" + name
    #下载到该目录
    path = os.getcwd()+ '/'+str(folder)+'/' + name
    print("File path:  ",path)
    with open(path, "wb") as file:
        print("File total size is:  ", content_size)
        for data in tqdm(iterable=resp.iter_content(1024), total=content_size, unit='k', desc=name):
            file.write(data)
    print("------","finish download with urllib\n\n")


#加载news_NLP的数据集
import pandas as pd
newsNLP_url_list=pd.read_csv('./data/NLP_data_list.csv', encoding='gbk')
if __name__ == '__main__':
    for i in range(3):
        url = newsNLP_url_list['link'][i]
        download_file(url, 'news_NLP')

