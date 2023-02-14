
"""获得字典"""
class Vocabulary():
    def __init__(self, sentences, min_number=3, max_number=None):
        """
        sentences: 全部的文本数据，可以是一个列表形式
        min_number: 字典中保留字的最小词频
        max_number: 字典中保留字的最大词频
        UKN: 未见过的词
        PAD: 填充词
        """
        self.sentences = sentences
        self.min_number = min_number
        self.max_number = max_number
        self.UKN = 'UKN'
        self.PAD = 'PAD'
        self.dict = {}
        self.voc = {'PAD':0, 'UKN':1}
        self.get_voc()
        
    def get_word_frequency(self):
        
        for sentence in self.sentences:
            for word in sentence:
                if word not in self.dict:
                    self.dict[word] = 1
                else:
                    self.dict[word]+=1
        
        if self.min_number is not None:
            self.dict= {word:value for word,value in self.dict.items() if value > self.min_number}
            
        if self.max_number is not None:
            self.dict = {word: value for word, value in self.dict.items() if value < self.max_number}
        return sorted(self.dict.items(), key = lambda x: x[1], reverse = False)
    
    def get_voc(self):
        wordnumber = self.get_word_frequency()
        for word, _ in wordnumber:
            self.voc[word] = len(self.voc)
        self.inverse = {wm[1]:wm[0] for wm in self.voc.items()}
        del self.dict
        del wordnumber

    def transform(self, sentence, max_len = None):
         
        """
        把句子转化为序列
        """
        if max_len is not None:
            if max_len > len(sentence):
                sentence = sentence + [self.PAD]*(max_len-len(sentence))
            if max_len < len(sentence):
                sentence = sentence[:max_len]
        for idex, word in  enumerate(sentence):
            if word in self.voc:
                sentence[idex] = self.voc[word]
            else:
                sentence[idex] = self.voc['UKN'] 
        return sentence
    def inverse_transform(self, indices):
        """
        把序列转化为句子
        """
        return [self.inverse.get(idx) for idx in indices]