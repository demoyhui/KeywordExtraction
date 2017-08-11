#coding:utf-8
import numpy as np
import lda
#import lda.datasets
import jieba
import codecs
import sys
reload(sys)
sys.setdefaultencoding('utf-8')

class LDA_Demo():
    def __init__(self, topics=2):
        self.n_topic = topics
        self.corpus = None
        self.vocab = None
        self.ppCountMatrix = None
        self.stop_words = None
        self.model = None

    def loadCorpusFromFile(self, fn):
        # 中文分词
        f = open(fn, 'r')
        text = f.readlines()
        text = r' '.join(text)

        seg_generator = jieba.cut(text,cut_all=False)
        seg_list = [i for i in seg_generator if i not in self.stop_words]
        seg_list = r' '.join(seg_list)
        # 切割统计所有出现的词纳入词典
        print seg_list
        seglist = seg_list.split(" ")
        self.vocab = []
        for word in seglist:
            if (word != u' ' and word not in self.vocab):
                self.vocab.append(word)

        CountMatrix = []
        f.seek(0, 0)
        # 统计每个文档中出现的词频
        for line in f:
            # 置零
            count = np.zeros(len(self.vocab),dtype=np.int)
            text = line.strip()
            # 但还是要先分词
            seg_generator = jieba.cut(text)
            seg_list = [i for i in seg_generator if i not in self.stop_words]
            seg_list = r' '.join(seg_list)
            seglist = seg_list.split(" ")
            # 查询词典中的词出现的词频
            for word in seglist:
                if word in self.vocab:
                    count[self.vocab.index(word)] += 1
            CountMatrix.append(count)
        f.close()
        #self.ppCountMatrix = (len(CountMatrix), len(self.vocab))
        self.ppCountMatrix = np.array(CountMatrix)
        #print self.ppCountMatrix

        print "load corpus from %s success!"%fn

    def setStopWords(self, word_list):
        self.stop_words = word_list

    def fitModel(self, n_iter = 1500, _alpha = 0.1, _eta = 0.01):
        self.model = lda.LDA(n_topics=self.n_topic, n_iter=n_iter, alpha=_alpha, eta= _eta, random_state= 1)
        self.model.fit(self.ppCountMatrix)

    def printTopic_Word(self, n_top_word = 8):
        for i, topic_dist in enumerate(self.model.topic_word_):
            topic_words = np.array(self.vocab)[np.argsort(topic_dist)][:-(n_top_word + 1):-1]

            print "Topic:",i,"\t",
            for word in topic_words:
                print word,
            print

    def printDoc_Topic(self):
        for i in range(len(self.ppCountMatrix)):
            print ("Doc %d:((top topic:%s) topic distribution:%s)"%(i, self.model.doc_topic_[i].argmax(),self.model.doc_topic_[i]))

    def printVocabulary(self):
        print "vocabulary:"
        for word in self.vocab:
            print word,
        print

    def saveVocabulary(self, fn):
        f = codecs.open(fn, 'w', 'utf-8')
        for word in self.vocab:
            f.write("%s\n"%word)
        f.close()

    def saveTopic_Words(self, fn, n_top_word = -1):
        if n_top_word==-1:
            n_top_word = len(self.vocab)
        f = codecs.open(fn, 'w', 'utf-8')
        for i, topic_dist in enumerate(self.model.topic_word_):
            topic_words = np.array(self.vocab)[np.argsort(topic_dist)][:-(n_top_word + 1):-1]
            f.write( "Topic:%d\t"%i)
            for word in topic_words:
                f.write("%s "%word)
            f.write("\n")
        f.close()

    def saveDoc_Topic(self, fn):
        f = codecs.open(fn, 'w', 'utf-8')
        for i in range(len(self.ppCountMatrix)):
            f.write("Doc %d:((top topic:%s) topic distribution:%s)\n" % (i, self.model.doc_topic_[i].argmax(), self.model.doc_topic_[i]))
        f.close()


if __name__ == '__main__':
    # word_all = []
    # with open('E:\mining\SMPCUP2017TASK1DATASET\\SMPCUP2017_TrainingData_Task1.TXT') as lines:
    #     for line in lines:
    #         for i in range(1,6):
    #             word = line.split('\001')[i]
    #             if i == 5:
    #                 word = word.strip('\n')
    #             if len(word) > 6 and word not in word_all:
    #                 word_all.append(word)
    # for word in word_all:
    #     jieba.add_word(word)
    stop_word=[]
    with open('./stopword.txt','r') as files:
        for file in files:
            file = file.strip('\r\n')
            stop_word.append(file)
        print stop_word
    _lda = LDA_Demo(topics=3)
    _lda.setStopWords(stop_word)
    _lda.loadCorpusFromFile(r'./text/05.txt')
    _lda.fitModel()
    _lda.printTopic_Word(n_top_word=10)
    _lda.printDoc_Topic()
    _lda.printVocabulary()




