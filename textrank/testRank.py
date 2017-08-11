#-*- encoding:utf-8 -*-
import jieba
import sys
reload(sys)
sys.setdefaultencoding('utf-8')

import codecs
from textrank4zh import TextRank4Keyword
text = codecs.open('./text/05.txt', 'r', 'utf-8').read()
tr4w = TextRank4Keyword(stop_words_file='./stopword.txt')  # 导入停止词

#使用词性过滤，文本小写，窗口为2
tr4w.train(text=text, speech_tag_filter=False, lower=True, window=2)

print '关键词：'
# 20个关键词且每个的长度最小为1
print '/'.join(tr4w.get_keywords(10, word_min_len=2))




