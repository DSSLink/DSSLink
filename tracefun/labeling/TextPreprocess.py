import os
import re
from nltk.stem import SnowballStemmer

current_dir = os.path.dirname(os.path.abspath(__file__))


class TextPreprocess:

    def __init__(self):
        self.stemmer = SnowballStemmer("english")
        # self.stopwords = stopwords.words('english')
        self.load_stopwords()

    def load_stopwords(self):
        with open(os.path.join(current_dir, 'stopwords.txt'), 'r', encoding='utf8') as f:
            stopwords = f.readlines()
        self.stopwords = [w.strip() for w in stopwords]

    def run(self, text):
        text = self.chararctorClean(text)
        # text = self.splitCamelCase(text)
        text = self.tolowerCase(text)
        # text = self.stemming(text)
        text = self.stopwords_remover(text)
        # text = self.lengthFilter(text, 2)
        return text

    def splitCamelCase(self, text):
        '''驼峰分割'''
        sb = list()
        words = text.split(" ")
        for word in words:
            splitted = re.sub('([A-Z][a-z]+)', r' \1', re.sub('([A-Z]+)', r' \1', word)).split()
            for w in splitted:
                sb.append(w)
        return " ".join(sb)

    def chararctorClean(self, text):
        '''保留字母和数字，删除标点符号'''
        sb = list()
        for m in re.findall("[a-zA-Z0-9]+", text):
            sb.append(m)
        return " ".join(sb)

    def lengthFilter(self, text, length):
        sb = list()
        words = text.split(" ")
        for word in words:
            if len(word) >= length:
                sb.append(word)
        return " ".join(sb)

    def tolowerCase(self, text):
        return text.lower()

    def stemming(self, text):
        '''词根还原'''
        sb = list()
        words = text.split(" ")
        for word in words:
            # word = self.stemmer.stem(word)
            word = self.stemming_word(word)
            sb.append(word)
        return " ".join(sb)

    def stemming_word(self, word):
        last_w = word
        while 1:
            next_w = self.stemmer.stem(last_w)
            if next_w == last_w:
                break
            last_w = next_w
        return next_w

    def stopwords_remover(self, text):
        '''取出停用词'''
        sb = list()
        words = text.split(" ")
        for word in words:
            if word not in self.stopwords:
                sb.append(word)
        return " ".join(sb)

preprocessor = TextPreprocess()

if __name__ == '__main__':
    text = 'cultural culturale'
    new_text = preprocessor.run(text)
    print(f'text: {text}')
    print(f'new_text: {new_text}')


