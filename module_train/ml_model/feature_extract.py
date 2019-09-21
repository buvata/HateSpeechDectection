import re
import emoji
import numpy as np
import pandas as pd
import scipy.sparse as sp


from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline, FeatureUnion


class Lowercase(BaseEstimator, TransformerMixin):
    def transform(self, x):
        return [s.lower() for s in x]

    def fit(self, x, y=None):
        return self


class NumUpperLetterFeature(BaseEstimator, TransformerMixin):
    def count_upper(self, s):
        n_uppers = sum(1 for c in s if c.isupper())
        return n_uppers / (1 + len(s))

    def transform(self, x):
        counts = [self.count_upper(s) for s in x]
        return sp.csr_matrix(counts, dtype=np.float64).transpose()

    def fit(self, x, y=None):
        return self


class NumUpperWordFeature(BaseEstimator, TransformerMixin):
    def count_upper(self,s):
        n_uppers = sum(1 for w in s.split() if w[0].isupper())
        return n_uppers / (1 + len(s.split()))

    def transform(self, x):
        counts = [self.count_upper(s) for s in x]
        return sp.csr_matrix(counts, dtype=np.float64).transpose()

    def fit(self, x, y=None):
        return self


class NumLowerLetterFeature(BaseEstimator, TransformerMixin):
    def count_upper(self, s):
        n_uppers = sum(1 for c in s if c.islower())
        return n_uppers / (1 + len(s))

    def transform(self, x):
        counts = [self.count_upper(s) for s in x]
        return sp.csr_matrix(counts, dtype=np.float64).transpose()

    def fit(self, x, y=None):
        return self


class NumWordsCharsFeature(BaseEstimator, TransformerMixin):
    def count_char(self, s):
        return len(s)

    def count_word(self, s):
        return len(s.split())

    def transform(self, x):
        count_chars = sp.csr_matrix([self.count_char(s) for s in x], dtype=np.float64).transpose()
        count_words = sp.csr_matrix([self.count_word(s) for s in x], dtype=np.float64).transpose()

        return sp.hstack([count_chars, count_words])

    def fit(self, x, y=None):
        return self


class NumEmojiFeature(BaseEstimator, TransformerMixin):
    def count_emoji(self, s):
        emoji_list = []
        for c in s:
            if c in emoji.UNICODE_EMOJI:
                emoji_list.append(c)
        return len(emoji_list) / (1 + len(s.split()))

    def transform(self, x):
        counts = [self.count_emoji(s) for s in x]
        return sp.csr_matrix(counts, dtype=np.float64).transpose()

    def fit(self, x, y=None):
        return self


def extract_feature():
    clf = Pipeline([
            ('features', FeatureUnion([
                ('custom_features_pipeline', Pipeline([
                    ('custom_features', FeatureUnion([
                        ('f01', NumWordsCharsFeature()),
                        ('f02', NumUpperLetterFeature()),
                        ('f03', NumUpperWordFeature()),
                        ('f04', NumEmojiFeature())
                    ], n_jobs=-1)),
                    ('scaler', StandardScaler(with_mean=False))
                ])),
                ('word_char_features_pipeline', Pipeline([
                    ('lowercase', Lowercase()),
                    ('word_char_features', FeatureUnion([
                        ('with_tone', Pipeline([
                            ('tf_idf_word', TfidfVectorizer(ngram_range=(1, 4), norm='l2', min_df=2))
                        ])),
                        ('with_tone_char', TfidfVectorizer(ngram_range=(1, 5), norm='l2', min_df=2, analyzer='char')),
                        ('with_tone_char_wb', TfidfVectorizer(ngram_range=(1, 5), norm='l2', min_df=2, analyzer='char_wb')),
                    ], n_jobs=-1))
                ]))
            ], n_jobs=-1)),
        ])
    return clf


if __name__ == '__main__':
    text = ['sp khong tot', 'dung rat duoc','giao hang nhanh', 'sp dung rat chan']
    y = [0, 0, 0, 1]
    test = ['dung chan']
    clf = extract_feature()
    vec = clf.fit_transform(text)
    # clf.transform(test)
    SVM = SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
    SVM.fit(vec, y)
