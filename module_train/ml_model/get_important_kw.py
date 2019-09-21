from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline

import seaborn
import matplotlib.pyplot as plt
import pandas as pd


def get_feature_importance(train_data, target, path_save_important_kw, n_feature_important=50):
    pipeline = Pipeline([
        ('vect', CountVectorizer(ngram_range=(1, 2))),
        ('tfidf', TfidfTransformer()),
        ('clf', RandomForestClassifier())
    ])
    pipeline.set_params(clf__n_estimators=10).fit(train_data, target)

    vocab = pipeline.named_steps['vect'].get_feature_names()

    feature_important = pipeline.named_steps['clf'].feature_importances_
    print(len(feature_important))
    plot_feature_importance(feature_important, vocab, path_save_important_kw, n_feature_important)


def plot_feature_importance(feature_important, list_vocabulary, path_save_important_kw,
                            n_feature_show=50):
    # list vocabulary: ['an_ninh', 'an_toàn', 'apec', 'ban'...]
    # list_feature_importance: [0, 0.124, 0, 0]
    index_sorted_important_ft = sorted(range(len(feature_important)), key=feature_important.__getitem__, reverse=True)
    list_best_important_ft = index_sorted_important_ft[0: n_feature_show]
    # get list probability of best ft:
    # now get list mapping vocabulary with number count
    list_best_ft_vocabulary = []
    list_best_ft_probability = []
    for e_index in list_best_important_ft:

        list_best_ft_vocabulary.append(list_vocabulary.__getitem__(e_index))
        list_best_ft_probability.append(feature_important.__getitem__(e_index))

    name_proba_ft_important = list(zip(list_best_ft_vocabulary, list_best_ft_probability))
    df_name_proba = pd.DataFrame(name_proba_ft_important, columns=['name_ft', 'probability'])
    df_name_proba.to_csv(path_save_important_kw)

    seaborn.scatterplot(x="name_ft", y="probability", data=df_name_proba)
    plt.show()
