import os

import analysis
import numpy as np
import gensim
from gensim import corpora, models, matutils
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

#ファイルの読み込み、及びタイトル・本文データの取得
path_list = ["dokujo-tsushin", "it-life-hack", "kaden-channel"]
w_list = []
labels = []
for p_list in path_list:
    path = "./text/"+p_list
    f_list = os.listdir(path)
    for lists in f_list:
        with open("./text/"+ p_list+ "/"+lists, encoding="utf-8_sig") as f:
            next(f)
            next(f)
            w = f.read().replace('\u3000','').replace('\n','')
            w_list.append(w)
            labels.append(path_list.index(p_list))


#辞書作成
words = analysis.get_words(w_list)
dictionary = corpora.Dictionary(words)
dictionary.filter_extremes(no_below = 200, no_above = 0.4)
#dictionary.save_as_text("./tmp/bow_test.txt")
#courpus = [dictionary.doc2bow(word) for word in words]


#コーパスを特徴ベクトルの変換
def vec2dense(vec, num_terms):
    return list(matutils.corpus2dense([vec], num_terms=num_terms).T[0])
data_all = [vec2dense(dictionary.doc2bow(words[i]),len(dictionary)) for i in range(len(words))]


#--------------------------------------------------------------------------

#トレーニング・テストデータ設定
train_data = data_all
X_train, X_test, y_train, y_test = train_test_split(train_data, labels, test_size=0.4, random_state=1)

#データ標準化
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

#学習モデル作成
clf = SVC(C = 2, kernel = 'rbf')
clf.fit(X_train_std, y_train)
#joblib.dump(clf, './tmp/test.cmp', compress = True)



#正解率
score = clf.score(X_test_std, y_test)
print("{:.3g}".format(score))


#未知予測
# pathes = os.listdir("./text/test/")
# for path in pathes:
#     test_list = []
#     test_doc = ""
#     with open("./text/test/"+path, encoding="utf-8_sig") as f1:
#         next(f1)
#         next(f1)
#         test_doc = f1.read()
#         test_list.append(test_doc)
#
#     test_words = analysis.get_words(test_list)
#     test_dense = [vec2dense(dictionary.doc2bow(test_words[i]),len(dictionary)) for i in range(len(test_words))]
#
#     #未知のデータを予測
#     predicted0 = clf.predict(test_dense)
#     print(path_list[int(predicted0)])
