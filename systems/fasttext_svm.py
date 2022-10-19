# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from gensim.models import FastText
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import precision_score
from sklearn import preprocessing
from sklearn.metrics import recall_score
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_validate
import io



# functions for calculating an average vector per tweet
def featureVecMethod(words, model, num_features):
    # Pre-initialising empty numpy array for speed
    featureVec = np.zeros(num_features, dtype="float32")
    nwords = 0
    # append a vector to each word
    for word in words:
        try:
            nwords = nwords + 1
            v = model[word]
        except KeyError:
            continue
        featureVec = featureVec + model[word]

    # dividing the result by number of words to get average
    if nwords != 0:
        featureVec = featureVec / nwords
    return featureVec


def getAvgFeatureVecsOOV(tweet_tokens, model, num_features):
    counter = 0
    tweetFeatureVecs = np.zeros((len(tweet_tokens), num_features), dtype="float32")
    #    all_tweets = len(tweets)
    for i, tweet in enumerate(tweet_tokens):
        tweetFeatureVecs[counter] = featureVecMethod(tweet, model, num_features)
        counter = counter + 1
    return tweetFeatureVecs


def text_to_fastext(list_of_tweets):
    # Loading pretrained word2vec model.
    pretrained_wv_path = "/ikerlariak/jfernandezde010/emnlp/embeddings/cc.it.300.bin"
    model = FastText.load_fasttext_format(pretrained_wv_path)

    # converting strings to tokens
    list_of_tweets_tokens = []
    for line in list_of_tweets:
        tokens = line.split()
        list_of_tweets_tokens.append(tokens)

    # calculating the average vector
    return getAvgFeatureVecsOOV(list_of_tweets_tokens, model, model.vector_size)

RESULTS_DIR='/results_fasttestsvm'
if __name__ == '__main__':

    # load datasets
    train_path = ""
    test_path = ""
    
    tweets_train = pd.read_csv(train_path, dtype={'user_id':'object'}, sep='\t', encoding="utf-8")
    tweets_train = tweets_train.fillna('')

    tweets_test = pd.read_csv(test_path, dtype={'user_id':'object'}, sep='\t', encoding="utf-8")
    tweets_test = tweets_test.fillna('')

    # list of tweets
    tweets_train_text = list(tweets_train.text.values)
    tweets_test_text = list(tweets_test.text.values)

    # labels
    labels_train = list(tweets_train.label.values)
    labels_test = list(tweets_test.label.values)
    #Encode the labes to integers
    le = preprocessing.LabelEncoder()
    le.fit(['AGAINST','FAVOR', 'NEUTRAL'])
    y_train = le.transform(labels_train)
    y_test = le.transform(labels_test)


    #x_train, x_test, y_train, y_test = train_test_split(X, labels_int, test_size=0.1, random_state=0)
    x_train = text_to_fastext(tweets_train_text)
    x_test = text_to_fastext(tweets_test_text)

    #making a numpy array
    x_train = np.array(x_train, dtype=np.float64)
    x_test = np.array(x_test, dtype=np.float64)
    y_train = np.array(y_train, dtype=np.float64)
    y_test = np.array(y_test, dtype=np.float64)


    print("===============================================")
    print("Grid search")

    #gridsearch for SVM model
    svc = svm.SVC(kernel="rbf")

    #define the grids of parameters
    parameters = {'C':[1, 10, 100, 300, 500, 700, 1000], 'gamma': [0.1, 0.001, 0.0001, 0.5, 0.75, 1]}

    svc = svm.SVC(gamma="scale")
    clf = GridSearchCV(svc, parameters, cv=5, n_jobs=5, verbose=True)
    clf.fit(x_train, y_train)

    #saving the result of grid search
    grid_result = pd.DataFrame(clf.cv_results_)
    file_name = RESULTS_DIR+'/grid_results.csv'
    grid_result.to_csv(file_name, encoding='utf-8', index=False)
    #select the best parameters 
    df_grid_first = grid_result.loc[grid_result['rank_test_score'] == 1] 
    C = list(df_grid_first.param_C.values)
    gamma = list(df_grid_first.param_gamma.values)

    print("===============================================")
    print("Cross Validation")

    clf = svm.SVC(kernel="rbf", gamma=gamma[0], C=C[0])

    cv_results = cross_validate(clf, X_train, y_train, cv=5, n_jobs=5, scoring='f1_macro')
    print('CV RESULTS ', cv_results['test_score'])

    y_pred = cross_val_predict(clf, X_train, y_train, cv=5, n_jobs=5)

    target_names = ["AGAINST", "FAVOR", "NEUTRAL"]
    cl_report = classification_report(y_train, y_pred, target_names=target_names, digits=4)
    cm = confusion_matrix(y_train, y_pred)
    print(cm)
    print("CROSS VALIDATION")
    print(cl_report)

    # saving classification reports
    report_df = pd.read_fwf(io.StringIO(cl_report), sep="\s+")
    report_df.to_csv(RESULTS_DIR + '/CV_report.csv', encoding='utf-8', index=False)


    print("================================================")
    print("Training and Testing", "gamma:",gamma[0],"c:",C[0])

    clf_train = svm.SVC(kernel="rbf", gamma=gamma[0], C=C[0], probability=True)
    y_pred_test = clf_train.predict(X_test)  # PREDICTIONS

    f_score_macro = f1_score(y_test, y_pred_test, average='macro')
    print('F1 macro ', f_score_macro)
    f_score_micro = f1_score(y_test, y_pred_test, average='micro')
    print('F1 micro ', f_score_micro)
    precision = precision_score(y_test, y_pred_test, average='macro')
    print('PRECISION ', precision)
    recall = recall_score(y_test, y_pred_test, average='macro')
    print('RECALL ', recall)
    cm = confusion_matrix(y_test, y_pred_test)
    print('CONFUSION MATRIX')
    print(cm)
    target_names = ["AGAINST", "FAVOR", "NEUTRAL"]

    # save the classification reports
    cl_report_test = classification_report(y_test, y_pred_test, target_names=target_names, digits=4)
    report_df_test = pd.read_fwf(io.StringIO(cl_report_test), sep="\s+")
    print('Classification report ')
    print(report_df_test)
    file_name = RESULTS_DIR + '/test_report.csv'
    report_df_test.to_csv(file_name, encoding='utf-8', sep='\t', index=False)

    # save the predicted data for user class distances classification
    tweets_test['predicted'] = list(map({0: 'AGAINST', 1: 'FAVOR', 2: 'NEUTRAL'}.get, y_pred_test))
    tweets_test['true'] = y_test
    tweets_test.to_csv(RESULTS_DIR + '/fasttext_predicted.csv', sep='\t', encoding='utf-8', index=False)


