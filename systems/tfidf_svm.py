# -*- coding: utf-8 -*-

import io
import pandas as pd 
from sklearn import svm
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import precision_score
from sklearn import preprocessing
from sklearn.metrics import recall_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_validate
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest

def words_to_tfidf (train_text, test_text):
    vectorizer = TfidfVectorizer()
    X_train_text = vectorizer.fit_transform(train_text)
    X_test_text = vectorizer.transform(test_text)
    return X_train_text, X_test_text


RESULTS_DIR='/results_tfidfsvm'
if __name__ == '__main__':

    #load datasets
    train_path = ""
    test_path = ""

    tweets_train = pd.read_csv(train_path, dtype={'user_id':'object'}, sep='\t', encoding="utf-8")
    tweets_train = tweets_train.fillna('')

    tweets_test = pd.read_csv(test_path, dtype={'user_id':'object'}, sep='\t', encoding="utf-8")
    tweets_test = tweets_test.fillna('')

    #list of tweets
    tweets_train_text = list(tweets_train.text.values)
    tweets_test_text = list(tweets_test.text.values)

    labels_train = list(tweets_train.label.values)
    labels_test = list(tweets_test.label.values)

    # tfidf vectorizing
    X_train, X_test = words_to_tfidf (tweets_train_text, tweets_test_text)
    print('X TRAIN ', X_train.shape)
    print('X TEST ', X_test.shape)

    le = preprocessing.LabelEncoder()
    le.fit(['AGAINST','FAVOR', 'NEUTRAL'])
    y_train = le.transform(labels_train)
    y_test = le.transform(labels_test)
    print('Y TRAIN ', y_train.shape)
    print('Y TEST', y_test.shape)

    print('=====================================================')
    print('grid search')

    num_features ='all' #12000
    pipeline1 = Pipeline(
        [("filter", SelectKBest(mutual_info_classif, k=num_features)),
        ("classification", svm.SVC(kernel="rbf"))])

    grid_parameters_tune = [{"classification__C": [1, 10, 100, 300, 500, 700, 1000], 
                             'classification__gamma': [0.1, 0.01, 0.001, 0.25, 0.5, 0.75, 1]}]
    model = GridSearchCV(pipeline1, grid_parameters_tune, cv=5, n_jobs=5, verbose=True)
    model.fit(X_train, y_train)

    grid_result = pd.DataFrame(model.cv_results_)
    grid_best = pd.DataFrame(model.best_params_, index=[0])
    file_name = RESULTS_DIR+'/grid_result.csv'
    grid_result.to_csv(file_name, encoding='utf-8', index=False)
    #select the best parameters 
    df_grid_first = grid_result.loc[grid_result['rank_test_score'] == 1]
    C = list(df_grid_first.param_classification__C.values)
    gamma = list(df_grid_first.param_classification__gamma.values)

    print('Parameters:\n', 'C: ', C,'gamma: ', gamma)
    df_grid_first.to_csv(RESULTS_DIR+'/best_param.csv', encoding='utf-8', index=False)

    print("===============================================")
    print("Cross Validation")

    clf = Pipeline(
        [("filter", SelectKBest(mutual_info_classif, k=num_features)),
         ("classification", svm.SVC(kernel="rbf", gamma=gamma[0], C=C[0]))])

    cv_results = cross_validate(clf, X_train, y_train, cv=5, n_jobs=5, scoring='f1_macro')
    print('CV RESULTS ', cv_results['test_score'])

    y_pred = cross_val_predict(clf, X_train, y_train, cv=5, n_jobs=5)

    target_names=["AGAINST", "FAVOR", "NEUTRAL"]
    cl_report = classification_report(y_train, y_pred, target_names=target_names, digits=4)
    cm = confusion_matrix(y_train, y_pred)
    print(cm)
    print("CROSS VALIDATION")
    print(cl_report)

    #saving classification reports
    report_df = pd.read_fwf(io.StringIO(cl_report), sep="\s+")
    report_df.to_csv(RESULTS_DIR+'/CV_report.csv', encoding='utf-8', index=False)

    print("================================================")
    print("Training and Testing")

    clf_train = Pipeline(
        [("filter", SelectKBest(mutual_info_classif, k=num_features)),
        ("classification", svm.SVC(kernel="rbf", gamma=gamma[0], C=C[0], probability=True)),])

    clf_train.fit(X_train, y_train)
    y_pred_test = clf_train.predict(X_test) # PREDICTIONS

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
    target_names=["AGAINST", "FAVOR", "NEUTRAL"]

    # save the classification reports
    cl_report_test = classification_report(y_test, y_pred_test, target_names=target_names, digits=4)
    report_df_test = pd.read_fwf(io.StringIO(cl_report_test), sep="\s+")
    print('Classification report ')
    print(report_df_test)
    file_name = RESULTS_DIR+'/test_report.csv'
    report_df_test.to_csv(file_name, encoding='utf-8', sep='\t', index=False)

    # save the predicted data for user class distances classification
    tweets_test['predicted'] = list( map({0: 'AGAINST', 1: 'FAVOR', 2: 'NEUTRAL'}.get, y_pred_test))
    tweets_test['true'] = y_test
    tweets_test.to_csv(RESULTS_DIR + '/tfidfsvm_predicted.csv', sep='\t', encoding='utf-8', index=False)