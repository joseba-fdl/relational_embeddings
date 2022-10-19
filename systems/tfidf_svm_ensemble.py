# -*- coding: utf-8 -*-

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest
import pandas as pd
from gensim.models import KeyedVectors
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import preprocessing
from sklearn.metrics import recall_score
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_validate
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import precision_score


fitx_embedding='relational_embedding_path' # RTs CIC corpus users + expanded users
embedding_vectors= KeyedVectors.load(fitx_embedding)
num_features_contextual=embedding_vectors.vector_size

def user_vector (users_list):
    featureVec_ctxt_list=[]
    for user in users_list:
        featureVec_ctxt=np.zeros(num_features_contextual,dtype="float32") # zeroen bektorea
        try: featureVec_ctxt = featureVec_ctxt + embedding_vectors[user] # erabiltzailea embedinean badago, GEHITU
        except: continue
        featureVec_ctxt_list.append(list(featureVec_ctxt))
    return featureVec_ctxt_list


if __name__ == '__main__':
    # load datasets
    train_path = ""
    test_path = ""

    tweets_train = pd.read_csv(train_path, dtype={'user_id': 'object'}, sep='\t', encoding="utf-8")
    tweets_train = tweets_train.fillna('')

    tweets_test = pd.read_csv(test_path, dtype={'user_id': 'object'}, sep='\t', encoding="utf-8")
    tweets_test = tweets_test.fillna('')

    # list of tweets
    tweets_train_text = list(tweets_train.text.values)
    tweets_test_text = list(tweets_test.text.values)

    ### list of users ###
    tweets_train_user = list(tweets_train.user_id.values)
    vectors_train_user = user_vector(tweets_train_user)
    tweets_test_user = list(tweets_test.user_id.values)
    vectors_test_user = user_vector(tweets_test_user)

    ### list of labels ###
    labels_train = list(tweets_train.label.values)
    labels_test = list(tweets_test.label.values)
    
    vectorizer = TfidfVectorizer()
    df_train_textua=pd.DataFrame(vectorizer.fit_transform(tweets_train_text).toarray())
    df_test_textua=pd.DataFrame(vectorizer.transform(tweets_test_text).toarray())
    df_train_erabs=pd.DataFrame(vectors_train_user)
    df_test_erabs=pd.DataFrame(vectors_test_user)

    X_train = pd.concat([df_train_textua,df_train_erabs], axis=1)
    X_test = pd.concat([df_test_textua,df_test_erabs], axis=1)

    print('X TRAIN ', X_train.shape)
    print('X TEST ', X_test.shape)

    # Encode the labes to integers
    le = preprocessing.LabelEncoder()
    le.fit(['AGAINST', 'FAVOR', 'NEUTRAL'])
    y_train = le.transform(labels_train)
    print('Y TRAIN ', y_train.shape)
    y_test = le.transform(labels_test)
    print('Y TEST', y_test.shape)

    print('=====================================================')
    print('grid search')

    pipeline1 = Pipeline(
        [("filter", SelectKBest(mutual_info_classif, k='all')),
         ("classification", svm.SVC(kernel="rbf"))])

    grid_parameters_tune = [{"classification__C": [1, 10, 100, 300, 500, 700, 1000],
                             'classification__gamma': [0.1, 0.01, 0.001, 0.25, 0.5, 0.75, 1]}]

    model = GridSearchCV(pipeline1, grid_parameters_tune, cv=5, n_jobs=5, verbose=True)
    model.fit(X_train, y_train)

    grid_result = pd.DataFrame(model.cv_results_)
    grid_best = pd.DataFrame(model.best_params_, index=[0])
    # select the best parameters
    df_grid_first = grid_result.loc[grid_result['rank_test_score'] == 1]
    C = list(df_grid_first.param_classification__C.values)
    gamma = list(df_grid_first.param_classification__gamma.values)
    print('Parameters:\n', 'C: ', C, 'gamma: ', gamma)

    print("===============================================")
    print("Cross Validation")

    clf = Pipeline(
        [("filter", SelectKBest(mutual_info_classif, k='all')),
         ("classification", svm.SVC(kernel="rbf", gamma=gamma[0], C=C[0]))])

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
    train_path_predictions = test_path.split('/')[-1]  # izenerako
    import io

    report_df = pd.read_fwf(io.StringIO(cl_report), sep="\s+")
    # saving predicted and wrong predicted examples
    tweets_train['predicted'] = y_pred
    tweets_train['true'] = y_train

    print("================================================")
    print("Training and Testing")

    clf_train = Pipeline([("filter", SelectKBest(mutual_info_classif, k='all')),
                          ("classification", svm.SVC(kernel="rbf", gamma=gamma[0], C=C[0], probability=True)), ])

    clf_train.fit(X_train, y_train)
    y_pred_test = clf_train.predict(X_test)  # PREDICTIONS
    y_pred_test_probabilities = clf_train.predict_proba(X_test)  # PREDICTIONS WITH PROBABILITIES

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

    test_path_predictions = test_path.split('/')[-1]
    # save the classification reports
    cl_report_test = classification_report(y_test, y_pred_test, target_names=target_names, digits=4)
    report_df_test = pd.read_fwf(io.StringIO(cl_report_test), sep="\s+")
    print('Classification report ')
    print(report_df_test)

