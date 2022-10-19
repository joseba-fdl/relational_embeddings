

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
from tqdm import tqdm
from sklearn import metrics


## Functtion to get class distances for a specific user ##
def taldeekiko_distantzia(erabiltzaile_bat, klaseka, relational_embedding):
    klaseka_dist = {}
    for klase in klaseka:
        dist_tot = 0
        erab_tot = 0
        for erabiltzaile in klaseka[klase]:
            try:
                if erabiltzaile_bat != erabiltzaile:
                    dist_tot += relational_embedding.similarity(erabiltzaile_bat, erabiltzaile)
                    erab_tot += 1
            except:
                continue
        if erab_tot != 0:
            klaseka_dist.update({klase: dist_tot / erab_tot})
        else:
            return "not-found"
    klaseka_dist_ordenatua = dict(sorted(klaseka_dist.items(), reverse=True, key=lambda x: float(x[1])))
    return klaseka_dist_ordenatua


if __name__ == '__main__':

    # PATHS and FILE NAMES #
    train_path = ""
    test_path = ""
    texual_predictions_path = ""

    #### TRAIN ####
    train_tweets=pd.read_csv(train_path, sep='\t', dtype={'user_id':'object'})
    df_train = train_tweets[['user_id', 'label']]

    #### TEST ####
    test_tweets=pd.read_csv(test_path, sep='\t', dtype={'user_id':'object'})
    df_test = test_tweets[['user_id', 'label']]

    ### predictions from textual classifiers ###
    df_predictions_textual_tweets=pd.read_csv(texual_predictions_path, sep='\t', dtype={'user_id':'object'})
    predictions_textual = df_predictions_textual_tweets['predicted'].tolist()


    #### TRAINERAKO ####
    train_users = df_train['user_id'].tolist()
    train_klase = df_train['label'].tolist()

    # traineko erabiltzaileak klaseka gorde #
    klaseka={}
    for klase,erabiltzaile in zip(train_klase,train_users):
        if klase not in klaseka:
            klaseka.update({klase: []})
        klaseka[klase].append(erabiltzaile)

    ### TEST ###
    test_users = df_test['user_id'].tolist()
    test_klase = df_test['label'].tolist()


    #### EMBEDDING ####
    relational_embedding = KeyedVectors.load("embedding_path and name")

    #### EBALUATION USER LEVEL #### TEST
    predict=[]
    test=[]
    for klase, erab, prediction_txt in tqdm(zip(test_klase, test_users, predictions_textual)):
        iragarpena_balioekin = taldeekiko_distantzia(erab, klaseka, relational_embedding)
        # ez-aurkituak ez konparatzeko #
        if iragarpena_balioekin != "not-found": # user found in the RelEmb
            prediction_ctxt = list(iragarpena_balioekin.keys())[0]
            test.append(klase) ## test
            predict.append(prediction_ctxt) ## predict
        else:  # when the user is not found in the RelEmb
            test.append(klase) ## test gold ##
            predict.append(prediction_txt) ## predict ## when the user is not found in the RelEmb: add prediction made by textuak classifiers
            #predict.append('NEUTRAL') # predict ## add neutral

    print(metrics.f1_score(test, predict, average=None))
    print(metrics.classification_report(test, predict))









