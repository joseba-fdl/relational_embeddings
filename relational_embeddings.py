
import pandas as pd
import os
from gensim.models import Word2Vec
from gensim.models import KeyedVectors


def training_relational_embeddings (file_name, emb_dimensions, pairs):
    file_embedding = file_name+"."+str(emb_dimensions)
    file_embedding_exists = os.path.isfile(file_embedding)
    if not file_embedding_exists:
        # train model
        model_interactions = Word2Vec(sentences=pairs, size=emb_dimensions, window=2, min_count=1, workers=1, sg=0)
        # save model
        interaction_vectors = model_interactions.wv
        interaction_vectors.save(file_embedding)

DIMENTSIOAK = [10,20,50]

if __name__ == "__main__":
    #### RETWEET DATASETAK IRAKURRI ####
    df_retweets = pd.read_csv("retweets.csv", dtype={"Source": "object", "Target": "object", "Weight":"int"}) # retweets
    df_retweets = df_retweets.loc[df_retweets.index.repeat(df_retweets.Weight)] # expand RTs using Weights
    ## RETWEETak df-tik zerrende batera pasa w2v entrenatzeko
    retweets = df_retweets[["Source", "Target"]].values.tolist()


    #### FRIENDS DATASETAK IRAKURRI ####
    df_friends = pd.read_csv("friend.csv", dtype={"Source": "object", "Target": "object"}) # friends
    ## FRIENDSak df-tik zerrende batera pasa w2v entrenatzeko
    friends = df_friends[["Source", "Target"]].values.tolist()


    for dims in DIMENTSIOAK :

        #### RT EMBEDDINGS
        training_relational_embeddings("rt_embedding", dims, retweets)

        #### FRIENDS EMBEDINAK
        training_relational_embeddings("friend_embedding", dims, friends)

        #### MIXED EMBEDDINGS  -  RT + FRIENDS
        training_relational_embeddings("mixed_embedding", dims, friends+retweets)



    







