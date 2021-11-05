from kfp.v2.dsl import (Artifact,
                        Dataset,
                        Input,
                        Model,
                        Output,
                        Metrics,
                        component)

@component(
    packages_to_install = [
        "pandas",
        "google-cloud-storage",
        "gcsfs",
        "fsspec",
        "scikit-learn==0.20.4",
        "nltk"

        
    ],
)


def cluster_model(in_dataset: Input[Dataset], out_dataset: Output[Dataset]):
    
    from nltk.cluster import KMeansClusterer
    import pandas as pd
    import numpy as np
    #from scipy.spatial import distance_matrix
    import nltk
    from sklearn.cluster import KMeans
    import ast
    
    df= pd.read_csv('{}/synopsis_emb.csv'.format(in_dataset.path))

    
    def clustering_synopsis(data,NUM_CLUSTERS = 6):
        
        data['embeds'] = data['embeds'].apply(ast.literal_eval)
        X = np.array(data['embeds'].tolist())
        
        #km = KMeans(n_clusters=NUM_CLUSTERS)
        #km.fit(X)
        #clusters = km.labels_.tolist()
        #data["cluster"] = clusters
        
        kclusterer = KMeansClusterer(NUM_CLUSTERS, distance=nltk.cluster.util.cosine_distance,repeats=25,avoid_empty_clusters=True)
        assigned_clusters = kclusterer.cluster(X, assign_clusters=True)
        data['cluster'] = pd.Series(assigned_clusters, index=data.index)
        data['centroid'] = data['cluster'].apply(lambda x: kclusterer.means()[x])

        return data, assigned_clusters
    
    embeds_clusters, _ = clustering_synopsis(df)
    
    #def distance_from_centroid(row):
        #return distance_matrix([row['embeds']], [row['centroid'].tolist()])[0][0]
    
    #embeds_clusters['distance_from_centroid'] = embeds_clusters.apply(distance_from_centroid, axis=1)
    
    dataset_path = out_dataset.path
    dataset_path = dataset_path.replace('/gcs/','gs://')
    
    embeds_clusters.to_csv("{}/synopsis_cluster.csv".format(dataset_path), index=False)
    
    