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
        "tensorflow==2.3.0",
        "tensorflow_text",
        "tensorflow_hub"
        
    ],
)


def emb_model(in_dataset: Input[Dataset], out_dataset: Output[Dataset]):

    import os
    import shutil

    import tensorflow as tf
    import tensorflow_hub as hub
    import tensorflow_text as text
    

    import numpy as np
    import pandas as pd

    synopsis_data = pd.read_csv('{}/synopsis_token_stopw.csv'.format(in_dataset.path))
    
    def normalization(embeds):
          norms = np.linalg.norm(embeds, 2, axis=1, keepdims=True)
          return embeds/norms
    
    synopsis_list = list(synopsis_data['sent_StopW'])
    title_list = list(synopsis_data['title'])
    genre_list = list(synopsis_data['genre'])
    
    synopsis_tf = tf.constant(synopsis_list)
    
    preprocessor = hub.KerasLayer("https://tfhub.dev/google/universal-sentence-encoder-cmlm/multilingual-preprocess/2")
    encoder = hub.KerasLayer("https://tfhub.dev/google/LaBSE/2")
    
    init = 200
    end =0
    embeds_concat = encoder(preprocessor(synopsis_tf[0:200]))['default']

    while((len(synopsis_tf) - len(embeds_concat)) >= 200):
        end = init + 200
        emb_aux = encoder(preprocessor(synopsis_tf[init:end]))['default']
        embeds_concat  = tf.concat([embeds_concat,emb_aux],0)
        init = end
    
    embeds_concat  = tf.concat([embeds_concat,encoder(preprocessor(synopsis_tf[end:]))['default']],0)
    
    synopsis_embeds_df = synopsis_data[['title', 'synopsis', 'genre', 'sent_StopW']]
    embeds_df = pd.DataFrame(embeds_concat.numpy())
    synopsis_embeds_df['embeds'] = embeds_df.values.tolist()
    
    dataset_path = out_dataset.path
    dataset_path = dataset_path.replace('/gcs/','gs://')
    
    synopsis_embeds_df.to_csv("{}/synopsis_emb.csv".format(dataset_path), index=False)