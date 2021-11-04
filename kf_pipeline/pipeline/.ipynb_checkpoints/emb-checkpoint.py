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
        "tensorflow",
        "tensorflow_text",
        "tensorflow_hub",
        "tf-models-official"
    ],
)


def emb_model(dataset: Input[Dataset]):

    import os
    import shutil

    import tensorflow as tf
    import tensorflow_hub as hub
    import tensorflow_text as text
    from official.nlp import optimization  # to create AdamW optmizer

    import numpy as np
    import pandas as pd

    synopsis_data = pd.read_csv('{}/synopsis_token_stopw.csv'.format(dataset.path))
    
    def normalization(embeds):
          norms = np.linalg.norm(embeds, 2, axis=1, keepdims=True)
          return embeds/norms
    
    synopsis_list = list(synopsis_data['sent_StopW'])
    title_list = list(synopsis_data['title'])
    genre_list = list(synopsis_data['genre'])
    
    synopsis_tf = tf.constant(synopsis_list)
    
    preprocessor = hub.KerasLayer("https://tfhub.dev/google/universal-sentence-encoder-cmlm/multilingual-preprocess/2")
    encoder = hub.KerasLayer("https://tfhub.dev/google/LaBSE/2")
    synopsis_embeds = encoder(preprocessor(synopsis_tf))['default']
    
    synopsis_embeds_df = synopsis_data[['title', 'synopsis', 'genre', 'sent_StopW']]
    embeds_df = pd.DataFrame(synopsis_embeds.numpy())
    synopsis_embeds_df['embeds'] = embeds_df.values.tolist()
    
    dataset_path = synopsis_embeds_df.path
    dataset_path = dataset_path.replace('/gcs/','gs://')
    
    df.to_csv("{}/synopsis_emb.csv".format(dataset_path), index=False)