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
        "fsspec"
    ],
)

def basic_prepoc(dataset: Output[Dataset], root_path:str):

    import pandas as pd
    import re
    
  
    df_synopsys_raw = pd.read_csv('{}/data/raw/synopsis_From_BQ.csv'.format(root_path))

    
    def prep_genre(genre):
    
        output = list()
    
        for i in genre:
        
            genre_string = i.strip()
        
            if len(genre_string) > 0:
                output.append(genre_string)
        
        output_str = ' '.join([str(elem) for elem in output])
    
        output_uniques =  list(set(output_str.split(' ')))
    
        final_output = ' '.join([str(elem) for elem in output_uniques])

        return final_output

    def replace_artigo(nome_programa):
        aconteceu = 0
        if nome_programa[-3:] == ", A":      
            padrao = ", A"
            sub = "A"
            aconteceu = 1
        elif nome_programa[-4:] == ", As":
            padrao = ", As"
            sub = "As"
            aconteceu = 1
        elif nome_programa[-3:] == ", O":      
            padrao = ", O"
            sub = "O"
            aconteceu = 1
        elif nome_programa[-4:] == ", Os":
            padrao = ", Os"
            sub = "Os"
            aconteceu = 1
        elif nome_programa[-5:] == ", The":
            padrao = ", The"
            sub = "The"
            aconteceu = 1

        if aconteceu == 1:
            temp_var = re.sub(padrao,"", nome_programa)
            return sub + " " + temp_var
        else:
            return nome_programa

    df_synopsys_raw['title'] = df_synopsys_raw['title'].apply(lambda x: replace_artigo(x))
    df_synopsys_raw['genre'] = df_synopsys_raw['genre'].str.replace("|", ",", regex=True)
    df_synopsys_raw['genre'] = df_synopsys_raw['genre'].str.replace("/", ",", regex=True)
    df_synopsys_raw['genre'] = df_synopsys_raw['genre'].apply(lambda x: x.split(','))
    df_synopsys_raw['genre'] = df_synopsys_raw['genre'].apply(lambda x: prep_genre(x))
    df_synopsys_raw['genre'] = df_synopsys_raw[['type', 'title', 'genre']].groupby(['type', 'title'])['genre'].transform(lambda x: ' '.join(x))
    df_synopsys_raw['synopsis_count'] = df_synopsys_raw['synopsis'].apply(lambda x: len(x))
    df_synopsys_raw['synopsis_max'] = df_synopsys_raw[['type', 'title', 'synopsis','synopsis_count']].groupby(['type', 'title'])['synopsis_count'].transform(max)
    df_synopsys_raw = df_synopsys_raw[df_synopsys_raw.synopsis_count == df_synopsys_raw.synopsis_max][['type', 'title', 'synopsis', 'genre']]
    df_synopsys_raw['genre'] = df_synopsys_raw['genre'].apply(lambda x: list(set(x.split(" "))))
    df_synopsys_raw['genre'] = df_synopsys_raw['genre'].apply(lambda x: " ".join(x))

    df_final = df_synopsys_raw[['type', 'title', 'synopsis','genre']].groupby(['type', 'title', 'synopsis','genre']).size().reset_index(name = "count_duplicates")

    df_final = df_final[['type', 'title', 'synopsis', 'genre']]
    
    dataset_path = dataset.path
    dataset_path = dataset_path.replace('/gcs/','gs://')
    
    df_final.to_csv("{}/synopsis_basic_clean.csv".format(dataset_path), index=False)