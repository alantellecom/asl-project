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
        "stop-words",
        "nltk"
    ],
)




def token_stopw_preproc(basic_clean_dataset: Input[Dataset], token_stopw_dataset: Output[Dataset]):


    import re
    import nltk
    import string
    import unicodedata
    import pandas as pd
    from stop_words import get_stop_words

    nltk.download('wordnet')
    nltk.download('stopwords')
    nltk.download('punkt')
    
    re_tab_pattern       = r"(\t)"
    re_return_pattern    = r"(\r)"
    re_comma_pattern     = r"(,{2,})"
    re_space_pattern     = r"(\s{1,})"
    re_linefeed_pattern  = r"(\n)"

    re_tab_obj           = re.compile(re_tab_pattern)
    re_return_obj        = re.compile(re_return_pattern)
    re_comma_obj         = re.compile(re_comma_pattern)
    re_space_obj         = re.compile(re_space_pattern)
    re_linefeed_obj      = re.compile(re_linefeed_pattern)

    stop_ptBr = set(nltk.corpus.stopwords.words('portuguese'))
    punctuation = list(string.punctuation)
    stop_ptBr.update(punctuation)
    
    def strip_accents(text):
        try:
            text = unicode(text, 'utf-8')
        except NameError: # unicode is a default on python 3 
            pass

        text = unicodedata.normalize('NFD', text)\
               .encode('ascii', 'ignore')\
               .decode("utf-8")
        return str(text)

    def remove_dot(text):
        return re.sub(r'\.(?!\d)', '', text)

    def remove_numbers(text):
        return re.sub(r'[0-9]', '', text)

    def all_Regex_transformations(text, has_space=True):
        final_text = text

        if has_space is False:
            final_text = final_text.replace(' ' , '_')

        final_text = strip_accents(final_text)
        final_text = remove_dot(final_text)
        final_text = remove_numbers(final_text)
        final_text = re.sub(re_comma_obj     , ''  , final_text)
        final_text = final_text.replace('\t' , ' ')
        final_text = final_text.replace('\n' , ' ')
        final_text = final_text.replace('\r' , ' ')
        final_text = final_text.replace('\\' , ' ')
        final_text = final_text.replace(','  , ' ')
        final_text = final_text.replace('.'  , ' ')
        final_text = final_text.replace('-'  , ' ')
        final_text = final_text.replace('/'  , ' ')    
        final_text = final_text.replace('"'  , ' ')
        final_text = final_text.replace(')'  , ' ')
        final_text = final_text.replace('('  , ' ')
        final_text = final_text.replace('!'  , ' ')
        final_text = final_text.replace('?'  , ' ')
        final_text = re.sub(re_space_obj     , ' '  , final_text)

        return final_text.lower()
    
    df = pd.read_csv('{}/synopsis_basic_clean.csv'.format(basic_clean_dataset.path))
    
    df['sinopse_prep']  = df.apply(lambda x: all_Regex_transformations(text=x['synopsis']) , axis=1 )
    df['sinopse_token'] = df['sinopse_prep'].apply(nltk.tokenize.word_tokenize) 
    df['sinopse_token_stop'] = df['sinopse_token'].apply(lambda x: [item for item in x if item not in stop_ptBr])
    df['word len'] = df['sinopse_token_stop'].apply(lambda x: len(x)>10)
    df = df[df['word len']==True]
    df['sent_StopW'] = df['sinopse_token_stop'].apply(lambda x: ' '.join(x))
    
    dataset_path = token_stopw_dataset.path
    dataset_path = dataset_path.replace('/gcs/','gs://')
    
    df.to_csv("{}/synopsis_token_stopw.csv".format(dataset_path), index=False)