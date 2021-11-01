from kfp.dsl import (Artifact,
                        Dataset,
                        Input,
                        Model,
                        Output,
                        Metrics,
                        component)

@component(
    packages_to_install = [
        "pandas",
        "scikit-learn==0.20.4",
        "google-cloud-storage",
        "gcsfs",
        "fsspec"

    ],
)

def preproc_split_dataset(dataset_train: Output[Dataset],dataset_validation: Output[Dataset], dataset_test: Output[Dataset],root_path:str):

  import pandas as pd
  from sklearn.model_selection import train_test_split
  from sklearn import preprocessing
  import os
  from google.cloud import storage
    
  file_name = 'iris.txt'
  
  os.system('curl -o {}  https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'.format(file_name))  
  blob = storage.blob.Blob.from_string('{}/dataset/{}'.format(root_path,file_name), client=storage.Client())
  blob.upload_from_filename(file_name)
    
    
  columns = ['sepal_length','sepal_width','petal_length','petal_width','class']

  dfIris = pd.read_csv('{}/dataset/iris.txt'.format(root_path), names=columns)
  le = preprocessing.LabelEncoder()
  dfIris['classEncoder'] = le.fit_transform(dfIris['class'])
    
  X_train, X_val_test, y_train, y_val_test = train_test_split(dfIris.drop(['classEncoder','class'], axis=1), dfIris['classEncoder'], test_size=0.2)

  X_val, X_test, y_val, y_test = train_test_split(X_val_test,y_val_test, test_size=0.5)
    
  df_train = pd.concat([X_train,y_train],axis=1)
  df_val = pd.concat([X_val,y_val],axis=1)
  df_test = pd.concat([X_test,y_test],axis=1)
  
  dataset_train_path = dataset_train.path
  dataset_train_path = dataset_train_path.replace('/gcs/','gs://')
    
  dataset_validation_path = dataset_validation.path
  dataset_validation_path = dataset_validation_path.replace('/gcs/','gs://')

  dataset_test_path = dataset_test.path
  dataset_test_path = dataset_test_path.replace('/gcs/','gs://')
    
  df_train.to_csv(dataset_train_path, index=False)
  df_val.to_csv(dataset_validation_path, index=False)
  df_test.to_csv(dataset_test_path , index=False)
