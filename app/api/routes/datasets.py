import pandas as pd
from fastapi import APIRouter, HTTPException
from nltk.corpus import stopwords
from db.postgres import get_db_connection

router = APIRouter()

stop_words_english = set(stopwords.words('english'))
stop_words_indonesian = set(stopwords.words('indonesian'))
stop_words = stop_words_english.union(stop_words_indonesian)

@router.get("/dataset-word-count")
async def dataset_word_count():
    try:
        with get_db_connection() as conn:
            cur = conn.cursor()
            cur.execute('''
                SELECT datasetname 
                FROM dataset 
                WHERE (datasetpermission = 0 OR datasetpermission = 2) 
                  AND hidden = False;
            ''')
            result = cur.fetchall()
            cur.close()
        
        df = pd.DataFrame(result, columns=['datasetname'])
        df['datasetname'] = df['datasetname'].apply(
            lambda x: [word.lower() for word in x.split() if word.lower() not in stop_words]
        )
        
        df = df.dropna().explode('datasetname')
        df = df[df['datasetname'].str.strip() != '']
        
        word_counts = df['datasetname'].value_counts().reset_index(name='frequency')
        word_counts.columns = ['datasetname', 'frequency']
        data = word_counts.head(50).to_json()
        
        return {
            "StatusCode": 200,
            "Status": "OK",
            "data": data
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/featurestore-word-count")
async def featurestore_word_count():
    try:
        with get_db_connection() as conn:
            cur = conn.cursor()
            cur.execute('''
                SELECT name 
                FROM feature_group 
                WHERE (permission = 0 OR permission = 2) 
                  AND hidden = False;
            ''')
            result = cur.fetchall()
            cur.close()
        
        df = pd.DataFrame(result, columns=['name'])
        df['name'] = df['name'].apply(
            lambda x: [word.lower() for word in x.split() if word.lower() not in stop_words]
        )
        
        df = df.dropna().explode('name')
        df = df[df['name'].str.strip() != '']
        
        word_counts = df['name'].value_counts().reset_index(name='frequency')
        word_counts.columns = ['name', 'frequency']
        data = word_counts.head(50).to_json()
        
        return {
            "StatusCode": 200,
            "Status": "OK",
            "data": data
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/name-notebook-word-count")
async def name_notebook_word_count():
    try:
        with get_db_connection() as conn:
            cur = conn.cursor()
            cur.execute('SELECT name_notebook FROM notebook_zeppelin WHERE public = True;')
            result = cur.fetchall()
            cur.close()
        
        df = pd.DataFrame(result, columns=['name_notebook'])
        df['name_notebook'] = df['name_notebook'].apply(
            lambda x: [word.lower() for word in x.split() if word.lower() not in stop_words]
        )
        
        df = df.explode('name_notebook').value_counts().reset_index(name='frequency')
        df = df[df['name_notebook'].str.strip() != '']
        data = df.head(50).to_json()
        
        return {
            "StatusCode": 200,
            "Status": "OK",
            "data": data
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/filemanagement-word-count")
async def filemanagement_word_count():
    try:
        with get_db_connection() as conn:
            cur = conn.cursor()
            cur.execute('SELECT filemanagementname FROM filemanagement WHERE filemanagementpermission IN (0, 2);')
            result = cur.fetchall()
            cur.close()
        
        df = pd.DataFrame(result, columns=['filemanagementname'])
        df['filemanagementname'] = df['filemanagementname'].apply(
            lambda x: [word.lower() for word in x.split() if word.lower() not in stop_words]
        )
        
        df = df.explode('filemanagementname').value_counts().reset_index(name='frequency')
        df = df[df['filemanagementname'].str.strip() != '']
        data = df.head(50).to_json()
        
        return {
            "StatusCode": 200,
            "Status": "OK",
            "data": data
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/model-zoo-word-count")
async def model_zoo_word_count():
    try:
        with get_db_connection() as conn:
            cur = conn.cursor()
            cur.execute('SELECT model_zoo_name FROM model_zoo WHERE model_zoo_permission IN (0, 2);')
            result = cur.fetchall()
            cur.close()
        
        df = pd.DataFrame(result, columns=['model_zoo_name'])
        df['model_zoo_name'] = df['model_zoo_name'].apply(
            lambda x: [word.lower() for word in x.split() if word.lower() not in stop_words]
        )
        
        df = df.explode('model_zoo_name').value_counts().reset_index(name='frequency')
        df = df[df['model_zoo_name'].str.strip() != '']
        data = df.head(50).to_json()
        
        return {
            "StatusCode": 200,
            "Status": "OK",
            "data": data
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))