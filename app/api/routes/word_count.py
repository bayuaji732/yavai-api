import pandas as pd
from fastapi import APIRouter, HTTPException
import nltk
from nltk.corpus import stopwords
from app.db.postgres import get_db_connection
from app.core import config

router = APIRouter()

if config.NLTK_DATA_DIR not in nltk.data.path:
    nltk.data.path.append(config.NLTK_DATA_DIR)

stop_words_english = set(stopwords.words('english'))
stop_words_indonesian = set(stopwords.words('indonesian'))
stop_words = stop_words_english.union(stop_words_indonesian)

@router.get("/datasets")
async def get_datasets_word_count():
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

@router.get("/feature-groups")
async def get_feature_groups_word_count():
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

@router.get("/notebooks")
async def get_notebooks_word_count():
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

@router.get("/file-management")
async def get_file_management_word_count():
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

@router.get("/model-zoo")
async def get_model_zoo_word_count():
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