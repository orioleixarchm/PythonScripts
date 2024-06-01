#Importing packages
import pandas as pd
import json

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

#Reading json
file_path='//belgrid.net/FSApp/Rabbitmqmanager/OPER_DATA/ACC/Acc/Asset/InfrastructureChangeRequestCreated.OutBckQ/2024/05/'
filename='acc sch_20240506150007_001.json'
url=f'{file_path}{filename}'

with open(url,'r') as file:
    jsonF=json.load(file)
    data=pd.json_normalize(jsonF)
    df = data.drop(columns='content').reset_index(drop=True)

content=pd.DataFrame()
for item in data['content']:
    iter=pd.json_normalize(json.loads(item[3:])).drop(columns='InfrastructureChangeRequest.InfrastructureNeeds').reset_index(drop=True)  # encoding='utf-8-sig'
    inneriter=pd.json_normalize(json.loads(item[3:]),record_path=["InfrastructureChangeRequest","InfrastructureNeeds"]).reset_index(drop=True) #encoding='utf-8-sig'
    iterf=pd.concat([iter,inneriter], axis=1)
    content = pd.concat([content, iterf], axis=0).reset_index(drop=True)

df=pd.concat([content,df], axis=1).T.reset_index()
col_mapping={'index':'Attribute'}
for i in range(0, df.shape[1]):
    col_mapping.update({i:f'Message{i+1}'})
df.columns=df.columns.map(col_mapping)
print(df)


