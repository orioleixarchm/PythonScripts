#Importing packages
import requests
import pandas as pd
import urllib3
import numpy as np

#Parameter definition
urllib3.disable_warnings()
environment='https://inthubacc.belgrid.net/assethub_v1.0.acc'
section='physicalobject'
assetname='circuitbreaker'
urlID=f'{environment}/api/{section}/all{assetname}asset'
username='IU3309@belgrid.net'
password='Elnumero13*'
sel_id=[35724]

#Connecting to IDs API
response=requests.get(urlID, auth=(username,password), verify=False)

#Parsing JSON ID
json=response.json()
ID=[]
for item in json:
    for _, values in item.items():
        for _, values1 in values.items():
            ID.append(values1)

if not sel_id:
    IDsubset=ID[0]
else:
    IDsubset=sel_id

#Connecting and parsing Details API and JSON
colnames=[]
colvalues=[]

for id in IDsubset:
    urlDetail = f'{environment}/api/{section}/{assetname}assetdetail/{id}'
    responseD=requests.get(urlDetail, auth=(username,password), verify=False)
    jsonD = responseD.json()
    for key, item in jsonD.items():
        if isinstance(item, dict):
            for key1, item1 in item.items():
                if isinstance(item1, dict):
                    for key2, item2 in item1.items():
                        if len(colnames) < len(jsonD):
                            if key in ('maintainer'):
                                colnames.append(key)
                                colvalues.append(item2)
                            else:
                                colnames.append(key2)
                                colvalues.append(item2)
                        else:
                            colvalues.append(item2)
                else:
                    if key in ('subFleet', 'fleet', 'macroFleet', 'insulationMaterial', 'manufacturer'):
                        if len(colnames) < len(jsonD):
                            colnames.append(key)
                            colvalues.append(item1)
                        else:
                            colvalues.append(item1)
                    else:
                        if len(colnames) < len(jsonD):
                            colnames.append(key1)
                            colvalues.append(item1)
                        else:
                            colvalues.append(item1)
        else:
            if key in ('owners'):
                for _, value in item[0].items():
                    for _, value1 in value.items():
                        if len(colnames) < len(jsonD):
                            colnames.append(key)
                            colvalues.append(value1)
                        else:
                            colvalues.append(value1)
            else:
                if len(colnames) < len(jsonD):
                    colnames.append(key)
                    colvalues.append(item)
                else:
                    colvalues.append(item)

#Creating and reshaping Values' arrays
valores_flat = np.array(colvalues)



#Reading as Dataframe
df=pd.DataFrame({'ID': IDsubset})
print(jsonD)
print(colnames)
print(colvalues)
print(len(jsonD))
print(len(colnames))
print(len(colvalues))
