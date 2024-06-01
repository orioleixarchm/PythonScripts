#Importing packages
import sys

import requests
import pandas as pd
import urllib3
import numpy as np

#Parameter definition
urllib3.disable_warnings()
envr= input("Acceptance (acc) or Production (prod) data: ") or "acc"
environment=np.where(envr == "acc",'https://inthubacc.belgrid.net/assethub_v1.0.acc', np.where(envr == "prod",'https://inthubprod.belgrid.net/assethub_v1.0.prod',''))
section=input("AH Section (topology, location, physicalobject...): ") or "physicalobject"
assetname=input("Asset Type (AH nomenclature): ") or "powertransformer"
asst=input("API link with 'asset' part (Yes/No): ") or "Yes"
assetpart= np.where(asst == "Yes", 'asset', '') #See the link in swagger
urlID=f'{environment}/api/{section}/all{assetname}{assetpart}'
username='IU3309@belgrid.net'
password='Elnumero13*'
ID=input("Equipment ID: ")

#Connecting to IDs API
response=requests.get(urlID, auth=(username,password), verify=False)

#Parsing JSON ID
json=response.json()
IDs=[]
for item in json:
    for _, value in item.items():
        for _, value1 in value.items():
            if isinstance(value1, list):
                for i in value1:
                    for _, values2 in i.items():
                        for _, values3 in values2.items():
                            IDs.append(values3)
            elif isinstance(value1, dict):
                for _, values4 in value1.items():
                    IDs.append(values4)
            else:
                IDs.append(value1)
if not ID:
    ID=IDs[0]

if section in ('location', 'Location'):
    print(f'\n\n\n There are {pd.Series(IDs).nunique()} distinct equipemtns IDs for the {assetname} kind.')
    sys.exit()

#Connecting and parsing Details API and JSON
colnames=[]
colvalues=[]

urlDetail = f'{environment}/api/{section}/{assetname}{assetpart}detail/{ID}'
responseD=requests.get(urlDetail, auth=(username,password), verify=False)
jsonD = responseD.json()
for key, item in jsonD.items():
    if isinstance(item, dict):
        if not item:
            colnames.append(key)
            colvalues.append('')
        else:
            for key1, item1 in item.items():
                if isinstance(item1, dict):
                    for key2, item2 in item1.items():
                        if isinstance(item2, dict):
                            for key3, item3 in item2.items():
                                if isinstance(item3, dict):
                                    for key4, item4 in item3.items():
                                        if key4 in ('name', 'fullName', 'id', 'fr', 'nl', 'en', assetname):
                                            if key3 in ('name', 'fullName', 'id', 'fr', 'nl', 'en', assetname):
                                                if key2 in ('name', 'fullName', 'id', 'fr', 'nl', 'en', assetname):
                                                    if key1 in ('name', 'fullName', 'id', 'fr', 'nl', 'en', assetname):
                                                        colnames.append(key)
                                                        colvalues.append(item4)
                                                    else:
                                                        colnames.append(key1)
                                                        colvalues.append(item4)
                                                else:
                                                    colnames.append(key2)
                                                    colvalues.append(item4)
                                            else:
                                                colnames.append(key3)
                                                colvalues.append(item4)
                                        else:
                                            colnames.append(key4)
                                            colvalues.append(item4)
                                else:
                                    colnames.append(key1)
                                    colvalues.append(item3)
                        elif isinstance(item2, list):
                            if not item2:
                                colnames.append(key2)
                                colvalues.append('')
                            else:
                                for sublist in item2:
                                    if isinstance(sublist, dict):
                                        for cle, valeur in sublist.items():
                                            if isinstance(valeur, dict):
                                                for cle1, valeur1 in valeur.items():
                                                    if isinstance(valeur1,dict):
                                                        for cle2, valeur2 in valeur1.items():
                                                            if cle2 in ('name', 'fullName', 'id', assetname):
                                                                if cle1 in ('name', 'fullName', 'id', 'fr', 'nl', 'en', assetname):
                                                                    if cle in ('name', 'fullName', 'id', 'fr', 'nl', 'en', assetname):
                                                                        if key2 in ('name', 'fullName', 'id', 'fr', 'nl', 'en', assetname):
                                                                            if key1 in ('name', 'fullName', 'id', 'fr', 'nl', 'en', assetname):
                                                                                colnames.append(key)
                                                                                colvalues.append(valeur2)
                                                                            else:
                                                                                colnames.append(key1)
                                                                                colvalues.append(valeur2)
                                                                        else:
                                                                            colnames.append(key2)
                                                                            colvalues.append(valeur2)
                                                                    else:
                                                                        colnames.append(cle)
                                                                        colvalues.append(valeur2)
                                                                else:
                                                                    colnames.append(cle1)
                                                                    colvalues.append(valeur2)
                                                            elif cle2 in ('fr', 'nl', 'en'):
                                                                if cle1 in ('name', 'fullName', 'id', assetname):
                                                                    if cle in ('name', 'fullName', 'id', assetname):
                                                                        if key2 in ('name', 'fullName', assetname):
                                                                            if key1 in ('name', 'fullName', assetname):
                                                                                colnames.append(key + '_' + cle2)
                                                                                colvalues.append(valeur2)
                                                                            else:
                                                                                colnames.append(key1 + '_' + cle2)
                                                                                colvalues.append(valeur2)
                                                                        else:
                                                                            colnames.append(key2 + '_' + cle2)
                                                                            colvalues.append(valeur2)
                                                                    else:
                                                                        colnames.append(cle + '_' + cle2)
                                                                        colvalues.append(valeur2)
                                                                else:
                                                                    colnames.append(cle1 + '_' + cle2)
                                                                    colvalues.append(valeur2)
                                                            else:
                                                                colnames.append(cle2)
                                                                colvalues.append(valeur2)
                                                    else:
                                                        if cle1 in ('name', 'fullName', 'id', 'fr', 'nl', 'en', assetname):
                                                            if cle in ('name', 'fullName', 'id', 'fr', 'nl', 'en', assetname):
                                                                if key2 in ('name', 'fullName', 'id', 'fr', 'nl', 'en', assetname):
                                                                    if key1 in ('name', 'fullName', 'id', 'fr', 'nl', 'en', assetname):
                                                                        colnames.append(key)
                                                                        colvalues.append(valeur1)
                                                                    else:
                                                                        colnames.append(key1)
                                                                        colvalues.append(valeur1)
                                                                else:
                                                                    colnames.append(key2)
                                                                    colvalues.append(valeur1)
                                                            else:
                                                                colnames.append(cle)
                                                                colvalues.append(valeur1)
                                                        else:
                                                            colnames.append(cle1)
                                                            colvalues.append(valeur1)
                                            else:
                                                if cle in ('name', 'fullName', 'id', 'fr', 'nl', 'en', assetname):
                                                    if key2 in ('name', 'fullName', 'id', 'fr', 'nl', 'en', assetname):
                                                        if key1 in ('name', 'fullName', 'id', 'fr', 'nl', 'en', assetname):
                                                            colnames.append(key)
                                                            colvalues.append(valeur)
                                                        else:
                                                            colnames.append(key1)
                                                            colvalues.append(valeur)
                                                    else:
                                                        colnames.append(key2)
                                                        colvalues.append(valeur)
                                                else:
                                                    colnames.append(cle)
                                                    colvalues.append(valeur)
                                    else:
                                        colnames.append(key2)
                                        colvalues.append(sublist)
                        else:
                            if key2 in ('name', 'fullName', 'id', assetname):
                                if key1 in ('name', 'fullName', 'id', 'fr', 'nl', 'en', assetname):
                                    colnames.append(key)
                                    colvalues.append(item2)
                                else:
                                    colnames.append(key1)
                                    colvalues.append(item2)
                            elif key2 in ('fr', 'nl', 'en'):
                                if key1 in ('name', 'fullName', 'id', 'fr', 'nl', 'en', assetname):
                                    colnames.append(key + '_' + key2)
                                    colvalues.append(item2)
                                else:
                                    colnames.append(key1 + '_' + key2)
                                    colvalues.append(item2)
                            else:
                                colnames.append(key2)
                                colvalues.append(item2)
                elif isinstance(item1, list):
                    if not item1:
                        colnames.append(key1)
                        colvalues.append('')
                    else:
                        for itlist in item1:
                            if isinstance(itlist, dict):
                                for clau, valor in itlist.items():
                                    if clau in ('name', 'fullName', 'id', 'fr', 'nl', 'en', assetname):
                                        colnames.append(key1)
                                        colvalues.append(valor)
                                    else:
                                        colnames.append(clau)
                                        colvalues.append(valor)
                            else:
                                colnames.append(key1)
                                colvalues.append(itlist)
                else:
                    if key1 in ('name', 'fullName', 'id', 'fr', 'nl', 'en', assetname):
                        colnames.append(key)
                        colvalues.append(item1)
                    else:
                        colnames.append(key1)
                        colvalues.append(item1)
    elif isinstance(item, list):

        if not item:
            colnames.append(key)
            colvalues.append('')
        else:
            for lista in item:
                if isinstance(lista, dict):
                    if not lista:
                        colnames.append(key)
                        colvalues.append('')
                    else:
                        for llave, value in lista.items():
                            if isinstance(value, dict):
                                for llave1, value1 in value.items():
                                    if isinstance(value1, dict):
                                        for llave2, value2 in value1.items():
                                            if isinstance(value2, dict):
                                                for llave3, value3 in value2.items():
                                                        if isinstance(value3, dict):
                                                            for llave4, value4 in value3.items():
                                                                if llave4 in ('name', 'fullName', 'id', 'fr', 'nl', 'en', assetname):
                                                                    if llave3 in ('name', 'fullName', 'id', 'fr', 'nl', 'en', assetname):
                                                                        if llave2 in ('name', 'fullName', 'id', 'fr', 'nl', 'en', assetname):
                                                                            if llave1 in ('name', 'fullName', 'id', 'fr', 'nl', 'en', assetname):
                                                                                if llave in ('name', 'fullName', 'id', 'fr', 'nl', 'en', assetname):
                                                                                    colnames.append(key)
                                                                                    colvalues.append(value4)
                                                                                else:
                                                                                    colnames.append(llave)
                                                                                    colvalues.append(value4)
                                                                            else:
                                                                                colnames.append(llave1)
                                                                                colvalues.append(value4)
                                                                        else:
                                                                            colnames.append(llave2)
                                                                            colvalues.append(value4)
                                                                    else:
                                                                        colnames.append(llave3)
                                                                        colvalues.append(value4)
                                                                else:
                                                                    colnames.append(llave4)
                                                                    colvalues.append(value4)
                                                        else:
                                                            if llave3 in ('name', 'fullName', 'id', 'fr', 'nl', 'en', assetname):
                                                                if llave2 in ('name', 'fullName', 'id', 'fr', 'nl', 'en', assetname):
                                                                    if llave1 in ('name', 'fullName', 'id', 'fr', 'nl', 'en', assetname):
                                                                        if llave in ('name', 'fullName', 'id', 'fr', 'nl', 'en', assetname):
                                                                            colnames.append(key)
                                                                            colvalues.append(value3)
                                                                        else:
                                                                            colnames.append(llave)
                                                                            colvalues.append(value3)
                                                                    else:
                                                                        colnames.append(llave1)
                                                                        colvalues.append(value3)
                                                                else:
                                                                    colnames.append(llave2)
                                                                    colvalues.append(value3)
                                                            else:
                                                                colnames.append(llave3)
                                                                colvalues.append(value3)
                                            else:
                                                if llave2 in ('name', 'fullName', 'id', assetname):
                                                    if llave1 in ('name', 'fullName', 'id', assetname):
                                                        if llave in ('name', 'fullName', 'id', assetname):
                                                            colnames.append(key)
                                                            colvalues.append(value2)
                                                        else:
                                                            colnames.append(llave)
                                                            colvalues.append(value2)
                                                    else:
                                                        colnames.append(llave1)
                                                        colvalues.append(value2)
                                                else:
                                                    colnames.append(llave2)
                                                    colvalues.append(value2)
                                    elif isinstance(value1, list):
                                        if not value1:
                                            colnames.append(llave1)
                                            colvalues.append('')
                                        else:
                                            for llista in value1:
                                                if isinstance(llista, dict):
                                                    for llkey, llvalues in llista.items():
                                                        if isinstance(llvalues,dict):
                                                            for llkey1, llvalues1 in llvalues.items():
                                                                if llkey1 in ('name', 'fullName', 'id', 'fr', 'nl', 'en', assetname):
                                                                    if llkey in ('name', 'fullName', 'id', 'fr', 'nl', 'en', assetname):
                                                                        if llave1 in ('name', 'fullName', 'id', 'fr', 'nl', 'en', assetname):
                                                                            if llave in ('name', 'fullName', 'id', 'fr', 'nl', 'en', assetname):
                                                                                colnames.append(key)
                                                                                colvalues.append(llvalues1)
                                                                            else:
                                                                                colnames.append(llave)
                                                                                colvalues.append(llvalues1)
                                                                        else:
                                                                            colnames.append(llave1)
                                                                            colvalues.append(llvalues1)
                                                                    else:
                                                                        colnames.append(llkey)
                                                                        colvalues.append(llvalues1)
                                                                else:
                                                                    colnames.append(llkey1)
                                                                    colvalues.append(llvalues1)
                                                        else:
                                                            if llkey in ('name', 'fullName', 'id', 'fr', 'nl', 'en', assetname):
                                                                if llave1 in ('name', 'fullName', 'id', 'fr', 'nl', 'en', assetname):
                                                                    if llave in ('name', 'fullName', 'id', 'fr', 'nl', 'en', assetname):
                                                                        colnames.append(key)
                                                                        colvalues.append(llvalues)
                                                                    else:
                                                                        colnames.append(llave)
                                                                        colvalues.append(llvalues)
                                                                else:
                                                                    colnames.append(llave1)
                                                                    colvalues.append(llvalues)
                                                            else:
                                                                colnames.append(llkey)
                                                                colvalues.append(llvalues)
                                                else:
                                                    if llave1 in ('name', 'fullName', 'id', 'fr', 'nl', 'en', assetname):
                                                        if llave in ('name', 'fullName', 'id', 'fr', 'nl', 'en', assetname):
                                                            colnames.append(key)
                                                            colvalues.append(llista)
                                                        else:
                                                            colnames.append(llave)
                                                            colvalues.append(llista)
                                                    else:
                                                        colnames.append(llave1)
                                                        colvalues.append(llista)
                                    else:
                                        if llave1 in ('name', 'fullName', 'id', 'fr', 'nl','en',assetname):
                                            if llave in ('name', 'fullName', 'id', 'fr', 'nl','en',assetname):
                                                colnames.append(key)
                                                colvalues.append(value1)
                                            else:
                                                colnames.append(llave)
                                                colvalues.append(value1)
                                        else:
                                            colnames.append(llave1)
                                            colvalues.append(value1)
                            else:
                                if llave in ('name', 'fullName', 'id', 'fr', 'nl','en',assetname):
                                    colnames.append(key)
                                    colvalues.append(value)
                                else:
                                    colnames.append(llave)
                                    colvalues.append(value)
                else:
                    colnames.append(key)
                    colvalues.append(lista)
    else:
        colnames.append(key)
        colvalues.append(item)

#Reading as Dataframe
data = pd.DataFrame({'Attribute': colnames, 'Value': colvalues})

#Output and messages
print(data)
print(f'\n\n\n There are {pd.Series(IDs).nunique()} distinct equipemtns IDs for the {assetname} kind.')















for item in listas:
    jsonDN1 = pd.json_normalize(jsonD, sep="_", record_path=f'{item}', record_prefix=f'{item}').T.reset_index()
    if jsonDN1.shape[1]==1:
        jsonDN1.columns = ["Attribute"]
    elif jsonDN1.shape[1]==2:
        jsonDN1.columns = ["Attribute", "Value"]
    elif jsonDN1.shape[1]==3:
        jsonDN1.columns = ["Attribute", "Value", "Value2"]
    elif jsonDN1.shape[1]==4:
        jsonDN1.columns = ["Attribute", "Value", "Value2", "Value3"]
    elif jsonDN1.shape[1]==5:
        jsonDN1.columns = ["Attribute", "Value", "Value2", "Value3", "Value4"]
    elif jsonDN1.shape[1]==6:
        jsonDN1.columns = ["Attribute", "Value", "Value2", "Value3", "Value4", "Value5"]
    else:
        jsonDN1.columns = ["Attribute", "Value"]
    jsonDN = pd.concat([jsonDN, jsonDN1], axis=0)

#Output and messages
print(jsonDN)
print(f'\n\n\n There are {IDs.nunique()} distinct equipemtns IDs for the {assetname} kind.')