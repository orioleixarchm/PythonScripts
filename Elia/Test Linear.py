#Importing packages
import requests
import pandas as pd
import urllib3
import numpy as np

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

#Parameter definition
urllib3.disable_warnings()
envr = input("Acceptance (acc) or Production (prod) data: ") or "acc"
environment = np.where(envr == "acc",'https://inthubacc.belgrid.net/assethub_v1.0.acc', np.where(envr == "prod",'https://inthubprod.belgrid.net/assethub_v1.0.prod',''))
section = input("AH Section (topology, location, physicalobject...): ") or "physicalobject"
assetname = input("Asset Type (AH nomenclature): ") or "tower"
asst = input("API link with 'asset' part (Yes/No): ") or "No"
assetpart = np.where(asst == "Yes", 'asset', '') #See the link in swagger
urlID = f'{environment}/api/{section}/all{assetname}{assetpart}'
username = 'IU3309@belgrid.net'
password = 'Elnumero13*'
ID = input("Equipment ID: ")

#Connecting to IDs API
response=requests.get(urlID, auth=(username,password), verify=False)

#Parsing JSON ID
json = response.json()
IDs = pd.json_normalize(json, sep=".").iloc[:, 0]

if not ID:
    ID = IDs[0]

#Connecting to Details API
urlDetail = f'{environment}/api/{section}/{assetname}{assetpart}detail/{ID}'
responseD = requests.get(urlDetail, auth=(username,password), verify=False)
jsonD = responseD.json()
jsonDN = pd.json_normalize(jsonD, sep=".").T.reset_index()
jsonDN.columns = ["Attribute", "Value"]

print(jsonDN)

#Setting path for nested lists
listas = jsonDN.loc[jsonDN['Value'].apply(lambda x: isinstance(x, list)), "Attribute"].reset_index(drop=True)
listas = listas.drop(listas[listas == 'geographicalSite.contactPreference.postalAddresses'].index)
print(listas)


jsonDN = jsonDN.loc[~jsonDN['Attribute'].isin(listas), :]
for item in listas:
    print(item)
    jsonDN1 = pd.json_normalize(jsonD, sep=".", record_path=f'{item}', record_prefix=f'{item}').T.reset_index()
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
    jsonDN = pd.concat([jsonDN, jsonDN1], axis=0).reset_index(drop=True)

#Output and messages
print(jsonDN)
print(jsonDN.iloc[-6,1])
print(jsonDN.iloc[-6,1])
print(f'\n\n\n There are {IDs.nunique()} distinct equipemtns IDs for the {assetname} kind.')