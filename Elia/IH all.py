########################################################################################################################
#Author: Oriol Eixarch Oriol.EixarchIMejias@externel.be                                                                #
#Function: Infrastructure Business Analyst                                                                             #
#Creation date: 12/06/2024                                                                                             #
#Description: Data retrieval from IH'APIs (Elia and 50H), normalization of JSONs in tabular form and excel export.     #
#Note: To know the AH nomenclature and whether the "asset" part is included in them, reffer to the links found in the  #
# following page https://inthubacc.belgrid.net/assethub_v1.0.acc/swagger/index.html                                    #                                                           #
########################################################################################################################

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
envr = input("Acceptance or Production data (Acceptance/Production): ") or "Acceptance"
username, password = input('Introduce username (xxxx.belgrid.net) and password separated by a space: ').split()
environment = np.where(envr == "Acceptance",'https://inthubacc.belgrid.net/assethub_v1.0.acc','https://inthubprod.belgrid.net/assethub_v1.0.prod')
section = input("AH Section (topology, location, physicalobject...): ") or "physicalobject"
assetname = input("Asset Type (AH nomenclature): ") or "circuitBreaker"
asst = input("API link with 'asset' part (Y/N): ") or "Y" #See the link in swagger https://inthubacc.belgrid.net/assethub_v1.0.acc/swagger/index.html
owner = input("Elia of 50Hz Asset (Elia/50H)? ")
export = input("Would you like tot export the data (Y/N)? ") or "N"
path = input("Output path and file name (needs to end with .xlsx and use / instead of \): ") if export=='Y' else ''
urlID = f"{environment}{'/de' if owner=='50H' else ''}/api/{section}/all{assetname}{'asset' if asst=='Y' else ''}"
#username = 'IU3309@belgrid.net' #Username
#password = 'Elnumero14*' #Password
ID = input("Do you want to retrieve data for a particular ID? (blank for general ID data): ")

#Connecting to IDs API
response=requests.get(urlID, auth=(username,password), verify=False)

#Parsing JSON ID
json=response.json()
json_n = pd.json_normalize(json)

#Reading the details JSON based on the ID
if not ID:
    IDn = int(input("How many IDs needed?  ") or 1)
    ID = json_n.iloc[:IDn,-1]
    values = pd.DataFrame()
    if IDn > 1:
        for id in ID:
            urlDetail = f"{environment}{'/de' if owner=='50H' else ''}/api/{section}/{assetname}{'asset' if asst=='Y' else ''}detail/{id}"
            responseD = requests.get(urlDetail, auth=(username, password), verify=False)
            jsonD = responseD.json()
            json_detail = pd.json_normalize(jsonD)
            json_detail.index = [f'Asset_{id}']
            values = pd.concat([values, json_detail])

        print(values.T,'\n','\n','\n')
    elif IDn == 1:
        id = ID.values[0]
        print(id)
        urlDetail = f"{environment}{'/de' if owner=='50H' else ''}/api/{section}/{assetname}{'asset' if asst=='Y' else ''}detail/{id}"
        responseD = requests.get(urlDetail, auth=(username, password), verify=False)
        jsonD = responseD.json()
        values = pd.json_normalize(jsonD)
        values.index = [f'Asset_{id}']
        print(values.T,'\n','\n','\n')
else:
    id = ID
    print(id)
    urlDetail = f"{environment}{'/de' if owner=='50H' else ''}/api/{section}/{assetname}{'asset' if asst=='Y' else ''}detail/{id}"
    responseD = requests.get(urlDetail, auth=(username, password), verify=False)
    jsonD = responseD.json()
    values = pd.json_normalize(jsonD)
    values.index = [f'Asset_{id}']
    print(values.T, '\n', '\n', '\n')

if export == 'Y':
#    path=f'C:/Users/IU3309/OneDrive - ELIA GROUP/Desktop/Messages APMO/raw/Assets_{assetname}.xlsx' #To be replaced where the file needs to be output.
    values.T.to_excel(path, index=True)

