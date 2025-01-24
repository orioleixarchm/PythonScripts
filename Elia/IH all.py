########################################################################################################################
# Author: Oriol Eixarch                                                                                                #
# Contact: Oriol.EixarchIMejias@externel.be                                                                            #
# Function: Infrastructure Business Analyst                                                                            #
# Creation date: 12/06/2024                                                                                            #
# Description: Data retrieval from IH's APIs (Elia + 50H), normalization of JSONs in tabular format and excel export.  #
# Note: To know the AH nomenclature is included in said links refer to the links found in the following page           #
# https://inthubacc.belgrid.net/assethub_v1.0.acc/swagger/index.html                                                   #
########################################################################################################################                                                                                                                    #
# README                                                                                                               #
# The section "Parameter definition" asks the user to provide parameters regarding the environment to work with, their #
#                                                                                                                      #
# credentials, which section of the AH would they want to work with, which asset/activity or specific subpart, for     #
# which branch, Elia (Belgium) or 50H (Germany), it also provides the option to export the data into Excel provided    #
# that the users introduces a valid path. The user can select a particular ID to retrieve, if left blank the user will #
# be given the option to retrieve a number of his or her choice of IDs including all of them.                          #
# The section "Connecting to IDs API" retrieves all the IDs and only the IDs for the selected subtype of asset or      #
# activity within the subsection.                                                                                      #
# The section "Reading the details JSON based on ID" will retrieve the subselection items' details (all attributes)    #
# looping over a list of IDs that will be determined by the desired amount of IDs introduced by the user. It will also #
# export the data to an excel file following the user defined export path if he or she chooses to.                     #
########################################################################################################################

#Importing packages
import sys
import requests
import pandas as pd
import urllib3
import numpy as np

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

#Parameter definition
urllib3.disable_warnings()
envrR = int(input("Choose 0 for Acceptance or 1 for Production data: ") or 0)
envrmap = {0:"Acceptance", 1:"Production"}
envr = envrmap[envrR]
username, password = input('Introduce username (xxx@belgrid.net) and password separated by a space: ').split() or ('IU3309@belgrid.net', 'Elnumero15*')
environment = np.where(envr == "Acceptance",'https://inthubacc.belgrid.net/assethub_v1.0.acc','https://inthubprod.belgrid.net/assethub_v1.0.prod')
sectionR = int(input("AH Section (1 for Topology, 2 for Location, 3 for Physicalobject, 4 for Activity, 5 for Contact Point): ") or 1)
print('!WARNING! Currentpathdetdetail API not available. !WARNING!') if sectionR==1 else ''
sectionmap = {1:'Topology', 2:'Location', 3:'Physicalobject', 4:'Activity',5:'ContactPoint'}
section = sectionmap[sectionR]
assetname = input("Asset Type e.g circuitbreaker / Location e.g geographicalsite / Activity e.g pruninginspectionpolicy / Topology e.g line / ContactPoint e.g wgstoaddress (AH nomenclature): ") or "substation"
ownerR = int(input("0 for Elia or 1 for 50Hz Asset: ") or 0)
ownermap = {0:'Elia', 1:'50H'}
owner = ownermap[ownerR]
export = input("Would you like tot export the data (Y/N)? ") or "N"
pathR, filename = input("Output path and file name (separated by a semicolon ;) e.g C:\\Users\\Username Assets: ").replace("\\","/").split(';') if export=='Y' else ('', '')
path = f'{pathR}/{filename}.xlsx'
assetpartsBe = ['rectifiercontroller','currenttransformer','rectifier','meter','powertransformer','disconnector','gascompartment','baycontroller',
              'protectionrelay','alternator','motor','generatorcontroller','generatorbattery','fuse','voltagetransformer','inductor','resistor',
              'compressor','surgearester','circuitbreaker','dieselgenerator','faultrecorder']
assetpartsDe = ['dieselgenerator','busbar','voltagetransformer','earthingswitch','powertransformer', 'disconnector','circuitbreaker','shuntreactor',
                'currenttransformer','resistor','inductor','surgearester']

# Some assets APIs contain the word asset, whereas others do not, therefore sometimes we will need to include it in the url,
# those assets can be found in 'assetpartsDe' and 'assetpartsBe' lists; section names are not standard.
assetparts = assetpartsBe if owner=='Elia' else assetpartsDe
section = 'topology' if owner=='50H' and sectionR==1 else section
urlID = f"{environment}{'/de' if owner=='50H' else ''}/api/{section}/all{assetname}{'asset' if assetname in assetparts else ''}"
urlID = f"{environment}{'/de' if owner=='50H' else ''}/api/{section}/{assetname}" if section == 'ContactPoint' else urlID
ID = input("Do you want to retrieve data for a particular individual ID? If so write the ID, else leave this field blank: ")

#Connecting to IDs API
response=requests.get(urlID, auth=(username,password), verify=False)

#Parsing JSON ID
json=response.json()
json_n = pd.json_normalize(json)

#Reading the details JSON based on ID and other parameters
if section == 'ContactPoint':
    values = json_n
    print(values.T.reset_index(names='Attributes'))
    if export == 'Y':
        values.T.reset_index(names='Attributes').to_excel(path, index=False)
else:
    values = pd.DataFrame()
    if not ID:
        IDn = int(input("How many IDs needed? for all IDs write 0, blank for just 1: ") or 1)
        ID = json_n.iloc[:,0] if IDn == 0 else json_n.iloc[:IDn,0]
        if IDn != 1:
            for id in ID:
                if id == 'Removed':
                    print(f'Value of ID is: {id}; stopping the code as API is not in service')
                    sys.exit()
                urlDetail = f"{environment}{'/de' if owner=='50H' else ''}/api/{section}/{assetname}{'asset' if assetname in assetparts else ''}detail/{id}"
                responseD = requests.get(urlDetail, auth=(username, password), verify=False)
                jsonD = responseD.json()
                nested_data = pd.DataFrame()
                columns_to_remove = []
                for key, value in jsonD.items():
                    if isinstance(value,list) and value:
                        if pd.json_normalize(jsonD, record_path=key, errors='ignore').shape[0]==1:
                            nested = pd.json_normalize(jsonD, record_path=key, errors='ignore')
                            for col in nested.columns:
                                nested.rename(columns={col: f'{key}.{col}'}, inplace=True)
                            nested_data = pd.concat([nested_data, nested], axis=1)
                            columns_to_remove.append(key)
                        else:
                            nested_df = pd.json_normalize(jsonD, record_path=key, errors='ignore')
                            for col in nested_df.columns:
                                nested_values = nested_df.loc[:, col].tolist()
                                nested_complex = pd.DataFrame({f'{key}.{col}': [nested_values]})
                                nested_data = pd.concat([nested_data, nested_complex], axis=1)
                            columns_to_remove.append(key)
                json_detailR = pd.json_normalize(jsonD, errors='ignore')
                json_detailR.drop(columns=columns_to_remove, inplace=True)
                list_dict = pd.DataFrame()
                columns_to_remove = []
                for col in json_detailR.columns:
                    if isinstance(json_detailR[col].values[0], list) and json_detailR[col].values[0]:
                        list_dict = pd.concat([list_dict, pd.json_normalize(json_detailR[col].values[0][0])], axis=1)
                        columns_to_remove.append(col)
                json_detail = json_detailR.drop(columns=columns_to_remove)
                json_detail = pd.concat([json_detail, nested_data, list_dict], axis=1)
                json_detail = json_detail.drop(columns=['wellKnownText', 'projection']) if ('wellKnownText' or 'projection') in json_detail.columns else json_detail
                json_detail.index = [f'Item_{id}']
                values = pd.concat([values, json_detail], axis=0)
        elif IDn == 1:
            id = ID.values[0]
            if id == 'Removed':
                print(f'Value of ID is: {id}; stopping the code as API is not in service')
                sys.exit()
            urlDetail = f"{environment}{'/de' if owner=='50H' else ''}/api/{section}/{assetname}{'asset' if assetname in assetparts else ''}detail/{id}"
            responseD = requests.get(urlDetail, auth=(username, password), verify=False)
            jsonD = responseD.json()
            nested_data = pd.DataFrame()
            columns_to_remove = []
            for key, value in jsonD.items():
                if isinstance(value, list) and value:
                    if pd.json_normalize(jsonD, record_path=key, errors='ignore').shape[0] == 1:
                        nested = pd.json_normalize(jsonD, record_path=key, errors='ignore')
                        for col in nested.columns:
                            nested.rename(columns={col: f'{key}.{col}'}, inplace=True)
                        nested_data = pd.concat([nested_data, nested], axis=1)
                        columns_to_remove.append(key)
                    else:
                        nested_df = pd.json_normalize(jsonD, record_path=key, errors='ignore')
                        for col in nested_df.columns:
                            nested_values = nested_df.loc[:, col].tolist()
                            nested_complex = pd.DataFrame({f'{key}.{col}': [nested_values]})
                            nested_data = pd.concat([nested_data, nested_complex], axis=1)
                        columns_to_remove.append(key)
            json_detailR = pd.json_normalize(jsonD, errors='ignore')
            json_detailR.drop(columns=columns_to_remove, inplace=True)
            list_dict = pd.DataFrame()
            columns_to_remove = []
            for col in json_detailR.columns:
                if isinstance(json_detailR[col].values[0], list) and json_detailR[col].values[0]:
                    list_dict = pd.concat([list_dict, pd.json_normalize(json_detailR[col].values[0][0])], axis=1)
                    columns_to_remove.append(col)
            json_detail = json_detailR.drop(columns=columns_to_remove)
            json_detail = pd.concat([json_detail, nested_data, list_dict], axis=1)
            json_detail = json_detail.drop(columns=['wellKnownText', 'projection']) if ('wellKnownText' or 'projection') in json_detail.columns else json_detail
            json_detail.index = [f'Item_{id}']
            values = pd.concat([values, json_detail])

    else:
        id = ID
        urlDetail = f"{environment}{'/de' if owner == '50H' else ''}/api/{section}/{assetname}{'asset' if assetname in assetparts else ''}detail/{id}"
        responseD = requests.get(urlDetail, auth=(username, password), verify=False)
        jsonD = responseD.json()
        nested_data = pd.DataFrame()
        columns_to_remove = []
        for key, value in jsonD.items():
            if isinstance(value, list) and value:
                if pd.json_normalize(jsonD, record_path=key, errors='ignore').shape[0] == 1:
                    nested = pd.json_normalize(jsonD, record_path=key, errors='ignore')
                    for col in nested.columns:
                        nested.rename(columns={col: f'{key}.{col}'}, inplace=True)
                    nested_data = pd.concat([nested_data, nested], axis=1)
                    columns_to_remove.append(key)
                else:
                    nested_df = pd.json_normalize(jsonD, record_path=key, errors='ignore')
                    for col in nested_df.columns:
                        nested_values = nested_df.loc[:, col].tolist()
                        nested_complex = pd.DataFrame({f'{key}.{col}': [nested_values]})
                        nested_data = pd.concat([nested_data, nested_complex], axis=1)
                    columns_to_remove.append(key)
        json_detailR = pd.json_normalize(jsonD, errors='ignore')
        json_detailR.drop(columns=columns_to_remove, inplace=True)
        list_dict = pd.DataFrame()
        columns_to_remove = []
        for col in json_detailR.columns:
            if isinstance(json_detailR[col].values[0], list) and json_detailR[col].values[0]:
                list_dict = pd.concat([list_dict, pd.json_normalize(json_detailR[col].values[0][0])], axis=1)
                columns_to_remove.append(col)
        json_detail = json_detailR.drop(columns=columns_to_remove)
        json_detail = pd.concat([json_detail, nested_data, list_dict], axis=1)
        json_detail = json_detail.drop(columns=['wellKnownText', 'projection']) if ('wellKnownText' or 'projection') in json_detail.columns else json_detail
        json_detail.index = [f'Item_{id}']
        values = pd.concat([values, json_detail])

    print(values.T.reset_index(names='Attributes'), '\n', '\n', '\n')
    if export == 'Y':
        values.T.reset_index(names='Attributes').to_excel(path, index=False)

