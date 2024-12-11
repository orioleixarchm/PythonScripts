########################################################################################################################
# Author: Oriol Eixarch                                                                                                #
# Contact: Oriol.EixarchIMejias@externel.be                                                                            #
# Function: Infrastructure Business Analyst                                                                            #
# Creation date: 12/06/2024                                                                                            #
# Description: Data retrieval from IH's APIs (Elia + 50H), normalization of JSONs in tabular format and excel export.  #
# Note: To know the AH nomenclature is included in them, refer to the links found in the following page                #
# https://inthubacc.belgrid.net/assethub_v1.0.acc/swagger/index.html                                                   #
########################################################################################################################                                                                                                                    #
# README                                                                                                               #
# The section "Parameter definition" asks the user to provide parameters regarding the environment to work with, their #
# credentials, which section of the AH would they want to work with, which asset/activity or specific subpart, for     #
# which branch, Elia (Belgium) or 50H (Germany), it also provides the option to export the data into Excel provided    #
# that the users introduces a valid path. the user can select a particular ID to retrieve, if left blank the user will #
# be given the option to retrieve a number of his or her choice of IDs including all of them.                          #
# The section "Connecting to IDs API" retrieves all the IDs and only the IDs for the selected subtype of asset or      #
# activity within the subsection.                                                                                      #
# The section "Reading the details JSON based on ID" will retrieve the subselection items' details (all attributes)    #
# looping over a list of IDs that will be determined by the desired amount of IDs introduced by the user. It will also #
# export the data to an excel file following the user defined export path if he or she chooses to export the data.     #
########################################################################################################################

#Keys
#username = 'IU3309@belgrid.net'
#password = 'Elnumero15*'
#path = C:/Users/IU3309/OneDrive - ELIA GROUP/Desktop/Messages APMO/raw/PLACEHOLDER.xlsx

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
sectionR = int(input("AH Section (1 for Topology, 2 for Location, 3 for Physicalobject, 4 for Activity): ")) or 3
sectionmap = {1:'Topology', 2:'Location', 3:'Physicalobject', 4:'Physicalobject'}
section = sectionmap[sectionR]
assetname = input("Asset Type e.g circuitbreaker / Location e.g geographicalsite / Activity e.g pruningpolicy / Topology e.g line  (AH nomenclature): ") or "circuitbreaker"
owner = input("Elia of 50Hz Asset (Elia/50H)? ") or 'Elia'
export = input("Would you like tot export the data (Y/N)? ") or "N"
path = input("Output path and file name (needs to end with .xlsx and use / instead of \ e.g C:/Users/Username/Assets.xlsx): ") if export=='Y' else ''
assetpartsBe = ['rectifiercontroller','currenttransformer','rectifier','meter','powertransformer','disconnector','gascompartment','baycontroller',
              'protectionrelay','alternator','motor','generatorcontroller','generatorbattery','fuse','voltagetransformer','inductor','resistor',
              'compressor','surgearester','circuitbreaker','dieselgenerator','faultrecorder']
assetpartsDe = ['dieselgenerator','busbar','voltagetransformer','earthingswitch','powertransformer', 'disconnector','circuitbreaker','shuntreactor',
                'currenttransformer','resistor','inductor','surgearester']

# Some assets APIs contain the word asset, whereas others do not, therefore sometimes we will need to include it in the url, those assets can be found in 'assetpartsDe' and 'assetpartsBe' lists.

assetparts = assetpartsBe if owner=='Elia' else assetpartsDe
urlID = f"{environment}{'/de' if owner=='50H' else ''}/api/{section}/all{assetname}{'asset' if assetname in assetparts else ''}"
ID = input("Do you want to retrieve data for a particular individual ID? If so write the ID, else leave this field blank: ")

#Connecting to IDs API
response=requests.get(urlID, auth=(username,password), verify=False)

#Parsing JSON ID
json=response.json()
json_n = pd.json_normalize(json)

#Reading the details JSON based on ID
if not ID:
    IDn = int(input("How many IDs needed? for all IDs write 0, blank for just 1: ") or 1)
    ID = json_n.iloc[:,-1] if IDn == 0 else json_n.iloc[:IDn,-1]
    values = pd.DataFrame()
    if IDn != 1:
        for id in ID:
            urlDetail = f"{environment}{'/de' if owner=='50H' else ''}/api/{section}/{assetname}{'asset' if assetname in assetparts else ''}detail/{id}"
            responseD = requests.get(urlDetail, auth=(username, password), verify=False)
            jsonD = responseD.json()
            json_detail = pd.json_normalize(jsonD)
            json_detail.index = [f'Item_{id}']
            values = pd.concat([values, json_detail])

        print(values.T.reset_index(names='Attributes'),'\n','\n','\n')
    elif IDn == 1:
        id = ID.values[0]
        print(id)
        urlDetail = f"{environment}{'/de' if owner=='50H' else ''}/api/{section}/{assetname}{'asset' if assetname in assetparts else ''}detail/{id}"
        responseD = requests.get(urlDetail, auth=(username, password), verify=False)
        jsonD = responseD.json()
        values = pd.json_normalize(jsonD)
        values.index = [f'Item_{id}']
        print(values.T.reset_index(names='Attributes'),'\n','\n','\n')
else:
    id = ID
    print(id)
    urlDetail = f"{environment}{'/de' if owner=='50H' else ''}/api/{section}/{assetname}{'asset' if assetname in assetparts else ''}detail/{id}"
    responseD = requests.get(urlDetail, auth=(username, password), verify=False)
    jsonD = responseD.json()
    values = pd.json_normalize(jsonD)
    values.index = [f'Item_{id}']
    print(values.T.reset_index(names='Attributes'), '\n', '\n', '\n')

if export == 'Y':
    values.T.reset_index(names='Attributes').to_excel(path, index=False)

