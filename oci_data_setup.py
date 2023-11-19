"""
Download a dataset zipfile from Oracle Cloud Object Storage

"""

# OCI libraries
import ads
import ocifs
from ocifs import OCIFileSystem

import numpy as np
import pandas as pd

import os
from zipfile import ZipFile 

bucket_name = "" # INSERT OCI BUCKET_NAME
namespace = "" # INSERT OCI NAMESPACE
files = ["KLGrades.zip"] # replace with name of your zip file

obj_storage_url = f"oci://{bucket_name}@{namespace}/"

if "OCI_RESOURCE_PRINCIPAL_VERSION" in os.environ:
    # Use resource principal
    print("using Resource Principal for auth")
    ads.set_auth(auth="resource_principal")
else:
    # Use api_key with config file
    print("using API key for auth")
    ads.set_auth(auth="api_key") 
    
fs = OCIFileSystem(region="us-ashburn-1")

# Creating the local directory 
dirpath = f"./data/"

if not os.path.exists(dirpath):
    os.makedirs(dirpath)

for f in files:
    # Downloading the data from Object Storage using OCIFS (https://github.com/oracle/ocifs)
    if os.path.exists(os.path.join(dirpath, f)):
        if not os.path.exists(os.path.join(dirpath, f.split(".")[0])):
            with ZipFile(os.path.join(dirpath, f), 'r') as zipf:
                zipf.extractall(dirpath)
    else:
        fs.download(f"{obj_storage_url}{f}" ,os.path.join(dirpath, f))
        with ZipFile(os.path.join(dirpath, f), 'r') as zipf:
            zipf.extractall(dirpath)
    
    print("Extracted:", f)



