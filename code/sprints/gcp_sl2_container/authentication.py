import os
import json

# Environment variables for API credential storage
# import dotenv
from dotenv import dotenv_values


class ApiAuthentication:
    '''
    Class to store API authentication credentials in an object.

    Attributes

        dotenv_path: Path to .env file used by dotenv
        cred_source: Credential source:
            'local': Credentials come from a local .env file



    '''

    def __init__(self,
                 **kwargs):
        '''
        Initialize class

        '''

        self.dotenv_path = "../../../data/environment"
        self.cred_source = "dotenv"
        self.service_acct_file = "eternal-bongo-435614-b9-bf6a5e630e44.json"

        # Update any key word args
        self.__dict__.update(kwargs)

        # Get the database configuration
        self.__get_api_creds__()

    def __get_api_creds__(self):
        '''
        Method to retrieve API credentials

        '''

        self.apis_configs = {}

        # Get the dotenv configuration file
        if self.cred_source == "dotenv":
            creds_file = ".env"
            self.apis_configs = dotenv_values(os.path.join(self.dotenv_path,
                                                           creds_file))

        elif self.cred_source == "service_acct":
            self.apis_configs["service_acct"] = json.load(open(os.path.join(self.creds_path,
                                                self.creds_file)))



