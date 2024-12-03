import os

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

    dotenv_path = "../../../data/environment"
    cred_source = "local"

    def __init__(self,
                 **kwargs):
        '''
        Initialize class

        '''

        # Update any key word args
        self.__dict__.update(kwargs)

        # Get the database configuration
        self.__get_api_creds__()

    def __get_api_creds__(self):
        '''
        Method to retrieve API credentials

        '''

        # Get the dotenv configuration file
        if self.cred_source == "local":
            creds_file = ".env"
            self.apis_configs = dotenv_values(os.path.join(self.dotenv_path,
                                                           ".env"))
