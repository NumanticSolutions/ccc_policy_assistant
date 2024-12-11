# GCP
from google.cloud import secretmanager

def get_gcpsecrets(project_id,
                   secret_id,
                   version_id="latest"):
    """
    Access a secret version in Google Cloud Secret Manager.

    Args:
        project_id: GCP project ID.
        secret_id: ID of the secret you want to access.
        version_id: Version of the secret (defaults to "latest").

    Returns:
        The secret value as a string.
    """
    # Create the Secret Manager client
    client = secretmanager.SecretManagerServiceClient()

    # Build the resource name of the secret version
    name = f"projects/{project_id}/secrets/{secret_id}/versions/{version_id}"

    # Access the secret version
    response = client.access_secret_version(request={"name": name})

    # Return the payload as a string
    # Note: response.payload.data is a bytes object, decode it to a string
    return response.payload.data.decode("UTF-8")

def table_exists(dataset_id, table_id):
    '''
    Function to determine if a BigQuery dataset table exists
    :param dataset_id:
    :param table_id:
    :return:
    '''
    try:
        sql = ("SELECT 1 FROM `{}.{}` LIMIT 0").format(dataset_id, table_id)
        pandas_gbq.read_gbq(sql,  project_id=project_id)
        return True
    except pandas_gbq.gbq.GenericGBQException as e:
        if "Not found" in str(e):
            return False
        else:
            raise e