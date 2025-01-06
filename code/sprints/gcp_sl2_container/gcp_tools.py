# GCP
from google.cloud import secretmanager
from google.cloud import storage
import google
import google.oauth2.credentials
from google.auth import compute_engine
import google.auth.transport.requests
import os

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

def upload_directory_to_gcs(local_directory, gcs_project_id,
                            gcs_bucket_name, gcs_directory):
    '''
    Function to upload a directory to Google Cloud Storage

    :param local_directory:
    :param bucket_name:
    :param gcs_directory:

    :return:
    '''

    # Initialize GCS client
    storage_client = storage.Client(project=gcs_project_id)
    bucket = storage_client.bucket(bucket_name=gcs_bucket_name)

    for root, _, files in os.walk(local_directory):
        for file_name in files:
            local_file_path = os.path.join(root, file_name)
            relative_path = os.path.relpath(local_file_path, local_directory)

            # Check if files should be stored in subdirectory of directly in bucket
            if gcs_directory == "":
                blob = bucket.blob(os.path.join(relative_path))
            else:
                blob = bucket.blob(os.path.join(gcs_directory, relative_path))

            # Upload
            blob.upload_from_filename(local_file_path)
            print(f"Uploaded {local_file_path} to gs://{gcs_bucket_name}/{gcs_directory}{relative_path}")


def download_directory_from_gcs(gcs_project_id, gcs_bucket_name,
                                gcs_directory, local_directory):
    '''
    Function to download a folder in Google Cloud Storage bucket to a local directory

    :param local_directory:
    :param bucket_name:
    :param gcs_directory:

    :return:
    '''

    # Initialize GCS client
    storage_client = storage.Client(project=gcs_project_id)
    bucket = storage_client.bucket(bucket_name=gcs_bucket_name)

    blobs = bucket.list_blobs(prefix=gcs_directory)

    for blob in blobs:
        if not blob.name.endswith("/"):  # Avoid directory blobs
            relative_path = os.path.relpath(blob.name, gcs_directory)
            local_file_path = os.path.join(local_directory, relative_path)
            os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
            blob.download_to_filename(local_file_path)
            print(f"Downloaded {blob.name} to {local_file_path}")





def idtoken_from_metadata_server(url: str, service_account_email: str):
    """
    Use the Google Cloud metadata server in the Cloud Run (or AppEngine or Kubernetes etc.,)
    environment to create an identity token and add it to the HTTP request as part of an
    Authorization header. This is an OIDC token (I think).

    Args:
        url: The url or target audience to obtain the ID token for.
            Examples: http://www.example.com
    """

    request = google.auth.transport.requests.Request()
    # Set the target audience.
    # Setting "use_metadata_identity_endpoint" to "True" will make the request use the default application
    # credentials. Optionally, you can also specify a specific service account to use by mentioning
    # the service_account_email.

    # credentials = compute_engine.IDTokenCredentials(
    #     request=request, target_audience=url,
    #     use_metadata_identity_endpoint=True
    # )

    credentials = compute_engine.IDTokenCredentials(
        request=request, target_audience=url, service_account_email=service_account_email
    )

    # Get the ID token.
    # Once you've obtained the ID token, use it to make an authenticated call
    # to the target audience.
    credentials.refresh(request)
    # print(credentials.token)
    print("Generated ID token.")