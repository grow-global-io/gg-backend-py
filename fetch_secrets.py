import boto3
import json
from botocore.exceptions import ClientError, NoCredentialsError
import os


def get_secret(secret_name, region_name="us-east-1"):
    """
    Retrieve a secret from AWS Secrets Manager.
    
    Args:
        secret_name (str): The name of the secret in AWS Secrets Manager
        region_name (str): AWS region where the secret is stored (default: us-east-1)
    
    Returns:
        dict: The secret values as a dictionary
    
    Raises:
        Exception: If unable to retrieve the secret
    """
    
    # Create a Secrets Manager client
    session = boto3.session.Session()
    client = session.client(
        service_name='secretsmanager',
        region_name=region_name
    )
    
    try:
        get_secret_value_response = client.get_secret_value(
            SecretId=secret_name
        )
    except ClientError as e:
        # For a list of exceptions thrown, see
        # https://docs.aws.amazon.com/secretsmanager/latest/apireference/API_GetSecretValue.html
        raise e
    except NoCredentialsError:
        raise Exception("AWS credentials not found. Please configure your AWS CLI or environment variables.")
    
    # Decrypts secret using the associated KMS key.
    secret = get_secret_value_response['SecretString']
    
    try:
        # Try to parse as JSON
        return json.loads(secret)
    except json.JSONDecodeError:
        # If it's not JSON, return as a simple key-value pair
        # Assuming the secret name is the key and the secret value is the value
        return {secret_name: secret}


def load_secrets_as_env(secret_name, region_name="us-east-1"):
    """
    Load secrets from AWS Secrets Manager and set them as environment variables.
    
    Args:
        secret_name (str): The name of the secret in AWS Secrets Manager
        region_name (str): AWS region where the secret is stored (default: us-east-1)
    
    Returns:
        dict: The loaded secrets
    """
    try:
        secrets = get_secret(secret_name, region_name)
        
        # Set each secret as an environment variable
        for key, value in secrets.items():
            os.environ[key] = str(value)
        
        print(f"Successfully loaded {len(secrets)} secrets from AWS Secrets Manager")
        return secrets
        
    except Exception as e:
        print(f"Error loading secrets: {str(e)}")
        raise e


if __name__ == "__main__":
    # Test the function
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python fetch_secrets.py <secret_name> [region_name]")
        print("Example: python fetch_secrets.py my-app-secrets us-west-2")
        sys.exit(1)
    
    secret_name = sys.argv[1]
    region_name = sys.argv[2] if len(sys.argv) > 2 else "us-east-1"
    
    try:
        secrets = get_secret(secret_name, region_name)
        print(f"Successfully retrieved secrets from '{secret_name}':")
        for key in secrets.keys():
            print(f"  - {key}")
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)
