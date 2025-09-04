import os
import json
from dotenv import load_dotenv
import boto3
from botocore.exceptions import NoCredentialsError, ClientError

# Load environment variables from .env file
load_dotenv()

AWS_KEY = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET = os.getenv("AWS_SECRET_ACCESS_KEY")
REGION = os.getenv("AWS_DEFAULT_REGION", "ap-south-1")
MODEL_ID = os.getenv("BEDROCK_MODEL_ID", "meta.llama3-8b-instruct-v1:0")

def create_bedrock_client():
    """Create Amazon Bedrock client using boto3."""
    try:
        client = boto3.client(
            service_name="bedrock-runtime",
            region_name=REGION,
            aws_access_key_id=AWS_KEY,
            aws_secret_access_key=AWS_SECRET
        )
        return client
    except Exception as e:
        raise RuntimeError(f"Failed to create Bedrock client: {e}")

def run_inference(prompt: str, max_len: int = 512, temperature: float = 0.7, top_p: float = 0.9):
    """Send prompt to LLaMA 3 model and get response."""
    client = create_bedrock_client()

    payload = {
        "prompt": prompt,
        "max_gen_len": max_len,
        "temperature": temperature,
        "top_p": top_p
    }

    body = json.dumps(payload)

    try:
        response = client.invoke_model(
            modelId=MODEL_ID,
            body=body,
            contentType="application/json",
            accept="application/json",
        )
    except NoCredentialsError:
        raise SystemExit("AWS credentials not found. Please check your .env file or environment variables.")
    except ClientError as e:
        raise SystemExit(f"AWS ClientError: {e}")

    raw = response["body"].read()
    obj = json.loads(raw)

    # Return generation if available
    if "generation" in obj:
        return obj["generation"]
    if "outputs" in obj:
        return obj["outputs"]

    return obj

if __name__ == "__main__":
    prompt = "Give me 3 easy dinner ideas."
    print(f"Using region: {REGION}, model: {MODEL_ID}\n")
    output = run_inference(prompt)
    print("AI Response:\n", output)
