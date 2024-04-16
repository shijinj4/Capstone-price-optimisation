import json
import boto3
 
# Define the SageMaker runtime
sagemaker_runtime = boto3.client('sagemaker-runtime')
 
def lambda_handler(event, context):
    # Parse the JSON data from the request body
    print(event)
    values = json.loads(event['body'])
    print(values)
    response = sagemaker_runtime.invoke_endpoint(
        EndpointName='Custom-sklearn-model-2024-04-09-14-10-56', # Replace with your SageMaker endpoint
        ContentType='application/json',
        Body=json.dumps([values]) # Adjust the structure to match your model's expected input
    )
    print(response)
    print ("response_payload: {}".format(response))
    t = response['Body']
    j = t.read()
    print (j)
    # Create a response object with CORS headers
    return {
        'statusCode': 200,
        'headers': {
            'Access-Control-Allow-Origin': '*',  # Allow requests from all domains
            'Access-Control-Allow-Headers': 'Content-Type',  # Allow Content-Type header
            'Access-Control-Allow-Methods': 'OPTIONS,POST'  # Allow OPTIONS and POST methods
        },
        'body': j  # Send the SageMaker response back to the client
    }