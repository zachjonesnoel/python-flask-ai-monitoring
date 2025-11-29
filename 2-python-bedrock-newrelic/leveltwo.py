# import the New Relic Python Agent
# import newrelic.agent
import os
from flask import Flask, render_template, request
import boto3
from botocore.exceptions import ClientError
import json
import markdown

# initialize the New Relic Python agent
# newrelic.agent.initialize('newrelic.ini')

# Create a Bedrock Runtime client in the AWS Region you want to use.
AWS_ACCESS_KEY = os.environ.get("AWS_ACCESS_KEY")
AWS_SECRET_KEY = os.environ.get("AWS_SECRET_KEY")
client = boto3.client("bedrock-runtime", region_name="us-east-1", aws_access_key_id=AWS_ACCESS_KEY, aws_secret_access_key=AWS_SECRET_KEY)

# Set the model ID, e.g., Titan Text Premier.¨
# model_id = "amazon.titan-text-lite-v1"
# model_id = "anthropic.claude-3-sonnet-20240229-v1:0"
# model_id = "anthropic.claude-3-5-sonnet-20240620-v1:0"
# model_id = "anthropic.claude-v2"
# model_id = "anthropic.claude-v2:1"
# model_id = "anthropic.claude-3-haiku-20240307-v1:0"
# model_id = "ai21.jamba-1-5-mini-v1:0"
# model_id = "meta.llama3-8b-instruct-v1:0"
# model_id = "mistral.mistral-7b-instruct-v0:2"
# model_id = "deepseek.r1-v1:0"
# model_id="amazon.nova-micro-v1:0"
model_id = os.environ["MODEL"]

app = Flask(__name__, template_folder="../templates",
            static_folder="../static")

prompts = []
available_models = [
        "meta.llama3-8b-instruct-v1:0",
        "mistral.mistral-7b-instruct-v0:2"
]
try:
    with open("../prompts.txt", "r") as file:
        # Skip lines that are empty or comments (starting with //)
        prompts = [line.strip() for line in file if line.strip()
                   and not line.startswith("//")]
except Exception as e:
    print(f"Error reading prompts file: {e}")

def chatCompletion(prompt):
    print("prompt: "+prompt)
    # Format the request payload using the model's native structure.
    try:
        if model_id == "amazon.titan-text-lite-v1":
            native_request = json.dumps({
                "inputText": prompt,
                "textGenerationConfig": {
                    "maxTokenCount": 512,
                    "temperature": 0.5,
                },
            })
            response = client.invoke_model(modelId=model_id, body=native_request)
            model_response = json.loads(response["body"].read())
            response_text = model_response["content"][0]["text"]
        elif model_id == "mistral.mistral-7b-instruct-v0:2":
            formatted_prompt = f"""<s>[INST]{prompt} [/INST]"""
            native_request = json.dumps({
                "prompt":  formatted_prompt,
                "max_tokens": 400,
                "temperature": 0.7,
                "top_p": 0.7,
                "top_k": 50
            })
            response = client.invoke_model(modelId=model_id, body=native_request)
            print("raw response: "+str(response))
            model_response = json.loads(response["body"].read())
            print("model response: "+str(model_response))
            response_text = model_response["outputs"][0]["text"] 
        elif model_id == 'meta.llama3-8b-instruct-v1:0':
            formatted_prompt = f"""
                <|begin_of_text|><|start_header_id|>user<|end_header_id|>
                {prompt}
                <|eot_id|>
                <|start_header_id|>assistant<|end_header_id|>
                """
            response = client.invoke_model(
                modelId=model_id,
                body=json.dumps({
                    "prompt": formatted_prompt,
                    "max_gen_len": 512,
                    "temperature": 0.5,
                })
            )
            model_response = json.loads(response["body"].read())
            print("model response: "+str(model_response))
            response_text = model_response["generation"]
        else:
            native_request = json.dumps({
                "inputText": prompt,
                "textGenerationConfig": {
                    "maxTokenCount": 512,
                    "temperature": 0.5,
                },
            })
            response = client.invoke_model(modelId=model_id, body=native_request)
            # Decode the response body.
            model_response = json.loads(response["body"].read())
            response_text = model_response["content"][0]["text"]

    except (ClientError, Exception) as e:
        print(f"ERROR: Can't invoke '{model_id}'. Reason: {e}")
        exit(1)
    print(f"response: {response_text}")
    return response_text

@app.route("/")
def home():
    # Get available models
    available_models = [
        "meta.llama3-8b-instruct-v1:0",
        "mistral.mistral-7b-instruct-v0:2",
    ]
    default_model = os.environ.get("MODEL", "meta.llama3-8b-instruct-v1:0")
    return render_template("index.html", models=available_models, selected_model=default_model)



@app.route("/prompt", methods=["POST"])
def prompt():
    input_prompt = request.form.get("input")
    selected_model = request.form.get("model")
    if selected_model:
        model_id = selected_model
    output_prompt = chatCompletion(input_prompt)
    html_output = markdown.markdown(output_prompt)
    available_models = [
        "meta.llama3-8b-instruct-v1:0",
        "mistral.mistral-7b-instruct-v0:2",
    ]
    default_model = os.environ.get("MODEL", "meta.llama3-8b-instruct-v1:0")
    return render_template("index.html", output=html_output, models=available_models, selected_model=default_model)

# make the server publicly available via port 5004
# flask --app levelsix.py run --host 0.0.0.0 --port 5004
if __name__ == '__main__':
    app.run(host="0.0.0.0", debug=True, port=5004)
