"""
Example usage of Bedrock Agent Langfuse integration with streaming support.
"""
import time
import boto3
import uuid
import json
from core.timer_lib import timer
from core import instrument_agent_invocation, flush_telemetry
import logging
from flask import Flask, render_template, request
import markdown

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__, template_folder="../templates",
            static_folder="../static")

# Read prompts from the prompts.txt file
prompts = []
try:
    with open("../prompts.txt", "r") as file:
        # Skip lines that are empty or comments (starting with //)
        prompts = [line.strip() for line in file if line.strip()
                   and not line.startswith("//")]
except Exception as e:
    print(f"Error reading prompts file: {e}")


@instrument_agent_invocation
def invoke_bedrock_agent(
    inputText: str, agentId: str, agentAliasId: str, sessionId: str, **kwargs
):
    """Invoke a Bedrock Agent with instrumentation for Langfuse."""
    # Create Bedrock client
    bedrock_rt_client = boto3.client("bedrock-agent-runtime")
    use_streaming = kwargs.get("streaming", False)
    invoke_params = {
        "inputText": inputText,
        "agentId": agentId,
        "agentAliasId": agentAliasId,
        "sessionId": sessionId,
        "enableTrace": True,  # Required for instrumentation
    }

    # Add streaming configurations if needed
    if use_streaming:
        invoke_params["streamingConfigurations"] = {
            "applyGuardrailInterval": 10,
            "streamFinalResponse": use_streaming,
        }
    response = bedrock_rt_client.invoke_agent(**invoke_params)
    return response


def process_streaming_response(stream):
    """Process a streaming response from Bedrock Agent."""
    full_response = ""
    try:
        for event in stream:
            # Convert event to dictionary if it's a botocore Event object
            event_dict = (
                event.to_response_dict()
                if hasattr(event, "to_response_dict")
                else event
            )
            if "chunk" in event_dict:
                chunk_data = event_dict["chunk"]
                if "bytes" in chunk_data:
                    output_bytes = chunk_data["bytes"]
                    # Convert bytes to string if needed
                    if isinstance(output_bytes, bytes):
                        output_text = output_bytes.decode("utf-8")
                    else:
                        output_text = str(output_bytes)
                    full_response += output_text
    except Exception as e:
        print(f"\nError processing stream: {e}")
    return full_response


def agentInteraction(prompt):
    import os
    import base64
    start = time.time()
    with open('config.json', 'r') as config_file:
        config = json.load(config_file)

    # For Langfuse specifically but you can add any other observability provider:
    if "langfuse" in config:
        os.environ["OTEL_SERVICE_NAME"] = 'bedrock-energy-agent'
        os.environ["DEPLOYMENT_ENVIRONMENT"] = config["langfuse"]["environment"]
        project_name = config["langfuse"]["project_name"]
        environment = config["langfuse"]["environment"]
        langfuse_public_key = config["langfuse"]["langfuse_public_key"]
        langfuse_secret_key = config["langfuse"]["langfuse_secret_key"]
        # langfuse_api_url = "https://us.cloud.langfuse.com"
        langfuse_api_url = config["langfuse"]["langfuse_api_url"]

        # Create auth header
        auth_token = base64.b64encode(
            f"{langfuse_public_key}:{langfuse_secret_key}".encode()
        ).decode()

    # Set OpenTelemetry environment variables
    # os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] = "https://otlp.nr-data.net:443"
    # os.environ["OTEL_EXPORTER_OTLP_HEADERS"] = "api-key=NEW_RELIC_LICENSE_KEY"

    # Agent configuration
    agentId = config["agent"]["agentId"]
    agentAliasId = config["agent"]["agentAliasId"]
    sessionId = f"session-{int(time.time())}"

    # User information
    userId = config["user"]["userId"]
    agent_model_id = config["user"]["agent_model_id"]

    # Tags for filtering in Langfuse
    tags = ["bedrock-agent", "example", "development"]

    # Generate a custom trace ID
    trace_id = str(uuid.uuid4())

    # Prompt
    # question = config["question"]["question"]  # your prompt to the agent
    question = prompt
    # Set streaming mode: True for streaming final response, False for non-streaming
    streaming = False

    # Single invocation that works for both streaming and non-streaming
    response = invoke_bedrock_agent(
        inputText=question,
        agentId=agentId,
        agentAliasId=agentAliasId,
        sessionId=sessionId,
        show_traces=False,
        SAVE_TRACE_LOGS=False,
        userId=userId,
        tags=tags,
        trace_id=trace_id,
        project_name=project_name,
        environment=environment,
        langfuse_public_key=langfuse_public_key,
        langfuse_secret_key=langfuse_secret_key,
        langfuse_api_url=langfuse_api_url,
        streaming=streaming,
        model_id=agent_model_id,
    )

    # Handle the response appropriately based on streaming mode
    resp = response
    if isinstance(response, dict) and "error" in response:
        print(f"\nError: {response['error']}")
    elif streaming and isinstance(response, dict) and "completion" in response:
        print("\n🤖 Agent response (streaming):")
        if "extracted_completion" in response:
            print(response["extracted_completion"])
        else:
            process_streaming_response(response["completion"])
    else:
        # Non-streaming response
        print("\n🤖 Agent response:")
        if isinstance(response, dict) and "extracted_completion" in response:
            print(response["extracted_completion"])
            resp = response["extracted_completion"]
        elif (
            isinstance(response, dict)
            and "completion" in response
            and hasattr(response["completion"], "__iter__")
        ):
            print("Processing completion:")
            full_response = process_streaming_response(response["completion"])
            resp = full_response
            print(f"\nFull response: {full_response}")
        else:
            print("Raw response:")
            print(f"{response}")

    # Flush telemetry data
    flush_telemetry()
    timer.reset_all()

    return resp


@app.route("/")
def home():
    return render_template("index.html", prompts=prompts)


@app.route("/prompt", methods=["POST"])
def prompt():
    input_prompt = request.form.get("input")
    output_prompt = agentInteraction(input_prompt)
    html_output = markdown.markdown(output_prompt)
    # html_output = output_prompt

    return render_template("index.html", input=input_prompt, output=html_output, prompts=prompts)


# make the server publicly available via port 5004
# flask --app levelsix.py run --host 0.0.0.0 --port 5004
if __name__ == '__main__':
    app.run(host="0.0.0.0", debug=True, port=5004)
