import os
from flask import Flask, render_template, request
import google.generativeai as genai

genai.configure(api_key=os.environ["API_KEY"])

GEMINI_MODEL = "gemini-1.5-flash"
# GEMINI_MODEL = "gemini-1.5-flash-8b"
# GEMINI_MODEL = "gemini-1.5-pro"
# GEMINI_MODEL = "gemini-1.0-pro"

app = Flask(__name__, template_folder="../templates",
            static_folder="../static")

# taking the input from the user and returning the response from Gemini


def chatCompletion(prompt):
    model = genai.GenerativeModel(GEMINI_MODEL)
    response = model.generate_content(
        prompt,
        generation_config=genai.types.GenerationConfig(
            temperature=1.0,
        ),
    )
    responseText = ""
    if response.candidates:
        if response.candidates[0].content.parts:
            if response.candidates[0].content.parts[0].text:
                responseText = response.candidates[0].content.parts[0].text
    return responseText


@app.route("/")
def main():
    return render_template("index.html")


@app.route("/prompt", methods=["POST"])
def prompt():
    input_prompt = request.form.get("input")
    # call the function - chatCompletion and pass the input from the user
    output_prompt = chatCompletion(input_prompt)
    return render_template("index.html", output=output_prompt)


# make the server publicly available via port 5002
# flask --app levelfive.py run --host 0.0.0.0 --port 5002
if __name__ == '__main__':
    app.run(host="0.0.0.0", debug=True, port=5002)
