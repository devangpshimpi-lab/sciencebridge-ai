from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

app = Flask(__name__)
CORS(app)

# LOAD MODEL
model_path = "./student_reasoning_model_4"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model.to(device)

print("MODEL LOADED SUCCESSFULLY")


# MODEL INFERENCE FUNCTION
def analyze_argument(text):

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=256
    )

    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        score = outputs.logits.squeeze().item()

    score = max(0, min(1, score))

    return round(score, 4)


# API ROUTE
@app.route("/analyze", methods=["POST"])
def analyze():

    data = request.json
    text = data["text"]

    # SHORT / NON-ARGUMENT DETECTION
    if len(text.split()) < 8:
        return jsonify({
            "score": 0.10,
            "logic": "Input is too brief to contain meaningful argumentative structure.",

            "evidence": "No supporting evidence or justification detected.",

            "reasoning": "The response appears conversational rather than analytical.",

            "communication": "Please provide a complete argument or explanation for deeper analysis."
        })

    # GET MODEL SCORE
    score = analyze_argument(text)

    # SCORE INTERPRETATION
    if score >= 0.95:

        result = {
            "score": score,
            "logic": "Near-expert logical coherence with exceptionally strong analytical structure.",
            "evidence": "Evidence integration is highly persuasive and academically sophisticated.",
            "reasoning": "Advanced multi-layered reasoning and critical thinking depth detected.",
            "communication": "Exceptional communication clarity with research-level articulation."
        }

    elif score >= 0.90:

        result = {
            "score": score,
            "logic": "Exceptional logical coherence and advanced analytical structure.",
            "evidence": "Strong evidence integration and persuasive support.",
            "reasoning": "Highly sophisticated reasoning depth detected.",
            "communication": "Academic-level clarity and communication quality."
        }

    elif score >= 0.80:

        result = {
            "score": score,
            "logic": "Strong logical progression and structured reasoning.",
            "evidence": "Good supporting justification and analytical framing.",
            "reasoning": "Clear critical thinking with developed ideas.",
            "communication": "Well-organized and understandable communication."
        }

    elif score >= 0.70:

        result = {
            "score": score,
            "logic": "Moderate logical consistency with some structural weaknesses.",
            "evidence": "Basic supporting ideas are present.",
            "reasoning": "Reasoning depth is partially developed.",
            "communication": "Mostly clear but lacks stronger analytical precision."
        }

    elif score >= 0.60:

        result = {
            "score": score,
            "logic": "Limited logical flow and weak argumentative structure.",
            "evidence": "Insufficient supporting evidence detected.",
            "reasoning": "Surface-level reasoning patterns.",
            "communication": "Communication clarity needs improvement."
        }

    elif score >= 0.40:

        result = {
            "score": score,
            "logic": "Weak logical consistency with unclear argumentative progression.",
            "evidence": "Minimal supporting evidence or justification detected.",
            "reasoning": "Reasoning depth remains underdeveloped.",
            "communication": "Communication structure lacks clarity and precision."
        }

    else:

        result = {
            "score": score,
            "logic": "Weak logical coherence and fragmented reasoning.",
            "evidence": "Very limited argumentative support.",
            "reasoning": "Minimal critical thinking depth detected.",
            "communication": "Poor structural clarity and communication."
        }

    return jsonify(result)


# RUN SERVER
if __name__ == "__main__":
    app.run(debug=True, port=5001)
