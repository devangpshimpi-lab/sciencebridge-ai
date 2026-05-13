from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import os

app = Flask(__name__, template_folder="templates", static_folder="static")
CORS(app)

# =========================
# HOME ROUTE (FRONTEND)
# =========================
@app.route("/")
def home():
    return render_template("index.html")


# =========================
# MODEL LOAD (FIXED SAFE PATH)
# =========================
MODEL_PATH = os.environ.get("MODEL_PATH", "./student_reasoning_model")

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

print("MODEL LOADED SUCCESSFULLY")


# =========================
# INFERENCE FUNCTION
# =========================
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
        logits = outputs.logits

        # FIX: safer handling (works for 1D or 2D output)
        if logits.shape[-1] == 1:
            score = torch.sigmoid(logits).item()
        else:
            score = torch.softmax(logits, dim=-1)[0][1].item()

    score = max(0.0, min(1.0, float(score)))
    return round(score, 4)


# =========================
# API ROUTE
# =========================
@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.get_json(force=True)
    text = data.get("text", "")

    if not text.strip():
        return jsonify({
            "error": "Empty input"
        }), 400

    # SHORT INPUT CHECK
    if len(text.split()) < 8:
        return jsonify({
            "score": 0.10,
            "logic": "Input is too brief to contain meaningful argumentative structure.",
            "evidence": "No supporting evidence or justification detected.",
            "reasoning": "The response appears conversational rather than analytical.",
            "communication": "Please provide a complete argument or explanation for deeper analysis."
        })

    # MODEL SCORE
    score = analyze_argument(text)

    # RESPONSE MAP
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


# =========================
# RUN SERVER (PRODUCTION SAFE)
# =========================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    app.run(host="0.0.0.0", port=port)
