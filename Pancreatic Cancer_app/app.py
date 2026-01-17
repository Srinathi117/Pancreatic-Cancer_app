import os
import io

from flask import Flask, request, jsonify, render_template_string
from PIL import Image

import torch
from torch import nn
from torchvision import models, transforms
from torch_geometric.nn import GCNConv

# ========= Gemini client for chatbot =========
import google.generativeai as genai

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError(
        "GEMINI_API_KEY environment variable is not set. "
        "Set it with:  setx GEMINI_API_KEY \"your-key-here\"  in PowerShell, "
        "then restart VS Code."
    )

genai.configure(api_key=GEMINI_API_KEY)

# ================== CONFIG ==================
MODEL_PATH = "pancreas_gnn.pth"
IMG_SIZE = 224

# ================== DEVICE ==================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ================== TRANSFORMS ==================
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

# ================== GNN MODEL ==================
class PancreaticGCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_classes, dropout=0.5):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, num_classes)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = nn.functional.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x

# ================== LOAD MODEL FROM .PTH ==================
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"❌ Cannot find {MODEL_PATH}. "
        f"Make sure it is in the same folder as app.py."
    )

print(f"Loading model checkpoint from {MODEL_PATH}...")
checkpoint = torch.load(MODEL_PATH, map_location=device)

# CNN backbone
backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
backbone.fc = nn.Identity()
backbone.load_state_dict(checkpoint["backbone_state_dict"])
backbone = backbone.to(device)
backbone.eval()

# GNN
in_channels = 512
hidden_channels = 128
class_names = checkpoint["class_names"]
num_classes = len(class_names)

model = PancreaticGCN(
    in_channels=in_channels,
    hidden_channels=hidden_channels,
    num_classes=num_classes,
    dropout=0.0
).to(device)
model.load_state_dict(checkpoint["gcn_state_dict"])
model.eval()

print("Class names:", class_names)

CANCER_CLASS_INDICES = [
    i for i, c in enumerate(class_names)
    if "cancer" in c.lower() or "tumor" in c.lower()
]
print("Cancer class indices:", CANCER_CLASS_INDICES)

# ================== PREDICTION FUNCTION ==================
def predict_image_pytorch(pil_img: Image.Image):
    img_t = transform(pil_img).unsqueeze(0).to(device)  # [1, 3, H, W]

    with torch.no_grad():
        feat = backbone(img_t)  # [1, 512]
        edge_index_new = torch.tensor([[0], [0]], dtype=torch.long).to(device)
        out = model(feat, edge_index_new)  # [1, num_classes]
        probs = torch.softmax(out, dim=1)[0]

        pred_idx = int(torch.argmax(probs))
        pred_class = class_names[pred_idx]
        conf = float(probs[pred_idx])
        is_cancer = pred_idx in CANCER_CLASS_INDICES

    return pred_class, conf, is_cancer

# ================== FLASK APP ==================
app = Flask(__name__)

INDEX_HTML = """
<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8">
  <title>Pancreatic Cancer Prediction</title>
  <style>
    body { 
    font-family: Arial, sans-serif;
      margin: 0;
      background: url('https://shifaahospital.org/wp-content/uploads/2024/08/uthsa-cc-mobile-cancer-diagnosis-1982x1552-1.jpg') 
        no-repeat center center fixed;
      background-size: cover;
    }

    .overlay {
      background: rgba(255, 255, 255, 0.88);
      min-height: 100vh;
      padding: 30px 0;
    }

    .container {
      max-width: 900px; margin: 40px auto; background: #fff;
      padding: 20px 30px; border-radius: 8px;
      box-shadow: 0 4px 12px rgba(0,0,0,0.08);
    }

    h1 { text-align: center; }

    form { margin-top: 10px; display: flex; flex-direction: column; gap: 10px; }
    button {
      padding: 10px; border: none; border-radius: 4px;
      background: #007bff; color: white; cursor: pointer;
    }
    button:hover { background: #0056b3; }

    .result, .error {
      margin-top: 10px; padding: 10px; border-radius: 6px;
      font-size: 0.9rem;
    }
    .result { background: #eafaf1; border: 1px solid #c6f1d8; }
    .error { background: #fdecea; border: 1px solid #f5c2c0; color: #b3261e; }
    .hidden { display: none; }

    /* ===== Floating chatbot styles ===== */

    .chat-launcher {
      position: fixed;
      right: 20px;
      bottom: 20px;
      width: 60px;
      height: 60px;
      border-radius: 50%;
      border: none;
      background: linear-gradient(135deg, #6a11cb, #2575fc);
      color: #fff;
      font-size: 26px;
      display: flex;
      align-items: center;
      justify-content: center;
      box-shadow: 0 4px 12px rgba(0,0,0,0.3);
      cursor: pointer;
      z-index: 999;
    }

    .chat-widget {
      position: fixed;
      right: 20px;
      bottom: 90px;
      width: 320px;
      max-height: 450px;
      background: #ffffff;
      border-radius: 12px;
      box-shadow: 0 12px 30px rgba(0,0,0,0.2);
      display: none;
      flex-direction: column;
      overflow: hidden;
      z-index: 999;
      font-size: 0.9rem;
    }

    .chat-widget.open {
      display: flex;
    }

    .chat-header {
      background: linear-gradient(135deg, #6a11cb, #2575fc);
      color: #fff;
      padding: 10px 12px;
      display: flex;
      align-items: center;
      justify-content: space-between;
    }

    .chat-header-title {
      font-weight: bold;
      font-size: 0.95rem;
    }

    .chat-close {
      border: none;
      background: transparent;
      color: #fff;
      font-size: 18px;
      cursor: pointer;
    }

    .chat-messages {
      padding: 8px;
      flex: 1;
      overflow-y: auto;
      background: #fafafa;
    }

    .msg {
      margin-bottom: 6px;
      padding: 6px 8px;
      border-radius: 6px;
      max-width: 90%;
      word-wrap: break-word;
    }

    .msg.bot {
      background: #f1f1f1;
      color: #333;
      align-self: flex-start;
    }

    .msg.user {
      background: #007bff;
      color: #fff;
      align-self: flex-end;
      margin-left: auto;
    }

    .chat-input-row {
      display: flex;
      padding: 8px;
      border-top: 1px solid #eee;
      background: #fff;
      gap: 6px;
    }

    .chat-input-row input {
      flex: 1;
      padding: 8px;
      border-radius: 4px;
      border: 1px solid #ccc;
      font-size: 0.9rem;
    }

    .chat-input-row button {
      padding: 8px 10px;
      font-size: 0.85rem;
    }

    .chat-note {
      font-size: 0.7rem;
      color: #666;
      padding: 6px 8px;
      border-top: 1px solid #eee;
      background: #fafafa;
    }
    
  </style>
</head>
<body>
  <div class="container">
    <h1>Pancreatic Cancer Prediction</h1>

    <p>Upload a CT/MRI scan to help the AI provide an early-insight prediction.Always seek medical experts for real diagnosis.</p>
    <form id="upload-form">
      <label>Select image:</label>
      <input type="file" id="image" name="image" accept="image/*" required>
      <button type="submit">Predict</button>
    </form>

    <div id="result" class="result hidden">
      <h3>Prediction</h3>
      <p id="status-text"></p>
      <p id="details-text"></p>
    </div>

    <div id="error" class="error hidden"></div>
  </div>

  <!-- ===== Floating Chatbot UI ===== -->
  <button class="chat-launcher" id="chat-launcher">💬</button>

  <div class="chat-widget" id="chat-widget">
    <div class="chat-header">
      <div class="chat-header-title">AI Assistant</div>
      <button class="chat-close" id="chat-close">×</button>
    </div>
    <div class="chat-messages" id="chat-messages">
      <div class="msg bot">
        👋 Hello! I’m your AI assistant here to help you understand pancreatic cancer awareness, CT/MRI imaging and this AI model.
I can answer general questions but for real medical advice or report interpretation, always speak with a healthcare professional.
      </div>
    </div>
    <form id="chat-form" class="chat-input-row">
      <input type="text" id="chat-input" placeholder="Type your question..." required>
      <button type="submit">Send</button>
    </form>
    <div class="chat-note">
      ⚠️ This AI system offers general pancreatic health information. For personalized diagnosis or care, please always consult a doctor.
    </div>
  </div>

<script>
  // ===== Prediction JS =====
  const form = document.getElementById("upload-form");
  const resultDiv = document.getElementById("result");
  const statusText = document.getElementById("status-text");
  const detailsText = document.getElementById("details-text");
  const errorDiv = document.getElementById("error");

  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    resultDiv.classList.add("hidden");
    errorDiv.classList.add("hidden");
    errorDiv.textContent = "";

    const fileInput = document.getElementById("image");
    if (!fileInput.files.length) {
      errorDiv.textContent = "Please select an image.";
      errorDiv.classList.remove("hidden");
      return;
    }

    const formData = new FormData();
    formData.append("image", fileInput.files[0]);

    try {
      const resp = await fetch("/predict", { method: "POST", body: formData });
      const data = await resp.json();

      if (!resp.ok) throw new Error(data.error || "Prediction failed");

      resultDiv.classList.remove("hidden");
      if (data.is_cancer) {
        statusText.textContent = "⚠️ The model predicts: CANCER";
      } else {
        statusText.textContent = "✅ The model predicts: NO CANCER (Normal)";
      }
      detailsText.textContent =
        "Class: " + data.pred_class +
        " | Confidence: " + (data.confidence * 100).toFixed(2) + "%";
    } catch (err) {
      errorDiv.textContent = err.message;
      errorDiv.classList.remove("hidden");
    }
  });

  // ===== Floating Chatbot JS =====
  const launcher = document.getElementById("chat-launcher");
  const widget = document.getElementById("chat-widget");
  const closeBtn = document.getElementById("chat-close");
  const chatForm = document.getElementById("chat-form");
  const chatInput = document.getElementById("chat-input");
  const chatMessages = document.getElementById("chat-messages");

  launcher.addEventListener("click", () => {
    widget.classList.toggle("open");
  });

  closeBtn.addEventListener("click", () => {
    widget.classList.remove("open");
  });

  function addMessage(text, sender) {
    const div = document.createElement("div");
    div.classList.add("msg", sender);
    div.textContent = text;
    chatMessages.appendChild(div);
    chatMessages.scrollTop = chatMessages.scrollHeight;
  }

  chatForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    const text = chatInput.value.trim();
    if (!text) return;

    addMessage(text, "user");
    chatInput.value = "";
    chatInput.focus();

    const typingDiv = document.createElement("div");
    typingDiv.classList.add("msg", "bot");
    typingDiv.textContent = "Thinking...";
    chatMessages.appendChild(typingDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;

    try {
      const resp = await fetch("/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: text })
      });
      const data = await resp.json();
      chatMessages.removeChild(typingDiv);

      if (!resp.ok) {
        addMessage(data.error || "Something went wrong.", "bot");
        return;
      }
      addMessage(data.reply, "bot");
    } catch (err) {
      chatMessages.removeChild(typingDiv);
      addMessage("Error: " + err.message, "bot");
    }
  });
</script>
</body>
</html>
"""

@app.route("/", methods=["GET"])
def index():
    return render_template_string(INDEX_HTML)

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image file uploaded"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    try:
        img_bytes = file.read()
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        pred_class, conf, is_cancer = predict_image_pytorch(img)
        return jsonify({
            "pred_class": pred_class,
            "confidence": conf,
            "is_cancer": is_cancer
        })
    except Exception as e:
        print("Error during prediction:", e)
        return jsonify({"error": "Prediction failed"}), 500

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(silent=True) or {}
    user_message = (data.get("message") or "").strip()
    if not user_message:
        return jsonify({"error": "Empty message"}), 400

    system_prompt = (
        "You are an AI assistant on a pancreatic cancer demo website. "
        "You can explain how to use the site and give simple, general "
        "information about pancreatic cancer, CT/MRI imaging, and AI models. "
        "You must NOT provide medical diagnosis, predict the cancer stage, "
        "or give treatment advice. Always remind users to consult a "
        "qualified doctor for any medical concerns."
    )

    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        prompt = f"{system_prompt}\n\nUser question: {user_message}"
        response = model.generate_content(prompt)

        reply = (getattr(response, "text", "") or "").strip()
        if not reply:
            reply = (
                "I'm having trouble generating a reply right now. "
                "Please try again in a moment."
            )

        return jsonify({"reply": reply})

    except Exception as e:
        print("Gemini chat error:", repr(e))
        return jsonify({
            "reply": (
                "The AI chatbot service is currently unavailable. "
                "You can still use the image prediction part of this "
                "website. For medical questions, please consult a doctor."
            )
        }), 200

if __name__ == "__main__":
    app.run(debug=True)
