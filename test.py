import cv2
import torch
import torch.nn as nn

from flask import Flask, Response
from torchvision import models, transforms
from PIL import Image
import numpy as np

# =====================================
# Configuration
# =====================================

MODEL_PATH = "weights/best_model.pth"
IMAGE_SIZE = 224

DEVICE = torch.device("cpu")

print("Using device:", DEVICE)

# =====================================
# Load model
# =====================================

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE
)

classes = checkpoint["classes"]

model = models.efficientnet_b0(weights=None)

model.classifier[1] = nn.Linear(
    model.classifier[1].in_features,
    len(classes)
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.to(DEVICE)
model.eval()

print("Model Loaded")
print("Classes:", len(classes))

# =====================================
# Image Transform
# =====================================

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485,0.456,0.406],
        [0.229,0.224,0.225]
    )
])

# =====================================
# Camera
# =====================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Could not open camera")

print("Camera opened.")

# =====================================
# Flask
# =====================================

app = Flask(__name__)

def generate():

    while True:

        success, frame = cap.read()

        if not success:
            continue

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        image = Image.fromarray(rgb)

        image = transform(image)

        image = image.unsqueeze(0).to(DEVICE)

        with torch.no_grad():

            output = model(image)

            probs = torch.softmax(output, dim=1)

            confidence, pred = torch.max(probs, 1)

        pose = classes[pred.item()]
        conf = confidence.item() * 100

        text = f"{pose} ({conf:.1f}%)"

        cv2.putText(
            frame,
            text,
            (20,40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0,255,0),
            2
        )

        ret, buffer = cv2.imencode(".jpg", frame)

        frame_bytes = buffer.tobytes()

        yield (
            b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' +
            frame_bytes +
            b'\r\n'
        )

@app.route('/')

def index():

    return """
    <html>
    <head>
        <title>YogaFlow</title>
    </head>
    <body style="background:black;text-align:center;">
        <h2 style="color:white;">YogaFlow Live Detection</h2>
        <img src="/video_feed" width="900">
    </body>
    </html>
    """

@app.route('/video_feed')

def video_feed():

    return Response(
        generate(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

if __name__ == "__main__":

    print()
    print("====================================")
    print("Open your browser at:")
    print("http://localhost:5000")
    print("====================================")
    print()

    app.run(
        host="0.0.0.0",
        port=5000,
        threaded=True
    )
