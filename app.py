import os
from flask import Flask, render_template, request
import torch
from torchvision.models import resnet18
from torchvision import transforms
from PIL import Image
import google.generativeai as genai

# ---------------------------
# 1. Initialize Flask app
# ---------------------------
app = Flask(__name__)

# ---------------------------
# 2. Google Gemini API Key
# ---------------------------
genai.configure(api_key="AIzaSyBH0Zf1GJaPGqfub3_2cg0zX85T_u-awLo") # use your api instead of x

model_gemini = genai.GenerativeModel("models/gemini-2.5-flash")


# ---------------------------
# 3. Device & Classes
# ---------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
classes = ['aphids', 'armyworm', 'beetle', 'bollworm', 
           'grasshopper', 'mosquito', 'sawfly', 'stem_borer']

# ---------------------------
# 4. Load Model
# ---------------------------
model = resnet18(weights=None)
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, len(classes))
model.load_state_dict(torch.load("model/pest_model.pth", map_location=device))
model.to(device)
model.eval()

# ---------------------------
# 5. Image Transform
# ---------------------------
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

# ---------------------------
# 6. Advisory Dictionary
# ---------------------------
advisory_dict = {
    'aphids': "Spray neem oil or use ladybugs as biological control.",
    'armyworm': "Use Bacillus thuringiensis (Bt) spray or remove affected leaves.",
    'beetle': "Apply insecticidal soap or neem oil.",
    'bollworm': "Use pheromone traps and Bt sprays.",
    'grasshopper': "Use carbaryl or apply neem-based pesticide.",
    'mosquito': "Drain stagnant water and use larvicides.",
    'sawfly': "Handpick larvae and use neem spray.",
    'stem_borer': "Use light traps and remove infected plants."
}

# ---------------------------
# 7. Prediction Function
# ---------------------------
def predict_pest(image_path):
    image = Image.open(image_path).convert("RGB")
    img_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(img_tensor)
        _, predicted = torch.max(outputs, 1)
    pest_class = classes[predicted.item()]
    advice = advisory_dict.get(pest_class, "No advice available")
    return pest_class, advice

# ---------------------------
# 8. Gemini AI Ask Function
# ---------------------------
def ask_gemini(question):
    try:
        prompt = f"""
You are an agricultural expert.

Answer the following question in a clean, well-structured format:
- Use numbered steps
- Use short bullet points
- Use simple language suitable for farmers
- Avoid long paragraphs
- Add emojis where helpful
Rules:
- Do NOT use markdown
- Do NOT use ** or *
- Use simple numbered steps
- Each point on a new line

Question:
{question}
"""

        response = model_gemini.generate_content(prompt)
        return response.text.replace("**", "")

    except Exception as e:
        return f"Error: {str(e)}"

# ---------------------------
# 9. Flask Routes
# ---------------------------
@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    advice = None
    gpt_response = None

    if request.method == "POST":
        # Image upload prediction
        if "file" in request.files:
            file = request.files["file"]
            if file.filename != "":
                file_path = os.path.join("static", file.filename)
                file.save(file_path)
                prediction, advice = predict_pest(file_path)
        
        # Gemini chatbot query
        if "chat_input" in request.form:
            user_question = request.form["chat_input"]
            gpt_response = ask_gemini(user_question)

    return render_template("index.html", prediction=prediction, advice=advice, gpt_response=gpt_response)

# ---------------------------
# 10. Run Server
# ---------------------------
if __name__ == "__main__":
    app.run(debug=True)