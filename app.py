import streamlit as st
import torch
from transformers import BertTokenizer, BertForSequenceClassification

# load model
model = BertForSequenceClassification.from_pretrained("intent_bert_model")
tokenizer = BertTokenizer.from_pretrained("intent_bert_model")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

id2label = {
    0: "book",
    1: "cancel",
    2: "view",
    3: "confirm",
    4: "reschedule",
    5: "check_availability"
}

def predict_intent(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    pred = torch.argmax(outputs.logits).item()
    probs = torch.softmax(outputs.logits, dim=1)
    confidence = torch.max(probs).item()

    return id2label[pred], confidence

def generate_response(intent):
    responses = {
        "book": "Appointment booked 📅",
        "cancel": "Appointment canceled ❌",
        "reschedule": "Rescheduling 🔄",
        "view": "Here is your schedule 👀",
        "check_availability": "Available slots ⏰",
        "confirm": "Confirmed ✅"
    }
    return responses.get(intent, "I didn't understand 🤖")

# UI
st.title("🎓 UniMeet AI Assistant")

user_input = st.text_input("Enter your request:")

if st.button("Predict"):
    intent, conf = predict_intent(user_input)
    response = generate_response(intent)

    st.write("### 🤖 Response:", response)
    st.write(f"Intent: {intent} | Confidence: {round(conf,3)}")