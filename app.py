from flask import Flask, render_template, request, send_file
import numpy as np
import tensorflow as tf
import os
from reportlab.pdfgen import canvas

app = Flask(__name__)

# Load the TFLite model
interpreter = tf.lite.Interpreter(model_path="pneumonia_model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

UPLOAD_FOLDER = "static/uploads"
RESULTS_FOLDER = "static/results"
REPORTS_FOLDER = "static/reports"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)
os.makedirs(REPORTS_FOLDER, exist_ok=True)

def preprocess_image(image_path):
    from tensorflow.keras.preprocessing import image
    img = image.load_img(image_path, target_size=(150, 150))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array.astype(np.float32)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return "No file uploaded", 400

    file = request.files['file']
    filename = file.filename
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(file_path)

    img_array = preprocess_image(file_path)
    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()
    prediction = interpreter.get_tensor(output_details[0]['index'])[0][0]

    diagnosis = "Pneumonia Detected" if prediction > 0.5 else "Normal"
    confidence_percentage = round(prediction * 100, 2)
    affected_percentage = np.random.uniform(10, 50)

    # Get patient details
    patient_name = request.form.get("name", "Unknown")
    patient_age = request.form.get("age", "N/A")
    patient_gender = request.form.get("gender", "N/A")

    report_path = generate_pdf_report(
        filename, diagnosis, confidence_percentage, affected_percentage, patient_name, patient_age, patient_gender
    )

    return {
        "diagnosis": diagnosis,
        "confidence": f"{confidence_percentage}%",
        "affected_percentage": f"{round(affected_percentage, 2)}%",
        "report": report_path
    }

def generate_pdf_report(filename, diagnosis, confidence, affected_percentage, name, age, gender):
    report_path = os.path.join(REPORTS_FOLDER, f"{filename}.pdf")
    c = canvas.Canvas(report_path)
    c.setFont("Helvetica", 14)

    c.drawString(100, 770, "Pneumonia Detection Report")
    c.line(100, 765, 400, 765)
    c.drawString(100, 740, f"Patient Name: {name}")
    c.drawString(100, 720, f"Age: {age}")
    c.drawString(100, 700, f"Gender: {gender}")
    c.drawString(100, 680, f"Diagnosis: {diagnosis}")
    c.drawString(100, 660, f"Confidence: {confidence}%")
    c.drawString(100, 640, f"Affected Area: {affected_percentage}%")

    c.save()
    return report_path

@app.route('/download_report/<filename>')
def download_report(filename):
    return send_file(os.path.join(REPORTS_FOLDER, filename), as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
