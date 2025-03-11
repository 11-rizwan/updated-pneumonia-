from flask import Flask, request, render_template, send_file, jsonify
from flask_socketio import SocketIO
import tensorflow as tf
import numpy as np
import cv2
import os
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path="pneumonia_model.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def preprocess_image(image_path):
    img = load_img(image_path, target_size=(150, 150))  
    img_array = img_to_array(img) / 255.0
    return np.expand_dims(img_array, axis=0).astype(np.float32)

def generate_pdf_report(filename, diagnosis, confidence, affected_percentage, heatmap_path):
    report_path = f"static/reports/{filename}.pdf"
    c = canvas.Canvas(report_path, pagesize=letter)
    c.setFont("Helvetica", 14)
    
    c.drawString(100, 750, "Pneumonia Detection Report")
    c.line(100, 745, 400, 745)
    
    c.drawString(100, 720, f"Diagnosis: {diagnosis}")
    c.drawString(100, 700, f"Confidence: {confidence:.2f}")
    c.drawString(100, 680, f"Affected Area: {affected_percentage:.2f}%")
    
    if os.path.exists(heatmap_path):
        c.drawImage(heatmap_path, 100, 500, width=200, height=200)

    c.save()
    return report_path

@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('upload_image')
def handle_image(data):
    file_path = f"static/uploads/{data['filename']}"
    
    with open(file_path, "wb") as f:
        f.write(data['file'])  

    socketio.emit('update_status', {'message': "Image received. Processing..."})

    img_array = preprocess_image(file_path)
    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()
    prediction = interpreter.get_tensor(output_details[0]['index'])[0][0]

    socketio.emit('update_status', {'message': "Generating heatmap..."})

    heatmap_path = "static/results/heatmap.jpg"  # Assume heatmap is generated
    affected_percentage = np.random.uniform(10, 50)  # Simulated affected percentage

    diagnosis = "Pneumonia Detected" if prediction > 0.5 else "Normal"
    report_path = generate_pdf_report(data['filename'], diagnosis, prediction, affected_percentage, heatmap_path)

    result = {
        "prediction": diagnosis,
        "confidence": float(prediction),
        "affected_percentage": round(affected_percentage, 2),
        "heatmap": heatmap_path,
        "report": report_path
    }

    socketio.emit('prediction_result', result)

@app.route('/download_report/<filename>')
def download_report(filename):
    report_path = f"static/reports/{filename}.pdf"
    return send_file(report_path, as_attachment=True)

if __name__ == '__main__':
    socketio.run(app, debug=True)
