from flask import Flask, render_template, request, send_file, jsonify
import numpy as np
import pandas as pd
from keras import models
from tensorflow import image
from PIL import Image
import io
import base64
import os
from gtts import gTTS
from googletrans import Translator

app = Flask(__name__)

desc = pd.read_csv("C:/Users/Naga Sai/Desktop/Agriculture/disease detection/description.csv")
model = models.load_model("C:/Users/Naga Sai/Desktop/Agriculture/disease detection/models.keras")

dis = list(desc.disease.values)

translator = Translator()

def translate_text(text, target_language):
    
    translated = translator.translate(text, dest=target_language)
    return translated.text

def image_classifier(inp):
    inp = image.resize(inp, (256, 256))
    inp = np.expand_dims(inp, 0)
    pred = model.predict(inp)
    return dis[np.argmax(pred)], f"Confidence - {round(max(pred[0]) * 100, 2)}%"

def detail(pro):
    x = desc[desc["disease"] == pro]
    return list(x["desc"])[0], list(x["pre"])[0]

def generate_audio(text, filename):
    tts = gTTS(text)
    tts.save(filename)
    return filename

@app.route('/', methods=['GET', 'POST'])
def index():
    result, confidence, description, precautions = None, None, None, None
    image_data, selected_language = None, None
    result_audio, description_audio, precautions_audio = None, None, None

    if request.method == 'POST':
        selected_language = request.form.get('language')
        if 'image' in request.files:
            file = request.files['image']
            if file:
                img = Image.open(file.stream)

                # Encode image in base64
                buffered = io.BytesIO()
                img.save(buffered, format="JPEG")
                image_data = base64.b64encode(buffered.getvalue()).decode()

                # Run the classifier and fetch details
                pro, conf = image_classifier(img)
                des, pre = detail(pro)

                # Translate content if a language is selected
                if selected_language:
                    pro = translate_text(pro, selected_language)
                    conf = translate_text(conf, selected_language)
                    des = translate_text(des, selected_language)
                    pre = translate_text(pre, selected_language)

                # Generate audio for the translated content
                result_audio = generate_audio(pro, "static/audio/result_audio.mp3")
                description_audio = generate_audio(des, "static/audio/description_audio.mp3")
                precautions_audio = generate_audio(pre, "static/audio/precautions_audio.mp3")

                # Set results
                result, confidence, description, precautions = pro, conf, des, pre

    return render_template('index.html', result=result, confidence=confidence, 
                           description=description, precautions=precautions, 
                           image_data=image_data, selected_language=selected_language,
                           result_audio=result_audio, description_audio=description_audio,
                           precautions_audio=precautions_audio)


if __name__ == '__main__':
    app.run(debug=True)
