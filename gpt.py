from flask import Flask, render_template, request, send_file, jsonify
import numpy as np
import pandas as pd
from keras import models
from tensorflow import image
from PIL import Image
import io
import base64
from google.cloud import texttospeech
from googletrans import Translator

app = Flask(__name__)

desc = pd.read_csv("C:/Users/Naga Sai/Desktop/Agriculture/disease detection/description.csv")
model = models.load_model("C:/Users/Naga Sai/Desktop/Agriculture/disease detection/models.keras")

dis = list(desc.disease.values)

translator = Translator()

def translate_text(text, target_language):
    """
    Translate text using Google Translate API.
    """
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

def generate_audio_google(text, filename, language="en-US"):
    """
    Generate high-quality audio using Google Cloud Text-to-Speech API.
    """
    client = texttospeech.TextToSpeechClient()

    synthesis_input = texttospeech.SynthesisInput(text=text)
    voice = texttospeech.VoiceSelectionParams(
        language_code=language,
        ssml_gender=texttospeech.SsmlVoiceGender.NEUTRAL
    )
    audio_config = texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.MP3)

    response = client.synthesize_speech(input=synthesis_input, voice=voice, audio_config=audio_config)

    with open(filename, "wb") as out:
        out.write(response.audio_content)
    return filename

@app.route('/', methods=['GET', 'POST'])
def index():
    result, confidence, description, precautions = None, None, None, None
    image_data, selected_language = None, None
    result_audio, description_audio, precautions_audio = None, None, None

    if request.method == 'POST':
        selected_language = request.form.get('language')
        language_code = "en-US"  # Default language code
        if selected_language == "hi":
            language_code = "hi-IN"
        elif selected_language == "te":
            language_code = "te-IN"
        elif selected_language == "kn":
            language_code = "kn-IN"
        elif selected_language == "ml":
            language_code = "ml-IN"
        elif selected_language == "ta":
            language_code = "ta-IN"
        
        # Add more language mappings here if needed

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

                # Generate high-quality audio for the translated content
                result_audio = generate_audio_google(pro, "static/audio/result_audio.mp3", language_code)
                description_audio = generate_audio_google(des, "static/audio/description_audio.mp3", language_code)
                precautions_audio = generate_audio_google(pre, "static/audio/precautions_audio.mp3", language_code)

                # Set results
                result, confidence, description, precautions = pro, conf, des, pre

    return render_template('index.html', result=result, confidence=confidence, 
                           description=description, precautions=precautions, 
                           image_data=image_data, selected_language=selected_language,
                           result_audio=result_audio, description_audio=description_audio,
                           precautions_audio=precautions_audio)


if __name__ == '__main__':
    app.run(debug=True)
