import os
import tempfile
from flask import Flask, render_template, request, send_file, jsonify
from dotenv import load_dotenv

# ==== Google GenAI SDK ====
from google import genai
from google.genai import types

# ==== Whisper para STT local ====
import whisper

load_dotenv()

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 200 * 1024 * 1024  # 200 MB

whisper_model = whisper.load_model("small")


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/transcribe-translate', methods=['POST'])
def transcribe_translate():
    if 'audio' not in request.files:
        return jsonify({"error": "Nenhum arquivo de áudio enviado."}), 400

    audio_file = request.files['audio']
    ip_language = request.form.get("ip_language", "pt")  # default "pt"
    api_key = request.form.get("api_key") or os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return jsonify({"error": "Nenhuma API Key fornecida."}), 400

    client = genai.Client(api_key=api_key)

    # Criar arquivo temporário e fechar para evitar bloqueio no Windows
    tmp_audio = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    audio_path = tmp_audio.name
    audio_file.save(audio_path)
    tmp_audio.close()

    try:
        # Transcrição com Whisper
        whisper_result = whisper_model.transcribe(audio_path, language="en")
        transcribed_text = whisper_result["text"].strip()

        if not transcribed_text:
            return jsonify({"error": "Falha na transcrição"}), 500

        # Tradução com Gemini
        translate_resp = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=f"Translate the following English text to {ip_language}. Return only the translation:\n\n{transcribed_text}"
        )
        translated_text = (translate_resp.text or "").strip()

        if not translated_text:
            return jsonify({"error": "Falha na tradução"}), 500

        return jsonify({"translated_text": translated_text})

    except Exception as e:
        return jsonify({"error": f"Ocorreu um erro: {e}"}), 500

    finally:
        if os.path.exists(audio_path):
            try:
                os.unlink(audio_path)
            except:
                pass


@app.route('/generate-tts', methods=['POST'])
def generate_tts():
    data = request.json
    if not data.get("text"):
        return jsonify({"error": "Nenhum texto fornecido"}), 400

    text = (
        "Generate a clear and natural-sounding audio reading of the following text. "
        "Speak slightly faster than normal, with minimal pauses between sentences, "
        "but keep it easy to understand. "
        + data.get("text")
    )

    api_key = data.get("api_key") or os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return jsonify({"error": "Nenhuma API Key fornecida."}), 400

    client = genai.Client(api_key=api_key)

    try:
        tts_resp = client.models.generate_content(
            model="gemini-2.5-flash-preview-tts",
            contents=text,
            config=types.GenerateContentConfig(
                response_modalities=["AUDIO"],
                speech_config=types.SpeechConfig(
                    voice_config=types.VoiceConfig(
                        prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name="Achird")
                    )
                )
            )
        )
        pcm_data = tts_resp.candidates[0].content.parts[0].inline_data.data

        tmp_tts = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        tts_path = tmp_tts.name
        tmp_tts.close()

        import wave
        with wave.open(tts_path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(24000)
            wf.writeframes(pcm_data)

        return send_file(
            tts_path,
            mimetype="audio/wav",
            as_attachment=True,
            download_name="audio_traduzido.wav"
        )

    except Exception as e:
        return jsonify({"error": f"Ocorreu um erro: {e}"}), 500

    finally:
        if 'tts_path' in locals() and os.path.exists(tts_path):
            try:
                os.unlink(tts_path)
            except:
                pass


if __name__ == '__main__':
    app.run(debug=True)
