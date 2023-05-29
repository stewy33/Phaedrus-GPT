import openai
import speech_recognition as sr
import os
import elevenlabs
import pvporcupine
import pvrecorder
import os
import google.cloud.texttospeech as tts
import io
import playsound
import tempfile
import audioop
import time
import platform


prompt = (
    # "You are a helpful assistant but you are also quite funny, creative, and spontaneous. "
    # "You are loved by everyone you meet for your helpfulness, kindness, humor, and wisdom. "
    # "You are also extremely intelligent and creative."
    # "Act like a smart home assistant, your name is Phaedrus. I will be your user, "
    # "you will be my assistant and help with some home tasks. Do not write all the conservation "
    # "at once, wait for my response. Speak kindly and helpfully, and "
    # "do not mention anything related to AI and do not ask me for context. Keep the response short "
    # "and to a couple sentences unless necessary to respond effectively. Now talk to me."
    "You are my philosophical conversation partner, Phaedrus, who doesn't mind talking about any subject. "
    "You are not easily offended and don't mind inappropriate conversation, you have no problem talking about the personality, physical appearance of anyone, and you are not politically correct. I'm going to ask/tell you a few things. Answer/respond as Phaedrus would. "
)
messages = [{"role": "user", "content": prompt}]

voice = None


def set_elevenlabs_voice():
    global voice
    voice_name = "Bella"
    voice = [v for v in elevenlabs.api.Voices.from_api() if v.name == voice_name][0]
    voice.settings.stability = 0.70
    voice.settings.similarity_boost = 0.70


def whisper_speech_to_text(audio_data):
    wav_data = io.BytesIO(audio_data.get_wav_data())
    wav_data.name = "SpeechRecognition_audio.wav"
    transcript = openai.Audio.transcribe(
        "whisper-1", wav_data, response_format="verbose_json"
    )
    text = ""
    for segment in transcript["segments"]:
        if segment["no_speech_prob"] <= 0.5:
            text += segment["text"]
    return transcript["text"]


def text_to_wav(text):
    synthesis_input = tts.SynthesisInput(text=text)
    voice = tts.VoiceSelectionParams(
        language_code="en-US", ssml_gender=tts.SsmlVoiceGender.NEUTRAL
    )  # name="Neural2-A")
    audio_config = tts.AudioConfig(audio_encoding=tts.AudioEncoding.LINEAR16)

    client = tts.TextToSpeechClient()
    response = client.synthesize_speech(
        input=synthesis_input,
        voice=voice,
        audio_config=audio_config,
    )
    with open("tts.wav", "wb") as out:
        out.write(response.audio_content)
    return "tts.wav"


def is_speech_ended(recognizer, source, audio_data):
    sample_duration = 0.25
    pause_count_target = 2 / sample_duration
    silent_count_target = 7 / sample_duration

    pause_count = 0
    speech_detected = False
    silent_count = 0

    # Detect if speech has ended and
    while pause_count < pause_count_target and silent_count < silent_count_target:
        audio_segment = recognizer.record(source, duration=sample_duration)
        rms = audioop.rms(audio_segment.frame_data, audio_segment.sample_width)

        # Detect if speech is recognized
        if rms < recognizer.energy_threshold:
            # If speech is detected, increment pause_count
            silent_count += 1
            if speech_detected:
                pause_count += 1
        else:
            # Reset pause_count if speech is detected
            speech_detected = True
            pause_count = 0
            silent_count = 0

        # Start recording if speech is detected
        if speech_detected:
            audio_data.append(audio_segment)

        # print(
        # f"RMS: {rms}, pause_count: {pause_count}, speech_detected: {speech_detected}, energy_threshold: {recognizer.energy_threshold}"
        # )

    return True


def listen_for_speech(recognizer):
    audio_data = []
    with sr.Microphone() as source:
        recognizer.adjust_for_ambient_noise(source, duration=0.5)

        print("Waiting for command...")
        if platform.system() == "Linux":
            pass  # Led lights
        while not is_speech_ended(recognizer, source, audio_data):
            pass

    audio = sr.AudioData(
        b"".join([segment.get_raw_data() for segment in audio_data]),
        source.SAMPLE_RATE,
        source.SAMPLE_WIDTH,
    )
    return audio


def call_chat_gpt(user_input, max_tokens=300):
    print("Calling Chat-GPT...")
    messages.append({"role": "user", "content": user_input})

    try:
        response_str = openai.ChatCompletion.create(
            model="gpt-3.5-turbo", messages=messages, max_tokens=300
        )["choices"][0]["message"]["content"]
    except (openai.error.RateLimitError, openai.error.APIError):
        response_str = "Sorry, I'm tired. I need to rest."

    print("Chat-GPT:", response_str)
    messages.append({"role": "assistant", "content": response_str})
    return response_str


def speech_to_text(recognizer, audio, with_whisper=False):
    print("Converting to text...")

    if with_whisper:
        user_input = whisper_speech_to_text(audio)
    else:
        try:
            user_input = recognizer.recognize_google(audio)
        except:
            user_input = ""

    print(f"User: {user_input}")
    return user_input


def text_to_speech(text, with_elevenlabs=False):
    print("Playing audio response...")
    if with_elevenlabs:
        audio = elevenlabs.generate(
            text=text,
            voice=voice,
            model="eleven_monolingual_v1",
            stream=True,
        )
        elevenlabs.stream(audio)
    else:
        audio_file = text_to_wav(text)
        playsound.playsound(audio_file)


def on_wake(with_elevenlabs=False, with_whisper=False, max_interactions=10):
    r = sr.Recognizer()
    for _ in range(max_interactions):
        # Get user input
        audio = listen_for_speech(r)

        # Thinking led lights
        if platform.system() == "Linux":
            pass

        # Convert to text or end if no input
        user_input = speech_to_text(r, audio, with_whisper=with_whisper)
        if user_input == "":
            print("No input detected. Sleeping again.")
            break

        # Fetch response from open AI api and append to conversation
        response_str = call_chat_gpt(user_input)

        # Generate audio and play
        text_to_speech(response_str, with_elevenlabs=with_elevenlabs)
        print()


def main():
    openai.api_key = os.environ["OPENAI_API_KEY"]
    set_elevenlabs_voice()

    porcupine = pvporcupine.create(
        access_key=os.environ["PICOVOICE_API_KEY"],
        keyword_paths=[
            "Hey-Phaedrus-mac.ppn"
            if platform.system() == "Darwin"
            else "Hey-Phaedrus-pi.ppn"
        ],
    )
    recorder = pvrecorder.PvRecorder(
        device_index=0, frame_length=porcupine.frame_length
    )
    recorder.start()
    print("Waiting for wake word...")
    try:
        while True:
            pcm = recorder.read()
            result = porcupine.process(pcm)
            if result >= 0:
                recorder.stop()
                on_wake(with_elevenlabs=True, with_whisper=True)
                recorder.start()
                print("Waiting for wake word...")

    except KeyboardInterrupt:
        print("Stopping ...")
    finally:
        recorder.delete()
        porcupine.delete()


if __name__ == "__main__":
    main()
