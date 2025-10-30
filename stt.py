from transformers import pipeline
import json

# Load the large Whisper model from Hugging Face
pipe = pipeline(
    task="automatic-speech-recognition",
    model="openai/whisper-base"
)

# Transcribe + translate Hindi audio to English
result = pipe("audios/12_Exercise 1 - Pure HTML Media Player.mp3", generate_kwargs={"task": "translate", "language": "hi"},return_timestamps=True)

print(result["text"])
with open("output.json","w") as f:
    json.dump(result,f)
