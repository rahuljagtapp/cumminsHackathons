# `pip3 install assemblyai` (macOS)
# `pip install assemblyai` (Windows)

import assemblyai as aai

aai.settings.api_key = "687e91e1cec74f9cb5380252de9ebd9d"
transcriber = aai.Transcriber()

transcript = transcriber.transcribe("hi.mp3")
# transcript = transcriber.transcribe("./my-local-audio-file.wav")

print(transcript.text)