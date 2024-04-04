from django.shortcuts import render,redirect
from django.http import JsonResponse
import os
from django.conf import settings
import assemblyai as aai
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

import os
from tortoise.models.classifier import AudioMiniEncoderWithClassifierHead
from glob import glob
import io
import librosa
import torch
import torch.nn.functional as F

import numpy as np


tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)  # Adjust num_labels according to your task

    # Load the fine-tuned model weights
model.load_state_dict(torch.load('fine_tuned_distilbert.pth'))

    # Define the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Move the model to the device
model.to(device)

def upload_audio(request):
    if request.method == 'POST' and request.FILES['audio_file']:
        uploaded_file = request.FILES['audio_file']
        # Handle the uploaded file
        x=handle_uploaded_file(uploaded_file)
        return render(request, 'Main.html', {'result': x})

    return render(request, 'Main.html', {'result': None})


def handle_uploaded_file(uploaded_file):
    # Specify the directory where you want to save the uploaded file
    upload_dir = os.path.join(settings.MEDIA_ROOT, 'audio')
    os.makedirs(upload_dir, exist_ok=True)
    # Write the uploaded file to disk
    with open(os.path.join(upload_dir, uploaded_file.name), 'wb') as destination:
        for chunk in uploaded_file.chunks():
            destination.write(chunk)
    aai.settings.api_key = "687e91e1cec74f9cb5380252de9ebd9d"
    transcriber = aai.Transcriber()
    transcript = transcriber.transcribe(os.path.join(upload_dir, uploaded_file.name))
    audioclip=load_audio(uploaded_file)
    y=classify_audio_clip(audioclip)
    print("outputis ::  "+y+"\n\n")
    x=(transcript.text)
    predict(x)
    return x


def predict(text):
    # Initialize the tokenizer and model
    

    # Define a function to predict the label for a single input
    def predict_single_input(text, model, tokenizer):
        # Tokenize the input text
        inputs = tokenizer(text, padding=True, truncation=True, return_tensors='pt')

        # Move the input tensors to the device
        inputs = {key: tensor.to(device) for key, tensor in inputs.items()}

        # Forward pass through the model to get predictions
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits

        # Convert logits to probabilities using softmax
        probabilities = torch.softmax(logits, dim=1)

        # Get the predicted label
        predicted_label = torch.argmax(probabilities, dim=1).item()

        return predicted_label, probabilities[0]


    # Example usage:
    # Define a text to test
    text_to_test = text
    predicted_label, probabilities = predict_single_input(text_to_test, model, tokenizer)

    # Print the results
    print("Predicted class:", predicted_label)
    print("Probabilities:", probabilities)
    return predicted_label




def process_uploaded_file(uploaded_file):
    
    data = uploaded_file.read()  # Read the audio file
    tensor = torch.tensor(data)  # Convert data into a PyTorch tensor
    return tensor




def classify_audio_clip(clip):

    classifier = AudioMiniEncoderWithClassifierHead (2, spec_dim=1, embedding_dim=512, depth=5, downsample_factor=4,
                                                    resnet_blocks=2, attn_blocks=4, num_attn_heads=4, base_channels=32,
                                                    dropout=0, kernel_size=5, distribute_zero_label=False)
    state_dict = torch.load('classifier.pth', map_location=torch.device('cpu'))
    clip = process_uploaded_file(clip)
    clip = clip.unsqueeze(0)
    classifier.load_state_dict(state_dict)
    results = F.softmax(classifier(clip), dim=-1)
    return results[0][0]

def load_audio(audiopath, sampling_rate=22000):
    if isinstance(audiopath, str): # If the input is a file path
        if audiopath.endswith('.mp3'):
            audio, lsr = librosa.load(audiopath, sr=sampling_rate)
            audio = torch.FloatTensor(audio)
        elif audiopath.endswith('.wav'):
            audio, sr = read(audiopath)
            if sr != sampling_rate:
                audio = librosa.resample(np.float32(audio), sr, sampling_rate)
            audio = torch.FloatTensor(audio)
        else:
            assert False, f"Unsupported audio format provided: {audiopath[-4:]}"
    elif isinstance(audiopath, io.BytesIO): # If the input is file content
        audio, lsr = torchaudio.load(audiopath)
        audio = audio[0] # Remove any channel data
        if lsr != sampling_rate:
            audio = torchaudio.functional.resample(audio, lsr, sampling_rate)
    if torch.any(audio > 2) or not torch.any(audio < 0):
        print(f"Error with audio data. Max={audio.max()} min={audio.min()}")
    audio = torch.clamp(audio, -1, 1)
    return audio.unsqueeze(0)