# views.py

from django.shortcuts import render
from django.http import JsonResponse
from .sample import transcribe_audio  # Importing the function from sample.py

def upload_audio(request):
    if request.method == 'POST' and request.FILES.get('audioFile'):
        uploaded_file = request.FILES['audioFile']
        # Assuming 'uploaded_file' contains the uploaded audio file
        # Process the uploaded file (e.g., save it to a temporary location)
        audio_file_path = '/path/to/uploaded/audio/file.mp3'  # Change this to the actual file path
        transcript = transcribe_audio(audio_file_path)
        return JsonResponse({'success': True, 'transcript': transcript})
    return JsonResponse({'success': False, 'error': 'No file uploaded or invalid request method.'})
def home(request):
    return render(request, 'home.html')
