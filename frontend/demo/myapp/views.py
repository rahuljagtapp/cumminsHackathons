from django.shortcuts import render
from django.http import JsonResponse

def upload_audio(request):
    if request.method == 'POST' and request.FILES.get('audioFile'):
        uploaded_file = request.FILES['audioFile']
        # Process the uploaded file here (e.g., save it to the server, perform audio processing, etc.)
        # Example: uploaded_file.save('/path/to/save/' + uploaded_file.name)
        return JsonResponse({'success': True})
    return JsonResponse({'success': False, 'error': 'No file uploaded or invalid request method.'})
