from django.shortcuts import render
from django.shortcuts import render, redirect
from django.contrib.auth import login, authenticate, logout
from django.contrib import messages
from django.contrib.auth.forms import AuthenticationForm
from .forms import UserRegisterForm
from django.contrib.auth.decorators import login_required
from .utils import get_similar_movies
from .models import Recommendation
from .models import Movie 
import re
import requests
import random
import openai
import requests
from django.http import JsonResponse
from groq import Groq
import os
from django.conf import settings
client = Groq(api_key='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')
openai.api_key = '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'
openai.api_base = "https://api.groq.com/openai/v1"
API_KEY = '89a4748b3788935d5e08221e4ed6f7ef'
BASE_URL = 'https://suno-apiv2-eight.vercel.app'

def range_1_to_5():
    return range(1, 6)
def index(request):
    url = f'https://api.themoviedb.org/3/movie/popular?api_key={API_KEY}&language=en-US&page=1'
    response = requests.get(url)
    movies = response.json().get('results', [])

    # Préparer l'URL de base des images
    image_base_url = 'https://image.tmdb.org/t/p/w500/'

    # Ajouter l'URL complète pour chaque image de film et calculer la note moyenne sur 2
    for movie in movies:
        movie['poster_url'] = image_base_url + movie.get('poster_path', '')
        movie['vote_average_div_2'] = movie['vote_average'] / 2

    context = {'movies': movies[:8], 'range_1_to_5': range(1, 6)}  # Limiter à 8 films
    return render(request, 'index.html', context)

# views.py


def register(request):
    if request.method == 'POST':
        form = UserRegisterForm(request.POST)
        if form.is_valid():
            form.save()
            username = form.cleaned_data.get('username')
            messages.success(request, f'Votre compte a été créé avec succès !')
            return redirect('login')
    else:
        form = UserRegisterForm()
    return render(request, 'register.html', {'form': form})

def login_view(request):
    if request.method == 'POST':
        form = AuthenticationForm(request, data=request.POST)
        if form.is_valid():
            username = form.cleaned_data.get('username')
            password = form.cleaned_data.get('password')
            user = authenticate(username=username, password=password)
            if user is not None:
                login(request, user)
                messages.success(request, f'Vous êtes connecté !')
                return redirect('index')
            else:
                messages.error(request, 'Nom d\'utilisateur ou mot de passe incorrect.')
    else:
        form = AuthenticationForm()
    return render(request, 'login.html', {'form': form})

def logout_view(request):
    logout(request)
    messages.success(request, 'Vous avez été déconnecté.')
    return redirect('login')

@login_required  # Ceci nécessite que l'utilisateur soit connecté
def profile(request):
    return render(request, 'profile.html', {'user': request.user})

from django.shortcuts import render

def custom_404_view(request, exception):
    return render(request, '404.html', status=404)
@login_required
def recommend_similar_movies(request):
    similar_movies = []
    movie_title = ""

    if request.method == 'POST':
        movie_title = request.POST.get('movie_title')

        # Obtenir les films similaires à l'aide de l'algorithme optimisé
        similar_movies = get_similar_movies(movie_title)

        # Vérifier si le film est trouvé dans la base de données
        movie = Movie.objects.filter(title__icontains=movie_title).first()

        # Si le film n'est pas trouvé, l'ajouter à la base de données
        if not movie:
            # Utiliser le premier film similaire pour créer une nouvelle entrée de film
            if similar_movies:  # Vérifier qu'il y a des films similaires
                movie_data = similar_movies[0]  # Prenons le premier film similaire

                # Créer une nouvelle instance de Movie
                movie = Movie(
                    title=movie_data['title'],
                    release_date=movie_data['release_date'],
                    overview=movie_data['overview'],
                    poster_path=movie_data['poster_path']
                )
                movie.save()  # Sauvegarder le nouveau film dans la base de données

        # Si le film est trouvé ou a été ajouté, enregistrer la recommandation
        if movie:
            recommendation = Recommendation.objects.create(
                user=request.user,
                movie=movie,
                similar_movies=[{"title": m['title'], "id": m['id']} for m in similar_movies]  # Enregistrer les titres et IDs des films similaires
            )

        return render(request, 'recommendations.html', {'similar_movies': similar_movies, 'movie_title': movie_title})

    return render(request, 'recommendations.html', {'similar_movies': similar_movies, 'movie_title': movie_title})
def generate_image(request):

    image_url = None
    error_message = None

    if request.method == 'POST':
        description = request.POST.get('description')
        selected_style = request.POST.get('style')  # Récupérer le style sélectionné

        # Liste de mots-clés associés aux films
        movie_keywords = [
            'film', 'movie', 'character', 'plot', 'scene', 'actor', 
            'actress', 'director', 'genre', 'cinema', 'trailer'
        ]

        # Vérifier si la description contient des mots-clés de film
        if any(re.search(r'\b' + keyword + r'\b', description, re.IGNORECASE) for keyword in movie_keywords):
            # Ajouter le style de dessin à la description
            modified_description = f"{description}, {selected_style}"
            image_url = f"https://image.pollinations.ai/prompt/{modified_description}"
        else:
            error_message = "Veuillez entrer une description liée aux films ou personnages de films."

    return render(request, 'image_generator.html', {'image_url': image_url, 'error_message': error_message})

def generate_text(prompt):
    """Use Groq AI to generate a response based on a user prompt."""
    response = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model="llama3-8b-8192"
    )
    return response.choices[0].message.content.strip()

def text_to_speech(text):
    """Convert the given text to speech using Deepgram."""
    max_length = 2000
    if len(text) > max_length:
        text = text[:max_length]  # Truncate text to max_length characters

    url = "https://api.deepgram.com/v1/speak?model=aura-asteria-en"
    headers = {
        "Authorization": f"Token {'76dcd77db53a7659884675d0bfbad83679d5ab1f'}",
        "Content-Type": "application/json",
    }
    payload = {"text": text}

    response = requests.post(url, json=payload, headers=headers)

    print(f"Response Status Code: {response.status_code}")
    print(f"Response Body: {response.text}")

    if response.status_code == 200:
        return response.content  # Return audio bytes
    else:
        raise Exception(f"Failed to generate speech: {response.status_code}, {response.text}")






def movie_ending_view(request):
    """Handle user requests for custom movie endings."""
    if request.method == "POST":
        prompt = request.POST.get('prompt', "Imagine a new ending for Inception.")

        # Generate movie ending using Groq AI
        generated_text = generate_text(prompt)

        # Convert the generated text to speech
        audio_content = text_to_speech(generated_text)

        # Ensure the static directory exists
        static_dir = os.path.join(settings.BASE_DIR, "ForjaApp", "static")
        os.makedirs(static_dir, exist_ok=True)

        # Save the audio content to a file
        audio_file_path = os.path.join(static_dir, "generated_audio.mp3")
        with open(audio_file_path, "wb") as f:
            f.write(audio_content)

        return render(request, "movie_ending.html", {
            "generated_text": generated_text,
            "audio_path": f"{settings.STATIC_URL}generated_audio.mp3"  # Use STATIC_URL to construct the path
        })

    return render(request, "movie_ending.html")

def generate_audio_by_prompt(prompt):
    """Send prompt to the Vercel API to generate audio."""
    url = f"{BASE_URL}/api/generate"
    payload = {
        "prompt": prompt,
        "make_instrumental": False,
        "wait_audio": True  # Synchronous generation
    }
    try:
        response = requests.post(url, json=payload, headers={'Content-Type': 'application/json'})
        response.raise_for_status()  # Raise an exception for 4xx/5xx responses
        return response.json()
    except Exception as e:
        print(f"Error in generate_audio_by_prompt: {e}")
        return None

def get_audio_information(audio_id):
    """Check the status and retrieve the audio URL."""
    url = f"{BASE_URL}/api/get?ids={audio_id}"
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for 4xx/5xx responses
        return response.json()
    except Exception as e:
        print(f"Error in get_audio_information: {e}")
        return None

def song_writer_view(request):
    """Django view to generate song lyrics and audio based on a prompt."""
    audio_url = None
    generated_lyrics = None

    if request.method == 'POST':
        prompt = request.POST.get('prompt', 'Write a pop song about dreams.')

        # Generate lyrics using your model
        try:
            generated_lyrics = client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="llama3-8b-8192"
            ).choices[0].message.content.strip()
        except Exception as e:
            return JsonResponse({"error": f"Failed to generate lyrics: {e}"}, status=500)

        # Call the Vercel API to generate audio from the lyrics
        audio_response = generate_audio_by_prompt(generated_lyrics)
        print("Audio Response:", audio_response)  # Debugging the raw response

        if audio_response and isinstance(audio_response, list) and len(audio_response) > 0:
            audio_id = audio_response[0].get('id')
            if not audio_id:
                return JsonResponse({"error": "Audio ID not found in the response"}, status=500)
        else:
            return JsonResponse({"error": "Invalid audio response"}, status=500)

        # Poll for audio status
        try:
            for _ in range(60):
                audio_info = get_audio_information(audio_id)
                print("Audio Info:", audio_info)  # Debugging the raw response

                if (audio_info and isinstance(audio_info, list) and len(audio_info) > 0 and 
                    audio_info[0].get("status") == "streaming"):
                    audio_url = audio_info[0].get("audio_url")
                    break
                time.sleep(5)
        except Exception as e:
            return JsonResponse({"error": f"Error retrieving audio information: {e}"}, status=500)

    return render(request, 'song_writer.html', {
        'generated_lyrics': generated_lyrics,
        'audio_url': audio_url
    })