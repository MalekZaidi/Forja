import json
import logging
from platform import processor

import PIL
import torch
from django.shortcuts import render, get_object_or_404
from django.shortcuts import render, redirect
from django.contrib.auth import login, authenticate, logout
from django.contrib import messages
from django.contrib.auth.forms import AuthenticationForm
from .forms import UserRegisterForm, MovieForm, GenreForm
from django.contrib.auth.decorators import login_required
from .utils import get_similar_movies
from .models import Recommendation
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import openai  # Make sure you have the OpenAI package installed

import requests
from django.template.loader import get_template
from django.http import Http404
from .models import Movie, Genre
import re
from django.shortcuts import render
from .models import Movie  # Adjust the import based on your project structure
import speech_recognition as sr
from transformers import CLIPProcessor, CLIPModel
from PIL import Image


# Load the CLIP model and processor once
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")
def index(request):
    return render(request, 'index.html')


# views.py

logger = logging.getLogger(__name__)


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
            if similar_movies:  # Vérifiez qu'il y a des films similaires
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
                similar_movies=[{"title": m['title'], "id": m['id']} for m in similar_movies]
                # Enregistrer les titres et IDs des films similaires
            )

        return render(request, 'recommendations.html', {'similar_movies': similar_movies, 'movie_title': movie_title})

    return render(request, 'recommendations.html', {'similar_movies': similar_movies, 'movie_title': movie_title})


def generate_image(request):
    image_url = None
    error_message = None

    if request.method == 'POST':
        description = request.POST.get('description')

        # Liste de mots-clés associés aux films
        movie_keywords = ['film', 'movie', 'character', 'plot', 'scene', 'actor', 'actress', 'director', 'genre',
                          'cinema', 'trailer']

        # Vérifier si la description contient des mots-clés de film
        if any(re.search(r'\b' + keyword + r'\b', description, re.IGNORECASE) for keyword in movie_keywords):
            image_url = f"https://image.pollinations.ai/prompt/{description}"
        else:
            error_message = "Veuillez entrer une description liée aux films."

    return render(request, 'image_generator.html', {'image_url': image_url, 'error_message': error_message})

# List all movies
def movie_list(request):
    genres = Genre.objects.all()  # Get all genres
    selected_genre = request.GET.get('genre')

    if selected_genre:
        movies = Movie.objects.filter(genres__name=selected_genre)
    else:
        movies = Movie.objects.all()  # Get all movies if no genre is selected

    return render(request, 'movie_list.html', {
        'movies': movies,
        'genres': genres,  # Make sure genres are passed here
        'selected_genre': selected_genre
    })



# Add a new movie
@login_required
def movie_create(request):
    if request.method == 'POST':
        form = MovieForm(request.POST, request.FILES)  # Include request.FILES for image upload
        if form.is_valid():
            movie = form.save(commit=False)  # Create the movie instance but don't save yet
            movie.save()  # Save the movie instance
            form.save_m2m()  # Save the many-to-many relationships (genres)
            print("Movie saved successfully!")  # Debug line
            return redirect('index')  # Redirect to the index page after saving
        else:
            print("Form errors:", form.errors)  # Debug line to see form errors
    else:
        form = MovieForm()

    return render(request, 'add_movie.html', {'form': form})  # Render the form

@login_required
def update_movie(request, movie_id):
    movie = get_object_or_404(Movie, id=movie_id)

    if request.method == 'POST':
        form = MovieForm(request.POST, request.FILES, instance=movie)  # Include request.FILES for file uploads
        if form.is_valid():
            form.save()
            messages.success(request, 'Movie updated successfully!')
            return redirect('index')  # Ensure 'index' is a valid URL name
    else:
        form = MovieForm(instance=movie)  # Pre-populate the form with existing movie data

    return render(request, 'update_movie.html', {'form': form, 'movie': movie})

# Delete Movie
@login_required
def delete_movie(request, movie_id):
    movie = get_object_or_404(Movie, id=movie_id)
    if request.method == 'POST':
        movie.delete()
        messages.success(request, 'Movie deleted successfully!')
        return redirect('index')
    return render(request, 'delete_movie.html', {'movie': movie})


def index(request):
    genres = Genre.objects.all()  # Get all genres for the dropdown
    selected_genre = request.GET.get('genre')  # Get the selected genre from query parameters

    # Get all movies for the carousel
    all_movies = Movie.objects.all()

    # Filter movies based on the selected genre
    if selected_genre:
        filtered_movies = Movie.objects.filter(genres__name=selected_genre)
    else:
        filtered_movies = Movie.objects.all()  # Show all if no genre is selected

    return render(request, 'index.html', {
        'all_movies': all_movies,
        'filtered_movies': filtered_movies,
        'genres': genres,
        'selected_genre': selected_genre
    })


def voice_search(request):
    if request.method == 'POST':
        # Here, implement the voice recognition logic
        recognizer = sr.Recognizer()
        with sr.Microphone() as source:
            print("Listening...")
            audio = recognizer.listen(source)
            try:
                query = recognizer.recognize_google(audio)
                print("You said: " + query)
                # Search your Movie model
                results = Movie.objects.filter(title__icontains=query)
                return render(request, 'search_results.html', {'results': results})
            except sr.UnknownValueError:
                return render(request, 'voice_search.html', {'error': "Could not understand audio"})
            except sr.RequestError:
                return render(request, 'voice_search.html',
                              {'error': "Could not request results from Google Speech Recognition service"})


def services(request):
    return render(request, 'services.html')


@csrf_exempt  # Only if you want to disable CSRF protection (not recommended for production)
def voice_search(request):
    if request.method == "POST":
        query = request.POST.get('query', '')

        if not query:
            return JsonResponse({'error': 'No query provided'}, status=400)

        try:
            results = Movie.objects.filter(title__icontains=query)

            results_data = [
                {
                    'title': movie.title,
                    'description': getattr(movie, 'overview', 'No description available'),
                    'release_date': getattr(movie, 'release_date', 'Unknown release date'),
                    'poster_url': movie.poster_path.url if movie.poster_path else None,
                    # Use poster_path instead of poster
                }
                for movie in results
            ]

            return JsonResponse({'results': results_data})

        except Exception as e:
            logger.error("Error occurred during voice search: %s", str(e))
            return JsonResponse({'error': str(e)}, status=500)

    return JsonResponse({'error': 'Invalid request'}, status=400)


@csrf_exempt
def movie_recognition(request):
    if request.method == 'POST':
        image_file = request.FILES.get('image')
        if image_file:
            try:
                # Open the image using PIL
                image = Image.open(image_file).convert("RGB")

                # Prepare a more extensive list of relevant movie titles
                movie_titles = [
                    'The Dark Knight',
                    'Batman',
                    'Batman Begins',
                    'The Dark Knight Rises',
                    'Joker',
                    'Suicide Squad',
                    'The Killing Joke',
                    'Batman vs. Superman: Dawn of Justice',
                    'Titanic',
                    'Inception',
                    'Gladiator',
                    'Jurassic Park',
                    'Iron Man',
                    'The Avengers',

                    # Add more relevant titles as needed
                ]

                # Process the image and text
                inputs = processor(text=movie_titles, images=image, return_tensors="pt", padding=True)

                # Make predictions
                with torch.no_grad():
                    outputs = model(**inputs)
                    logits_per_image = outputs.logits_per_image
                    probs = logits_per_image.softmax(dim=1)

                # Get the top K predictions
                top_k_indices = probs[0].topk(5)
                results = [
                    {"title": movie_titles[i.item()], "probability": probs[0][i].item()}
                    for i in top_k_indices.indices
                ]

                # Adjust the probability threshold for filtering
                threshold = 0.15  # Try different values based on testing
                filtered_results = [
                    result for result in results if result['probability'] >= threshold
                ]

                # Optional: Add logic to include related titles if certain ones are found
                if any("The Dark Knight" in result['title'] for result in filtered_results):
                    filtered_results.append({"title": "Batman", "probability": 0.8})  # Example addition

                return JsonResponse({"results": filtered_results})
            except Exception as e:
                print(f"Error processing the image: {e}")
                return JsonResponse({"error": str(e)}, status=500)

    return render(request, 'movie_recognition.html')


@csrf_exempt  # You can use this to avoid CSRF issues, but be cautious

@csrf_exempt
def get_summary_and_ending(movie_title):
    openai.api_key = 'YOUR_API_KEY'  # Replace with your actual API key

    try:
        # Get the summary
        summary_response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": f"Can you give me a summary of the movie '{movie_title}'?"}]
        )
        summary = summary_response['choices'][0]['message']['content']

        # Get a different ending
        ending_response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": f"Suggest a different ending for the movie '{movie_title}'."}]
        )
        different_ending = ending_response['choices'][0]['message']['content']

        return summary, different_ending

    except Exception as e:
        logger.error(f"Error fetching data from OpenAI: {e}")
        return None, None

import openai


from transformers import pipeline
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt

summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")


@csrf_exempt
def get_movie_summary_and_ending(request):
    if request.method == 'POST':
        movie_title = request.POST.get('movie_title')
        if not movie_title:
            return JsonResponse({'error': 'No movie title provided'}, status=400)

        omdb_api_key = 'dc75851e'
        omdb_url = f'https://www.omdbapi.com/?t={movie_title}&apikey={omdb_api_key}'

        try:
            response = requests.get(omdb_url)
            data = response.json()

            if data.get('Response') == 'True':
                plot = data.get('Plot', 'No plot found.')
                # Generate a different ending using the correct function
                alternative_ending = generate_different_ending(plot)

                return JsonResponse({
                    'title': data['Title'],
                    'summary': plot,
                    'alternative_ending': alternative_ending
                })

            return JsonResponse({'error': 'Movie not found'}, status=404)

        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)

    return JsonResponse({'error': 'Invalid request method'}, status=400)


def generate_different_ending(original_plot):
    ai_api_token = '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'  # Your Hugging Face API token
    ai_api_url = 'https://api-inference.huggingface.co/models/distilgpt2'  # Using distilgpt2 model

    headers = {
        'Authorization': f'Bearer {ai_api_token}',
        'Content-Type': 'application/json'
    }

    # Clearer prompt
    prompt = (
        f"Plot Summary: {original_plot}\n"
        "Let's Imagine a different ending for this movie, one that changes the outcome entirely. "

    )

    payload = {
        'inputs': prompt,
        'parameters': {
            'max_length': 150,  # Increased for more room to generate
            'temperature': 0.9,  # More creative variations
            'top_k': 50,  # Sampling top k tokens for diversity
            'top_p': 0.95,  # Nucleus sampling
        }
    }

    response = requests.post(ai_api_url, headers=headers, json=payload)

    if response.status_code == 200:
        response_json = response.json()
        if isinstance(response_json, list) and len(response_json) > 0:
            return response_json[0]['generated_text'].strip()  # Remove leading/trailing whitespace
        else:
            return "Could not generate an ending at this time."
    else:
        return f"API Error: {response.status_code}, {response.text}"

def add_genre(request):
    if request.method == 'POST':
        form = GenreForm(request.POST)
        if form.is_valid():
            form.save()  # Save the new genre
            return redirect('add_genre')  # Redirect back to the genre add page
    else:
        form = GenreForm()

    return render(request, 'add_genre.html', {'form': form})