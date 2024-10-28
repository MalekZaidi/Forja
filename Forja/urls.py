from django.conf import settings
from django.contrib import admin
from django.urls import path
from ForjaApp import views
from django.conf.urls.static import static
from ForjaApp.views import voice_search  # Adjusted import
from django.conf.urls import handler404

urlpatterns = [
    path('', views.index, name='index'),  # Keep this as the home page
    path('admin/', admin.site.urls),
    path('register/', views.register, name='register'),
    path('login/', views.login_view, name='login'),
    path('logout/', views.logout_view, name='logout'),
    path('profile/', views.profile, name='profile'),
    path('recommend-similar/', views.recommend_similar_movies, name='recommend_similar_movies'),
    path('generate-image/', views.generate_image, name='generate_image'),
    path('movies/create/', views.movie_create, name='create_movie'),
    path('movies/update/<int:movie_id>/', views.update_movie, name='update_movie'),
    path('movies/delete/<int:movie_id>/', views.delete_movie, name='delete_movie'),
    path('add_movie/', views.movie_create, name='add_movie'),  # Correct URL for adding a movie
    path('services/', views.services, name='services'),
    path('voice_search/', voice_search, name='voice_search'),
    path('movie_recognition/', views.movie_recognition, name='movie_recognition'),
    path('get_movie_summary/', views.get_movie_summary_and_ending, name='get_movie_summary'),
    path('add_genre/', views.add_genre, name='add_genre'),
    path('movies/', views.movie_list, name='movie_list'),

]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)

handler404 = 'ForjaApp.views.custom_404_view'
