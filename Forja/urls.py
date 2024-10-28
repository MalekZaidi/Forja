"""
URL configuration for Forja project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.1/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from ForjaApp import views
from django.conf.urls import handler404
from ForjaApp.views import submit_feedback,post_list, post_detail, post_create, post_update, post_delete, update_comment  # Import the view


urlpatterns = [path('', views.index, name='index'),
    path('admin/', admin.site.urls),  
    path('register/', views.register, name='register'),
    path('login/', views.login_view, name='login'),
    path('logout/', views.logout_view, name='logout'),     
    path('profile/', views.profile, name='profile'),
    path('recommend-similar/', views.recommend_similar_movies, name='recommend_similar_movies'),
    path('generate-image/', views.generate_image, name='generate_image'),
    path('movie-ending/', views.movie_ending_view, name='movie_ending'),
    path('song-writer/', views.song_writer_view, name='song_writer'),
    path('submit-feedback/', views.submit_feedback, name='submit_feedback'),
    path('user-feedback/', views.user_feedback_management, name='user_feedback_management'),
    path('posts/', post_list, name='post_list'),
    path('posts/<int:post_id>/', post_detail, name='post_detail'),
    path('posts/create/', post_create, name='post_create'),
    path('posts/update/<int:post_id>/', post_update, name='post_update'),
    path('posts/delete/<int:post_id>/', post_delete, name='post_delete'),
    path('comment/update/<int:comment_id>/', update_comment, name='update_comment'),

]
