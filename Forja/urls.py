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
from django.conf import settings
from django.contrib import admin
from django.urls import path
from ForjaApp import views
from django.conf.urls.static import static

from django.conf.urls import handler404


urlpatterns = [path('', views.index, name='index'),
    path('admin/', admin.site.urls),  
    path('register/', views.register, name='register'),
    path('login/', views.login_view, name='login'),
    path('logout/', views.logout_view, name='logout'),     
    path('profile/', views.profile, name='profile'),
    path('recommend-similar/', views.recommend_similar_movies, name='recommend_similar_movies'),
    path('generate-image/', views.generate_image, name='generate_image'),
    path('movies/create/', views.movie_create, name='create_movie'),
    path('movies/', views.movie_list, name='list_movies'),
    path('movies/update/<int:movie_id>/', views.update_movie, name='update_movie'),
    path('movies/delete/<int:movie_id>/', views.delete_movie, name='delete_movie'),
    path('', views.index, name='home'),  # URL for your home page


               ]
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
handler404 = 'ForjaApp.views.custom_404_view'  
