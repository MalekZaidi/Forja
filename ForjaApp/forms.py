from django import forms
from django.contrib.auth.models import User
from django.contrib.auth.forms import UserCreationForm

from ForjaApp.models import Movie
from .models import Genre  # Make sure to import the Genre model


class UserRegisterForm(UserCreationForm):
    email = forms.EmailField(required=True)

    class Meta:
        model = User
        fields = ['username', 'email', 'password1', 'password2']
class MovieForm(forms.ModelForm):
    genres = forms.ModelMultipleChoiceField(
        queryset=Genre.objects.all(),
        widget=forms.CheckboxSelectMultiple,  # Or another widget of your choice
        required=True
    )
    class Meta:
        model = Movie
        fields = ['title', 'release_date', 'overview', 'poster_path','genres']


class GenreForm(forms.ModelForm):
    class Meta:
        model = Genre
        fields = ['name']