from django import forms
from django.contrib.auth.models import User
from django.contrib.auth.forms import UserCreationForm

from ForjaApp.models import Movie


class UserRegisterForm(UserCreationForm):
    email = forms.EmailField(required=True)

    class Meta:
        model = User
        fields = ['username', 'email', 'password1', 'password2']
class MovieForm(forms.ModelForm):
    class Meta:
        model = Movie
        fields = ['title', 'release_date', 'overview', 'poster_path']