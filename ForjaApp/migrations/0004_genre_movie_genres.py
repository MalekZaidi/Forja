# Generated by Django 5.1.2 on 2024-10-27 22:06

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('ForjaApp', '0003_alter_movie_poster_path'),
    ]

    operations = [
        migrations.CreateModel(
            name='Genre',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('name', models.CharField(max_length=100)),
            ],
        ),
        migrations.AddField(
            model_name='movie',
            name='genres',
            field=models.ManyToManyField(related_name='movies', to='ForjaApp.genre'),
        ),
    ]