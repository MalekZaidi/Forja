# Generated by Django 5.1.1 on 2024-10-28 14:44

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('ForjaApp', '0004_userfeedback_delete_watchlist'),
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
