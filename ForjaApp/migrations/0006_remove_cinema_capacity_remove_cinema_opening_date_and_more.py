# Generated by Django 5.1.1 on 2024-10-28 15:36

import datetime
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('ForjaApp', '0005_cinema_cinemarating_watchlater'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='cinema',
            name='capacity',
        ),
        migrations.RemoveField(
            model_name='cinema',
            name='opening_date',
        ),
        migrations.RemoveField(
            model_name='cinema',
            name='overview',
        ),
        migrations.AddField(
            model_name='cinema',
            name='date_created',
            field=models.DateTimeField(auto_now_add=True, default=datetime.datetime(2024, 1, 1, 0, 0)),
            preserve_default=False,
        ),
        migrations.AddField(
            model_name='cinema',
            name='description',
            field=models.TextField(default=datetime.datetime(2024, 1, 1, 0, 0)),
            preserve_default=False,
        ),
        migrations.AlterField(
            model_name='cinema',
            name='location',
            field=models.CharField(max_length=255),
        ),
        migrations.AlterField(
            model_name='cinema',
            name='name',
            field=models.CharField(max_length=255),
        ),
    ]