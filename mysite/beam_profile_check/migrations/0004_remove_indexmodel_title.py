# Generated by Django 3.2 on 2021-07-27 14:57

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('beam_profile_check', '0003_indexmodel'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='indexmodel',
            name='title',
        ),
    ]