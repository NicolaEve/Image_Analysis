# Generated by Django 3.2.3 on 2021-05-24 10:35

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='BeamEnergy10fff',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('image', models.FileField(upload_to='images/')),
                ('title', models.CharField(max_length=200)),
            ],
            options={
                'db_table': 'beam_profile_check_image_10fff',
            },
        ),
        migrations.CreateModel(
            name='BeamEnergy10x',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('image', models.FileField(upload_to='images/')),
                ('title', models.CharField(max_length=200)),
            ],
            options={
                'db_table': 'beam_profile_check_image_10x',
            },
        ),
        migrations.CreateModel(
            name='BeamEnergy6x',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('image', models.FileField(upload_to='images/')),
                ('title', models.CharField(max_length=200)),
            ],
            options={
                'db_table': 'beam_profile_check_image_6x',
            },
        ),
        migrations.CreateModel(
            name='TransformView',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
            ],
        ),
    ]
