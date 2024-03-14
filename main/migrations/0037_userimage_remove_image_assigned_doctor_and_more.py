# Generated by Django 4.1.4 on 2024-02-13 09:57

from django.conf import settings
from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('main', '0036_image_userform_delete_userimage'),
    ]

    operations = [
        migrations.CreateModel(
            name='UserImage',
            fields=[
                ('patient_id', models.IntegerField(default=23863, primary_key=True, serialize=False, unique=True)),
                ('patient_name', models.CharField(max_length=255)),
                ('patient_email', models.EmailField(max_length=254)),
                ('appointed_doctor', models.CharField(max_length=255)),
                ('patient_age', models.IntegerField()),
                ('patient_gender', models.CharField(choices=[('Male', 'Male'), ('Female', 'Female'), ('Others', 'Others')], max_length=10)),
                ('center', models.CharField(choices=[('Mumbai', 'Mumbai'), ('Pune', 'Pune'), ('Nagpur', 'Nagpur')], max_length=50)),
                ('report_generation_date', models.DateField()),
                ('scan_date', models.DateField()),
                ('title', models.CharField(max_length=255)),
                ('image', models.ImageField(upload_to='uploads/')),
                ('uploaded_at', models.DateTimeField(auto_now_add=True)),
                ('is_annotated', models.BooleanField(default=False)),
                ('temperature_data', models.JSONField(blank=True, null=True)),
                ('status', models.CharField(default='default_status', max_length=100)),
                ('assigned_doctor', models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.SET_NULL, to=settings.AUTH_USER_MODEL)),
                ('uploader', models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.SET_NULL, related_name='uploaded_images', to=settings.AUTH_USER_MODEL)),
            ],
        ),
        migrations.RemoveField(
            model_name='image',
            name='assigned_doctor',
        ),
        migrations.RemoveField(
            model_name='image',
            name='uploader',
        ),
        migrations.DeleteModel(
            name='UserForm',
        ),
        migrations.DeleteModel(
            name='Image',
        ),
    ]
