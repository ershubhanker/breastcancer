# Generated by Django 4.1.4 on 2024-02-04 05:34

from django.conf import settings
from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('main', '0026_image_assigned_doctor_alter_userform_patient_id_and_more'),
    ]

    operations = [
        migrations.AddField(
            model_name='image',
            name='uploader',
            field=models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.SET_NULL, related_name='uploaded_images', to=settings.AUTH_USER_MODEL),
        ),
        migrations.AlterField(
            model_name='userform',
            name='patient_id',
            field=models.IntegerField(default=36600, primary_key=True, serialize=False, unique=True),
        ),
    ]
