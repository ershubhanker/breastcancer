# Generated by Django 4.1.4 on 2024-02-13 07:33

from django.conf import settings
from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('main', '0034_userimage_remove_image_assigned_doctor_and_more'),
    ]

    operations = [
        migrations.AddField(
            model_name='userimage',
            name='assigned_doctor',
            field=models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.SET_NULL, to=settings.AUTH_USER_MODEL),
        ),
        migrations.AlterField(
            model_name='userimage',
            name='patient_id',
            field=models.IntegerField(default=68225, primary_key=True, serialize=False, unique=True),
        ),
    ]