# Generated by Django 4.1.4 on 2024-02-19 09:32

from django.conf import settings
from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('main', '0044_userform_main_image_alter_userform_patient_id'),
    ]

    operations = [
        migrations.AlterField(
            model_name='userform',
            name='appointed_doctor',
            field=models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.SET_NULL, to=settings.AUTH_USER_MODEL),
        ),
        migrations.AlterField(
            model_name='userform',
            name='patient_id',
            field=models.IntegerField(default=20228, primary_key=True, serialize=False, unique=True),
        ),
    ]