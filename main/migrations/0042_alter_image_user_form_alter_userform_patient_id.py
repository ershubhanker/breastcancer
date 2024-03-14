# Generated by Django 4.1.4 on 2024-02-14 05:33

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('main', '0041_image_user_form_alter_userform_patient_id'),
    ]

    operations = [
        migrations.AlterField(
            model_name='image',
            name='user_form',
            field=models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='images', to='main.userform'),
        ),
        migrations.AlterField(
            model_name='userform',
            name='patient_id',
            field=models.IntegerField(default=82297, primary_key=True, serialize=False, unique=True),
        ),
    ]