# Generated by Django 4.1.4 on 2024-02-14 05:21

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('main', '0040_image_userform_delete_userimage'),
    ]

    operations = [
        migrations.AddField(
            model_name='image',
            name='user_form',
            field=models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.CASCADE, related_name='images', to='main.userform'),
        ),
        migrations.AlterField(
            model_name='userform',
            name='patient_id',
            field=models.IntegerField(default=34036, primary_key=True, serialize=False, unique=True),
        ),
    ]
