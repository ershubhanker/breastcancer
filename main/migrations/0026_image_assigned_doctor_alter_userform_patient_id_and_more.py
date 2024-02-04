# Generated by Django 4.1.4 on 2024-02-02 05:45

from django.conf import settings
from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('main', '0025_alter_userform_patient_id'),
    ]

    operations = [
        migrations.AddField(
            model_name='image',
            name='assigned_doctor',
            field=models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.SET_NULL, to=settings.AUTH_USER_MODEL),
        ),
        migrations.AlterField(
            model_name='userform',
            name='patient_id',
            field=models.IntegerField(default=98299, primary_key=True, serialize=False, unique=True),
        ),
        migrations.CreateModel(
            name='Assignment',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('doctor', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='doctor_assignments', to=settings.AUTH_USER_MODEL)),
                ('operator', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='operator_assignments', to=settings.AUTH_USER_MODEL)),
            ],
        ),
    ]