# Generated by Django 4.1.4 on 2024-02-24 09:32

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('main', '0046_pdfreport_alter_userform_appointed_doctor_and_more'),
    ]

    operations = [
        migrations.AlterField(
            model_name='userform',
            name='patient_id',
            field=models.IntegerField(default=38303, primary_key=True, serialize=False, unique=True),
        ),
    ]
