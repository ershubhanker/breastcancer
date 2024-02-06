from django.utils import timezone
# Create your models here.
from django.contrib.auth.models import AbstractUser,User
from django.db import models
import uuid
from django.http import request
import random
from django.db import models
from django.conf import settings

class Assignment(models.Model):
    operator = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name='operator_assignments')
    doctor = models.ForeignKey('Doctor', on_delete=models.CASCADE, related_name='doctor_assignments')


Operator =(  
    ("Operator", "Operator"), 
    ("Doctor", "Doctor"), 
)


class UserForm(models.Model):
    GENDER_CHOICES = [
        ('Male', 'Male'),
        ('Female', 'Female'),
        ('Others', 'Others'),
    ]

    CENTER_CHOICES = [
        ('Mumbai', 'Mumbai'),
        ('Pune', 'Pune'),
        ('Nagpur', 'Nagpur'),
    ]

    patient_id = models.IntegerField(primary_key=True, default=random.randint(10000, 99999), unique=True)
    patient_name = models.CharField(max_length=255)
    patient_email = models.EmailField()
    appointed_doctor = models.CharField(max_length=255)
    patient_age = models.IntegerField()
    patient_gender = models.CharField(max_length=10, choices=GENDER_CHOICES)
    center = models.CharField(max_length=50, choices=CENTER_CHOICES)
    report_generation_date = models.DateField()
    scan_date = models.DateField()

    def save(self, *args, **kwargs):
        super().save(*args, **kwargs)
class Image(models.Model):
    title = models.CharField(max_length=255)
    image = models.ImageField(upload_to='uploads/')
    uploaded_at = models.DateTimeField(auto_now_add=True)
    is_annotated = models.BooleanField(default=False)
    temperature_data = models.JSONField(null =True, blank=True)
    assigned_doctor = models.ForeignKey('Doctor', on_delete=models.SET_NULL, null=True, blank=True)
    uploader = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.SET_NULL, null=True, blank=True, related_name='uploaded_images')
class Doctor(AbstractUser):
    fullname = models.CharField(max_length=100, null=True, blank=True)
    specialty = models.CharField(max_length=100, null=True, blank=True)
    mobile = models.CharField(max_length=15, null=True, blank=True)
    address = models.CharField(max_length=500, null=True, blank=True)
    pincode = models.CharField(max_length=10, null=True, blank=True)
    credit_val = models.IntegerField(default=10)
    doctor_id = models.CharField(max_length=20, null=True,unique=True)
    geeks_field = models.CharField(max_length=10, null=True, blank=True,choices = Operator) 
    # email = models.EmailField(unique=True)

    # USERNAME_FIELD ='email'
    # REQUIRED_FIELDS= []

    def __str__(self):
        staff_status = "-(staff user)" if self.is_staff else " "
        return f"{self.username}  {staff_status}"
    
    # def save(self, *args, **kwargs):
        if not self.doctor_id:
    #         # Generate a unique doctor_id based on name and a number
    #         base_id = self.username.lower().replace(" ", "_")[:10]  # Using the first 10 characters of the username
            self.doctor_id = f"{self.pk}"

        super().save(*args, **kwargs)

    def get_credit(self):
        return self.credit_val


    def remove_credit(self):
        self.credit_val -= 1
        self.save()
  
class CreditRequest(models.Model):
    doctor = models.ForeignKey(Doctor, on_delete=models.CASCADE)
    request_date = models.DateTimeField(auto_now_add=True)
    credit_value = models.IntegerField()

    def __str__(self):
        return f"Credit Request by {self.doctor.fullname} on {self.request_date}"    

    

# Patient's retina Image 
class Patient(models.Model):
    YES_NO_CHOICES = [
        ('Yes', 'Yes'),
        ('No', 'No'),
    ]
    STAGE_CHOICES = [
        ('Type 1', 'Type 1'),
        ('Type 2', 'Type 2'),
    ]

    patient_id = models.CharField(max_length=20, blank=True, null=True)
    patient_name = models.CharField(max_length=100, null=True, blank=True)
    patient_email = models.CharField(max_length=100, null=True, blank=True)
    diabetes_status = models.CharField(max_length=3, choices=YES_NO_CHOICES, null=True, blank=True)
    diabetes_type = models.CharField(max_length=10, choices=STAGE_CHOICES, null=True, blank=True)
    left_image = models.ImageField(upload_to='media/patient_images/', null=True, blank=True)
    right_image = models.ImageField(upload_to='media/patient_images/', null=True, blank=True)
    # timestamp = models.DateTimeField(auto_now_add=True)
    timestamp = models.DateTimeField(default=timezone.now, null=True, blank=True)
    doctor = models.ForeignKey(Doctor, on_delete=models.SET_NULL, null=True, blank=True)


    def __str__(self):
        return f"Patient ID: {self.patient_id} patient_name:{self.patient_name}"
    
    def save(self, *args, **kwargs):
        if not self.doctor and hasattr(request, 'user'):
            self.doctor = request.user
        super(Patient, self).save(*args, **kwargs)
    



class EmailToken(models.Model):

    user = models.ForeignKey(
        Doctor,
        on_delete=models.CASCADE,
        related_name="email_user",
        blank=True,
        null=True,
    )
    email_token = models.CharField(max_length=50)
