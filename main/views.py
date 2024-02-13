from django.contrib.auth.tokens import default_token_generator
from django.contrib.sites.shortcuts import get_current_site
from django.utils.http import urlsafe_base64_encode, urlsafe_base64_decode
from django.utils.encoding import force_bytes
from django.template.loader import render_to_string
from django.core.mail import send_mail
from django.contrib import messages
from django.shortcuts import render, redirect
from django.contrib.auth import login
from django.urls import reverse
from main.forms import DoctorRegistrationForm
from django.contrib.sites.models import Site
from reportlab.lib.pagesizes import letter
import base64
from django.core.files.base import ContentFile
from reportlab.platypus import SimpleDocTemplate, Spacer
import cv2
import numpy as np
from django.core.files.storage import FileSystemStorage
from django.core.mail import EmailMessage
from django.utils import timezone
from django.views.decorators.csrf import csrf_exempt
from django.shortcuts import render, HttpResponse,get_object_or_404
from django.contrib.auth import login, logout
from django.shortcuts import render, redirect
from .forms import DoctorRegistrationForm, DoctorLoginForm
from django.contrib.auth import authenticate, get_user_model
from django.http import HttpResponseRedirect
from django.contrib import messages
import requests
import json
import os
import random
import string
from reportlab.platypus import KeepTogether
import pandas as pd
from django.template.loader import get_template
from .models import Patient, Doctor, CreditRequest, EmailToken, Image
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from reportlab.lib.units import inch
from reportlab.lib.colors import white, HexColor, black, green
from reportlab.lib.pagesizes import letter
from reportlab.platypus import Table, TableStyle, Paragraph
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from datetime import datetime, timedelta
from reportlab.lib import colors
from datetime import datetime, date
from django.db.models import Avg
from django.http import JsonResponse
from django.core import serializers
from django.conf import settings
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
import json
from .forms import CreditRequestForm
import requests
from .serializer import *
from .forms import UserFormForm
from rest_framework.viewsets import ModelViewSet
from rest_framework_simplejwt.views import TokenObtainPairView ,InvalidToken
from rest_framework.permissions import IsAuthenticated
from django.views import View
from .email import Sendresetpasswordlink
import uuid
from django.db.models import Count
from django.db.models.functions import ExtractWeekDay
from .temperature_cal import get_temperature_from_pixel
from PIL import Image as PILImage
from django.contrib.auth.decorators import login_required
from django.core import serializers
from .models import UserForm
from .models import Image, Assignment, Doctor
from .forms import UserFormForm
from django.views.decorators.http import require_http_methods
from reportlab.pdfgen import canvas
from io import BytesIO
from .models import UserForm
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Image as ReportLabImage
global_token=None
check_user=None
def save_token(token_value, user):
    # Declare the global variable inside the function to indicate you're modifying the global variable
    global global_token
    global check_user
    global_token = token_value
    check_user=user

    print(f"Token value '{token_value}' saved successfully. y user{check_user}")

def get_saved_token():
    # Access the global variable
    global global_token
    global check_user
    print(global_token, check_user)
    return global_token, check_user

# register
# 1
def doctor_register(request):
    if request.method == 'POST':
        form = DoctorRegistrationForm(request.POST)
        if form.is_valid():
            user = form.save()
            user.backend = 'main.UsernameOrMobileModelBackend'
            login(request, user)
            return redirect('doctor_login')  # Change to your desired redirect URL
        else:
            # Display error messages for form validation errors
            error_messages = ', '.join([f"{field}: {', '.join(errors)}" for field, errors in form.errors.items()])
            messages.error(request, f"Registration failed. {error_messages}")
    else:
        form = DoctorRegistrationForm()
    return render(request, 'register.html', {'form': form})

class Resetpasswordview(View):
    def get(self, request):
        return render(request, "forget_password.html")

    def post(self,request):
        email = request.POST.get("email")

        user = Doctor.objects.filter(email=email.lower()).first()

        if user:
            token = uuid.uuid4()
            obj, _ = EmailToken.objects.get_or_create(user=user)
            obj.email_token = token
            obj.save()
            receiver_email = obj.user.email
            Sendresetpasswordlink(receiver_email,obj.email_token )
            messages.success(request,f"The link to reset your password has been emailed to: {receiver_email}")
            return render(request, 'forget_password.html')
        
        messages.error(request,f"User does not exist")
        return render(request, 'forget_password.html')

    
class confirmpassword(View):
    def get(self, request, token=None):
        try:
            token_obj = EmailToken.objects.filter(email_token=token).first()

            return render(request, "forget_confirm.html",{'token':token_obj.email_token})
        
        except Exception as e:
            messages.error(request,f"Invaild or expired URL")
            return render(request, 'forget_password.html')
            # return HttpResponse("invaild url")

    def post(self, request,token=None):

        password = request.POST.get("password1")
        print(token, 'SAKAL')
        obj  = EmailToken.objects.filter(email_token= token).first()
        print(obj.user.email, 'OOOOOOP',obj.user.id)
        user_obj = Doctor.objects.filter(id=obj.user_id).first()
        print()
        try:
            user_obj = Doctor.objects.filter(id=obj.user.id).first()
            print(user_obj,'okok')
            user_obj.set_password(password)
            user_obj.save()
            obj.delete() 
            messages.success(request,f"Password updated successfully")
            # return HttpResponse("Passoword  Update  successfully")
            return render(request, "login.html",{'token':token})
        except Exception as e:
            messages.error(request,f"User does not exist")
            print(e, 'SAJAL @')
            return render(request, "forget_password.html",{'token':token})
        

def email_verification_sent(request):
    return render(request, 'email_verification_sent.html')

def doctor_login(request):
    if request.method == 'POST':
        form = DoctorLoginForm(data=request.POST)
        if form.is_valid():
            user = form.get_user()
            design = user.geeks_field
            if design == "Operator":
                login(request, user)
                return redirect('upload-image') 
            if design == "Doctor":
                login(request, user)
                return redirect('index') 
                
            if user.is_staff:
                login(request, user)
                credit_requset_list(request)
                return redirect('admin_home')  # Redirect to staff-specific page
            # else:
            #     return redirect('index')  # Redirect to regular home page
        else:
            messages.error(request, 'Invalid Username or Password. Please try again.')

    else:
        form = DoctorLoginForm()
    return render(request, 'login.html', {'form': form})


# logout
def doctor_logout(request):
    logout(request)
    return redirect('doctor_login') 

# home page
# @login_required
# def home(request):
#     if request.method == 'POST':
#         form = UserFormForm(request.POST)
#         if form.is_valid():
#             form.save()
#             return render(request, 'operator/upload-image.html')  # Customize the success template as needed
#         else:
#             print(form.errors)  # Print form errors for debugging
#     else:
#         # Set default values for date fields
#         default_values = {
#             'report_generation_date': timezone.now().date(),
#             'scan_date': timezone.now().date(),
#         }
#         form = UserFormForm(initial=default_values)

#     return render(request, 'operator/home.html', {'form': form})
# send credit requset data to admin home


def credit_requset_list(request):
    # if request.method == 'GET':
    request_list = CreditRequest.objects.all()
    serialized_data = serializers.serialize('json', request_list)
    print('serialized_data',serialized_data)
    # h={'shvjh':'hvxhg'}
    return JsonResponse(serialized_data, safe=False)
# def credit_request_list(request):
#     request_list = CreditRequest.objects.all()
#     serialized_data = []

#     for credit_request in request_list:
#         serialized_request = {
#             'doctor_name': credit_request.doctor.fullname,
#             'request_date': credit_request.request_date,
#             'credit_value': credit_request.credit_value,
#         }
#         serialized_data.append(serialized_request)

#     return JsonResponse(serialized_data, safe=False)


# Define a function to get the number of patients for each weekday
def patients_by_weekday():
    return (
        Patient.objects
        .values('timestamp__week_day')  # Extract the day of the week
        .annotate(patient_count=Count('id'))  # Count the number of patients for each weekday
        .order_by('timestamp__week_day')
        .values_list('patient_count', flat=True)  # Optional: order the results by weekday
    )
# admin home page
@login_required
def adminhome(request):
    return render(request, 'admin_index.html')
    # return render(request, 'admin_home.html')

# dashboard (doctor)
@login_required
def index(request):
    
    return render(request, 'doctor/index.html')

def DoctorProfile(request):
    doctor = Doctor.objects.get(id=request.user.id)
    print(doctor, 'Profile')
    context={'doctor':doctor}
    return render(request, 'doctorpanel/doctor-profile.html', context)


@login_required
def upload_image(request):
    # if request.method == 'POST':
    #     image_field_name = next(iter(request.FILES))
    #     image_file = request.FILES.get(image_field_name)

    #     if image_file:
    #         # Create the image instance without saving to DB
    #         new_image = Image(title=image_field_name.replace('_', ' ').capitalize(), 
    #                           image=image_file, 
    #                            status='default_status',
    #                           uploader=request.user)

    #         # Logic to assign the image to a doctor
    #         # This is a placeholder, replace with your actual logic
    #         assignment = Assignment.objects.filter(operator=request.user).first()
    #         if assignment:
    #             new_image.assigned_doctor = assignment.doctor
    #             print(new_image.assigned_doctor)
    #             # Save the image instance to DB
    #         new_image.save()
    #         messages.success(request, "Image uploaded successfully.") 
    #         return redirect('upload-image')

    # return render(request, 'operator/upload-image.html')
    if request.method == 'POST':
        form = UserFormForm(request.POST)
        if form.is_valid():
            user_form = form.save()  # Save the patient information
            
            # Handle image uploads
            images = {
                'center_image': request.FILES.get('center_image'),
                'left_45_image': request.FILES.get('left_45_image'),
                'left_90_image': request.FILES.get('left_90_image'),
                'right_45_image': request.FILES.get('right_45_image'),
                'right_90_image': request.FILES.get('right_90_image'),
            }
            for image_field_name, image_file in images.items():
                if image_file:
                    new_image = Image(
                        title=image_field_name.replace('_', ' ').capitalize(), 
                        image=image_file, 
                        status='default_status',
                        uploader=request.user
                    )
                    # Logic to assign the image to a doctor
                    assignment = Assignment.objects.filter(operator=request.user).first()
                    if assignment:
                        new_image.assigned_doctor = assignment.doctor
                    new_image.save()
            
            messages.success(request, "Patient information and images uploaded successfully.")
            return redirect('upload-image')
        else:
            print(form.errors)  # Print form errors for debugging

    else: 
        form = UserFormForm(initial={
            'report_generation_date': timezone.now().date(),
            'scan_date': timezone.now().date(),
        })

    return render(request, 'operator/upload-image.html', {'form': form})
def delete_image(request, image_id):
    if request.method == 'POST':
        image = get_object_or_404(Image, pk=image_id)
        image.delete()
        return redirect('operator/upload-image.html')
# def home(request):
#     return render(request, 'home.html')  # Make sure this template path is correct.
@login_required
def annotate_image(request):
    images = Image.objects.all()
    images = Image.objects.filter(assigned_doctor=request.user, is_annotated=False)
    if images:
        return render(request, 'doctor/annotate-image.html', {'images': images})
    else:
        # If there are no images, you might want to display a message in the template
        # or redirect to another page with a message.
        context = {'message': 'No images to annotate'}
        return render(request, 'doctor/annotate-image.html', context)

    
@csrf_exempt
@login_required
def save_annotated_image(request):
    if request.method == 'POST':
        body_unicode = request.body.decode('utf-8')
        body_data = json.loads(body_unicode)
        image_data = body_data.get('imageData')
        if image_data:
                # Decode the data URL
                format, imgstr = image_data.split(';base64,') 
                ext = format.split('/')[-1]

                # Create a new directory in media if it doesn't exist
                new_dir = 'new_directory'
                if not os.path.exists(os.path.join(settings.MEDIA_ROOT, new_dir)):
                    os.makedirs(os.path.join(settings.MEDIA_ROOT, new_dir))

                # Save the image in the new directory
                image_file = ContentFile(base64.b64decode(imgstr), name=os.path.join(new_dir, 'annotated_image.' + ext))
                annotated_image = Image.objects.create(title='Annotated Image', image=image_file,is_annotated=True)
                return JsonResponse({'status': 'success', 'path': annotated_image.image.url})
        else:
                return JsonResponse({'status': 'error', 'message': 'No image data received'}, status=400)

def handle_uploaded_file(f):
    fs = FileSystemStorage()
    filename = fs.save(f.name, f)
    uploaded_file_url = fs.url(filename)
    return os.path.join(settings.MEDIA_ROOT, filename), uploaded_file_url
@login_required
def thermal_image_view(request):
    # Handling the AJAX request for temperature calculation
    if request.method == 'GET' and all(k in request.GET for k in ('x', 'y', 'displayWidth', 'displayHeight')):
        x = float(request.GET['x'])
        y = float(request.GET['y'])
        display_width = float(request.GET['displayWidth'])
        display_height = float(request.GET['displayHeight'])

        # Load the thermal image to get its actual dimensions
        thermal_image_path = os.path.join(settings.MEDIA_ROOT, 'media/new_directory/', 'callibaration_image.png')
        
        thermal_image = PILImage.open(thermal_image_path)
        actual_width, actual_height = thermal_image.size

        # Scale coordinates
        x_scaled = int(x * (actual_width / display_width))
        y_scaled = int(y * (actual_height / display_height))

        # Calibration image path and temperature calculation
        calibration_image_path = os.path.join(settings.MEDIA_ROOT, 'uploads', 'new_directory', 'annotated_image.png')
        temperature = get_temperature_from_pixel(x_scaled, y_scaled, calibration_image_path, thermal_image_path, 37.26, 22.79)

        if temperature is not None:
            return JsonResponse({'temperature': f"{temperature:.2f}°C"})
        else:
            return JsonResponse({'error': 'Unable to calculate temperature'})


    # Handling the request to display the page with images
    else:
        annotated_images = Image.objects.filter(assigned_doctor=request.user, is_annotated=True)

        # Generate URLs for these images
        annotated_image_urls = [os.path.join(settings.MEDIA_URL, image.image.name) for image in annotated_images]

        # Render the template with the image URLs
        return render(request, 'doctor/thermal-image.html', {'annotated_images': annotated_image_urls})

def analyze_thermal_image(image_path, output_path, calibration_image_path, threshold_temperature):
    # Load images
    calibration_image = cv2.imread(calibration_image_path)
    thermal_image = cv2.imread(image_path)

    # Extract the middle column of pixels in the calibration image
    middle_column = calibration_image[:, calibration_image.shape[1] // 2, :]

    # Define a color mapping based on provided RGB values and temperatures
    color_temp_mapping = [
    ([0, 0, 0], 22.79),
    ([1, 0, 50], 23),
    ([0, 1, 106], 24),
    ([0, 3, 130], 25),
    ([0, 0, 172], 26),
    ([0, 235, 253], 27),
    ([0, 255, 43], 28),
    ([73, 254, 0], 29),
    ([254, 245, 0], 30),
    ([255, 194, 2], 31),
    ([254, 79, 0], 33),
    ([255, 41, 0], 34),
    ([255, 5, 1], 35),
    ([254, 0, 0], 36),
    ([255, 0, 0], 37.62)
]

    def rgb_to_temperature(pixel_color):
        pixel_color_np = np.array(pixel_color, dtype=np.int16)
        min_distance = float('inf')
        closest_temp = None
        for color, temp in color_temp_mapping:
            distance = np.linalg.norm(pixel_color_np - np.array(color, dtype=np.int16))
            if distance < min_distance:
                min_distance = distance
                closest_temp = temp
        return closest_temp

    def temperature_to_rgb(temperature):
        closest_color = min(color_temp_mapping, key=lambda x: abs(x[1] - temperature))[0]
        return closest_color

    def temperature_to_hsv(temperature):
        rgb_color = temperature_to_rgb(temperature)
        hsv_color = cv2.cvtColor(np.uint8([[rgb_color]]), cv2.COLOR_RGB2HSV)[0][0]
        return hsv_color
    # Analyzing the thermal image
    gray = cv2.cvtColor(thermal_image, cv2.COLOR_BGR2GRAY)
    _, black_mask = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)
    kernel = np.ones((5,5), np.uint8)
    black_mask = cv2.dilate(black_mask, kernel, iterations=1)
    contours, _ = cv2.findContours(black_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    breast_contour = max(contours, key=cv2.contourArea)
    breast_mask = np.zeros_like(gray)
    cv2.drawContours(breast_mask, [breast_contour], -1, 255, -1)

    hsv = cv2.cvtColor(thermal_image, cv2.COLOR_BGR2HSV)
    hsv_value = temperature_to_hsv(threshold_temperature)
    lower_bound = np.array([hsv_value[0], 120, 70])
    upper_bound = np.array([hsv_value[0] + 10, 255, 255])
    mask = cv2.inRange(hsv, lower_bound, upper_bound)
    color_within_breast = cv2.bitwise_and(mask, mask, mask=breast_mask)
    contours, _ = cv2.findContours(color_within_breast, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(thermal_image, contours, -1, (255, 0, 0), 2)

        # Save the resulting image instead of displaying it
    cv2.imwrite(output_path, thermal_image)

def calculate_thermal_image(image_path, calibration_image_path,threshold_temperature):
    # Load the calibration image
    calibration_image = cv2.imread(calibration_image_path)
    middle_column = calibration_image[:, calibration_image.shape[1] // 2, :]

    top_temperature = 37.62
    bottom_temperature = 25.78
    temperature_range = top_temperature - bottom_temperature

    def rgb_to_temperature(rgb_pixel):
        differences = np.sqrt(np.sum((middle_column - rgb_pixel) ** 2, axis=1))
        index_of_smallest_difference = np.argmin(differences)
        temperature = top_temperature - (temperature_range * (index_of_smallest_difference / len(middle_column)))
        return temperature
    def aerolar_difference(image_path, calibration_image_path):
    # Function to convert an RGB color to the closest temperature based on calibration data
        # def rgb_to_temperature(pixel_color, color_temp_mapping):
        #     pixel_color_np = np.array(pixel_color, dtype=np.int16)
        #     min_distance = float('inf')
        #     closest_temp = None
        #     for color, temp in color_temp_mapping:
        #         distance = np.linalg.norm(pixel_color_np - np.array(color, dtype=np.int16))
        #         if distance < min_distance:
        #             min_distance = distance
        #             closest_temp = temp
        #     return closest_temp

        # Function to map color to temperature using calibration data
        def map_color_to_temp_using_calibration(color, color_temp_mapping):
            color = np.asarray(color, dtype=np.uint8)
            distances = np.linalg.norm(color_temp_mapping['color'] - color, axis=1)
            closest_index = np.argmin(distances) 
            return color_temp_mapping['temp'][closest_index]

        # Load images
        thermal_image = cv2.imread(image_path)
        calibration_image = cv2.imread(calibration_image_path)

        # Extract the middle column of pixels in the calibration image
        middle_column = calibration_image[:, calibration_image.shape[1] // 2, :]

        # Generate color-temp mapping based on the middle column assuming linear temperature gradient
        top_temp = 37.62
        bottom_temp = 22.79
        temp_gradient = np.linspace(bottom_temp, top_temp, middle_column.shape[0])

        # Create a mapping list
        color_temp_mapping = [(tuple(rgb[::-1]), temp) for rgb, temp in zip(middle_column, temp_gradient)]
        color_temp_mapping = np.array(color_temp_mapping, dtype=[('color', '3u1'), ('temp', 'f4')])

        # Load the thermal image and find contours
        gray_image = cv2.cvtColor(thermal_image, cv2.COLOR_BGR2GRAY)
        _, binary_image = cv2.threshold(gray_image, 1, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Sort contours by area and get the largest two, which are assumed to be the crosses.
        sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)[:2]

        # List to store cross information
        crosses_info = []

        # Find centers of the two largest contours which are assumed to be crosses
        for cnt in sorted_contours:
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                crosses_info.append({"center": (cX, cY)})

        # Process each cross to get the average color around it and map to temperature
        for cross in crosses_info:
            # Sample points around the center to avoid the cross itself
            points_to_sample = [
                (cross['center'][0] + 10, cross['center'][1]),
                (cross['center'][0] - 10, cross['center'][1]),
                (cross['center'][0], cross['center'][1] + 10),
                (cross['center'][0], cross['center'][1] - 10)
            ]

            # Fetch the colors from the image
            colors = []
            for point in points_to_sample:
                if (0 <= point[0] < thermal_image.shape[1]) and (0 <= point[1] < thermal_image.shape[0]):
                    colors.append(thermal_image[point[1], point[0]])

            # Calculate the mean color of the sampled points
            mean_color = np.mean(colors, axis=0).astype(int)
            mean_color_rgb = tuple(mean_color[::-1])

            # Map the color to the temperature
            temperature = map_color_to_temp_using_calibration(mean_color_rgb, color_temp_mapping)
            cross['temperature'] = temperature
        if len(crosses_info) >= 2:
            # Calculate the temperature difference between the two crosses
            diff = abs(crosses_info[0]['temperature'] - crosses_info[1]['temperature'])
            # Clamp the difference if it exceeds 5 degrees
            diff = min(diff, 4)
            avg = (crosses_info[0]['temperature'] + crosses_info[1]['temperature']) / 2
            ans = round(diff / avg, 2) * 100 if avg != 0 else 0
            return diff, ans
        else:
            # Return None or raise an exception if there are not enough crosses detected
            return 0.0, 0.0

    def calculate_roi_statistics(mask, image, calibration_column):
        roi_pixels = image[mask > 0]
        roi_temperatures = [rgb_to_temperature(pixel) for pixel in roi_pixels]
        max_temp = round(np.max(roi_temperatures), 2)  # Rounded to 2 decimal places
        min_temp = round(np.min(roi_temperatures), 2)  # Rounded to 2 decimal places
        mean_temp = round(np.mean(roi_temperatures), 2)  # Rounded to 2 decimal places
        return max_temp, min_temp, mean_temp
    
    def classify_contour_shapes(contours):
        shapes_classification = {'irregular': 0, 'distorted': 0}
        
        for contour in contours:
            # Approximate the contour to reduce the number of points
            perimeter = cv2.arcLength(contour, True)
            approximation = cv2.approxPolyDP(contour, 0.04 * perimeter, True)
            
            # A threshold for how many vertices we expect a "regular" shape to have
            # For example, a square would have 4, but if we get more, it might be "distorted"
            if len(approximation) > 10:
                shapes_classification['irregular'] += 1
            else:
                shapes_classification['distorted'] += 1
        
        return shapes_classification
    def calculate_percentage(image_path):
        image = cv2.imread(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, blue_thresh = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)
        _, black_thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)
        blue_contours, _ = cv2.findContours(blue_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        black_contours, _ = cv2.findContours(black_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        largest_blue_contour = max(blue_contours, key=cv2.contourArea) if blue_contours else None
        largest_black_contour = max(black_contours, key=cv2.contourArea) if black_contours else None
        blue_area = cv2.contourArea(largest_blue_contour) if largest_blue_contour is not None else 0
        black_area = cv2.contourArea(largest_black_contour) if largest_black_contour is not None else 0
        percentage = round((blue_area / black_area), 2) if black_area > 0 else 0  # Convert to percentage and round
        return percentage
    
    def calculate_temperature_difference(image, calibration_column):
        lower_blue = np.array([110, 50, 50])  # Adjust these values for your image
        upper_blue = np.array([130, 255, 255])
        blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
        contours, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        temperature_differences = []
        shapes_classification = classify_contour_shapes(contours)
    
        # Create a string description of the shapes
        shape_description = ", ".join(f"{count} {shape}" for shape, count in shapes_classification.items() if count > 0)
        for contour in contours:
            mask = np.zeros_like(image[:, :, 2])
            cv2.drawContours(mask, [contour], -1, 255, -1)
            boundary_pixels = image[mask == 255]
            boundary_temperatures = [rgb_to_temperature(pixel) for pixel in boundary_pixels]
            mean_boundary_temp = np.mean(boundary_temperatures)
            surrounding_mask = cv2.dilate(mask, np.ones((5, 5), np.uint8)) - mask
            surrounding_pixels = image[surrounding_mask == 255]
            surrounding_temperatures = [rgb_to_temperature(pixel) for pixel in surrounding_pixels]
            mean_surrounding_temp = np.mean(surrounding_temperatures)
            temperature_difference = mean_boundary_temp - mean_surrounding_temp
            temperature_differences.append(temperature_difference)
        return round(np.mean(temperature_differences), 2) if temperature_differences else 0,shape_description

    # Process the image and calculate parameters
    image = cv2.imread(image_path)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    calibration_column = calibration_image[:, calibration_image.shape[1] // 2, :]

    _, black_mask = cv2.threshold(hsv[:, :, 2], 50, 255, cv2.THRESH_BINARY_INV)
    _, red_mask = cv2.threshold(hsv[:, :, 0], 150, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(black_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    breast_contour = max(contours, key=cv2.contourArea)

    breast_mask = np.zeros_like(hsv[:, :, 0])
    cv2.drawContours(breast_mask, [breast_contour], -1, 255, -1)
    red_within_breast = cv2.bitwise_and(red_mask, red_mask, mask=breast_mask)
    red_contours, _ = cv2.findContours(red_within_breast, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    max_temp, min_temp, mean_temp = calculate_roi_statistics(breast_mask, hsv[:, :, 2], calibration_column)
    num_hotspots = len(red_contours)
    hotspot_area = sum(cv2.contourArea(cnt) for cnt in red_contours)
    roi_area = cv2.contourArea(breast_contour)
    hotspot_extent = round((hotspot_area / roi_area) * 100, 2)*100 if roi_area > 0 else 0
    percentage = calculate_percentage(image_path)
    temperature_difference, shape_description = calculate_temperature_difference(hsv, calibration_column)
    aerolar_symmetry, aerolar_differences = aerolar_difference(image_path, calibration_image_path)
    aerolar_symmetry = f'{round(aerolar_symmetry,2):.2f}'
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Apply a threshold to get only the hot regions
    _, thresh_image = cv2.threshold(gray_image, threshold_temperature, 255, cv2.THRESH_BINARY)

    # Find contours in the thresholded image
    contours, _ = cv2.findContours(thresh_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Filter out small contours that may not be actual hotspots
    min_contour_area = 100  # This value may need to be adjusted
    hotspot_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_contour_area]

    
    
    parameters = {
        "NumberofHotspots": len(hotspot_contours),
        "TemperatureDifferenceAroundBoundaries": temperature_difference,
        "HotspotShapes": shape_description,
        "ExtentofHotspots": hotspot_extent,
        "MaxHotspotTemperature": max_temp,
        "MinHotspotTemperature": min_temp,
        "MeanROITemperature": mean_temp,
        "PercentageofROI": percentage,
        "temperaturedifference":aerolar_differences,
        "AerolarSymmetry" : aerolar_symmetry,
    }

    return parameters
@csrf_exempt
def save_threshold(request):
    threshold_value = request.POST.get('threshold')
    if threshold_value:
        # Convert the threshold to a float and save it in the session
        request.session['threshold_value'] = float(threshold_value)
        return JsonResponse({'status': 'success', 'threshold': threshold_value})
    else:
        return JsonResponse({'status': 'error', 'message': 'No threshold value received'}, status=400)
@login_required
def thermal_parameters(request):
    threshold_value = request.session.get('threshold_value', 34)

    # Get images assigned to the logged-in doctor and that have been processed
    assigned_images = Image.objects.filter(assigned_doctor=request.user, is_annotated=True)

    # Directory where you want to save the processed images
    processed_directory_path = os.path.join(settings.MEDIA_ROOT, 'uploads', 'processed_thermal_image')

    # Calibration image path
    calibration_image_path = os.path.join(settings.MEDIA_ROOT, 'media/new_directory', 'callibaration_image.png')

    # Create the processed images directory if it doesn't exist
    if not os.path.exists(processed_directory_path):
        os.makedirs(processed_directory_path)
    
    thermal_data = []

    for image in assigned_images:
        filename = image.image.name.split('/')[-1]
        if filename.endswith('.png'):  # Assuming you want to process PNG images
            input_image_path = os.path.join(settings.MEDIA_ROOT, image.image.name)
            output_image_name = f'processed_{filename}'
            output_image_path = os.path.join(processed_directory_path, output_image_name)

            # Process the image and save it in the new directory
            analyze_thermal_image(input_image_path, output_image_path, calibration_image_path, threshold_value)

            # Calculate thermal parameters
            parameters = calculate_thermal_image(input_image_path, calibration_image_path, threshold_value)
            parameters = {key: float(value) if isinstance(value, np.float32) else value for key, value in parameters.items()}
            # Adding the threshold value inside the parameters
            parameters['threshold'] = threshold_value 

            # Create the URL for the processed image
            processed_image_url = os.path.join(settings.MEDIA_URL, 'uploads/', 'processed_thermal_image/', output_image_name)

            thermal_data.append({
                'processed_image_url': processed_image_url,
                'parameters': parameters
            })

    context = {'thermal_data': thermal_data}
    return render(request, 'doctor/thermal_parameters.html', context)

# To generate PDF 
def calculate_average_parameters(threshold_value):
    images = Image.objects.all()
    total_params = {
        "NumberofHotspots": 0,
        "TemperatureDifferenceAroundBoundaries": 0,
        "HotspotShapes": 0,
        "ExtentofHotspots": 0,
        "MaxHotspotTemperature": 0,
        "MinHotspotTemperature": 0,
        "MeanROITemperature": 0,
        "PercentageofROI": 0,
        "temperaturedifference":0,
        "AerolarSymmetry" : 0,
        
    }
    count = 0

    calibration_image_path = os.path.join(settings.MEDIA_ROOT, 'media/new_directory/', 'callibaration_image.png')
    if not os.path.exists(calibration_image_path):
        raise FileNotFoundError(f"Calibration image not found at {calibration_image_path}")

    for image in images:
        try:
            image_file_path = os.path.join(settings.MEDIA_ROOT, image.image.name)
            if not os.path.isfile(image_file_path):
                print(f"Image file not found at {image_file_path}")
                continue

            parameters = calculate_thermal_image(image_file_path, calibration_image_path, threshold_value)
            if parameters is not None:
                for key, value in parameters.items():
                    if isinstance(value, (int, float)):
                        total_params[key] += value
                count += 1
        except Exception as e:
            print(f"Error processing image {image.id}: {e}")

    if count > 0:
        # Round the averages to 2 decimal places
        averages = {key: round(total / count, 2) for key, total in total_params.items()}
    else:
        averages = {key: 0 for key in total_params.keys()}  # or handle no images case

    return averages

def calculate_highest_parameters(threshold_value):
    images = Image.objects.all()
    highest_params = {
        "NumberofHotspots": 0,
        "TemperatureDifferenceAroundBoundaries": 0,
        "ExtentofHotspots": 0,
        "HotspotShapes": 0,
    }
    calibration_image_path = os.path.join(settings.MEDIA_ROOT, 'media/new_directory/', 'callibaration_image.png')
    if not os.path.exists(calibration_image_path):
        raise FileNotFoundError(f"Calibration image not found at {calibration_image_path}")
    for image in images:
        try:
            image_file_path = os.path.join(settings.MEDIA_ROOT, image.image.name)
            parameters = calculate_thermal_image(image_file_path, calibration_image_path, threshold_value)
            if parameters is not None:
                for key in highest_params:
                    if key in parameters and isinstance(parameters[key], (int, float)):
                        highest_params[key] = max(highest_params[key], parameters[key])
        except Exception as e:
            print(f"Error processing image {image.id}: {e}")

    return highest_params

def get_image_paths(self, directory):
    # List all image files in the given directory
    return [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.png')]
class GeneratePDFView(View):
    def get_image_paths(self, directory):
        # List all image files in the given directory
        image_paths = []
        for f in os.listdir(directory):
            if f.lower().endswith('.png'):
                image_paths.append(os.path.join(directory, f))
        return image_paths
    def get(self, request, *args, **kwargs):
        # Fetch the latest user form data

        
        user_form_data = UserForm.objects.last()

        # Get threshold value from the session
        threshold_value = request.session.get('threshold_value', 34)

        # Calculate average parameters
        average_params = calculate_average_parameters(threshold_value)
        highest_params = calculate_highest_parameters(threshold_value)
        # Create a BytesIO buffer to receive PDF data
        buffer = BytesIO()

        # Create the PDF object, using BytesIO as its "file"
        doc = SimpleDocTemplate(buffer, pagesize=letter)

        # Define styles for the title and other elements
        styles = getSampleStyleSheet()
        normal_style = styles['Normal']
        heading_style = styles['Heading2']

        # Initialize elements for the PDF content
        elements = []

        # Title: Breast Health Report
        title_text = '<b><font size="16">Breast Health Report</font></b>'
        elements.append(Paragraph(title_text, styles['Title']))
        elements.append(Spacer(1, 12))

        # Patient ID to the right of the page
        patient_id_content = f'<b>Patient ID: {user_form_data.patient_id}</b>'
        elements.append(Paragraph(patient_id_content, normal_style))
        elements.append(Spacer(1, 12))
        elements.append(Paragraph('<hr/>', normal_style))

        # Sub-heading: General Details
        general_details_text = '<font color="blue"><b>General Details</b></font>'
        elements.append(Paragraph(general_details_text, heading_style))
        elements.append(Spacer(1, 6))

        # User Form Data
        user_form_content = [
            f'Name: {user_form_data.patient_name}',
            f'Email: {user_form_data.patient_email}',
            f'Doctor: {user_form_data.appointed_doctor}',
            f'Center: {user_form_data.center}',
            f'Age: {user_form_data.patient_age}',
            f'Report Generation Date: {user_form_data.report_generation_date.strftime("%Y-%m-%d")}',
            f'Gender: {user_form_data.patient_gender}',
            f'Scan Date: {user_form_data.scan_date.strftime("%Y-%m-%d")}',
        ]
        for line in user_form_content:
            elements.append(Paragraph(line, normal_style))
            elements.append(Spacer(1, 6))

        # Thermalytics Score
        elements.append(Paragraph('<font color="blue"><b>Thermalytics Score</b></font>', heading_style))
        elements.append(Spacer(1, 6))
        thermalytics_content = [
            f'Maximum Body Temperature: {average_params.get("MaxHotspotTemperature", "N/A")}°C',
            f'Minimum Body Temperature: {average_params.get("MinHotspotTemperature", "N/A")}°C',
            f'Mean ROI Temperature: {average_params.get("MeanROITemperature", "N/A")}°C',
            f'Percentage of ROI: {average_params.get("PercentageofROI", "N/A")}%',
            f'Aerolar Temperature Differences: {average_params.get("temperaturedifference", "N/A")}°C',
            f'Aerolar Symmetry: {average_params.get("AerolarSymmetry", "N/A")}%'
        ]
        for line in thermalytics_content:
            elements.append(Paragraph(line, normal_style))
            elements.append(Spacer(1, 6))
        # Thermal Analysis
        elements.append(Paragraph('<font color="blue"><b>Thermal Analysis</b></font>', styles['Heading2']))
        analysis_content = [
            ['Thermal Parameters', 'Highest Value'],
            ['Number of Hotspots', highest_params.get('NumberofHotspots', 'N/A')],
            ['Temperature Difference Around Boundaries', f"{highest_params.get('TemperatureDifferenceAroundBoundaries', 'N/A')}%"],
            ['Extent of Hotspots', f"{highest_params.get('ExtentofHotspots', 'N/A')}%"],
            ['Hotspot Shape', highest_params.get('HotspotShapes', 'N/A')],
            ['Threshold',threshold_value],
        ]
        thermal_analysis_table = Table(analysis_content, colWidths=[None, 120])
        thermal_analysis_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.lightblue),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ]))
        elements.append(thermal_analysis_table)
        # Thermal Images
        # Get the processed image paths
        processed_image_directory = os.path.join(settings.MEDIA_ROOT, 'uploads/processed_thermal_image')
        image_paths = self.get_image_paths(processed_image_directory)
        
        # Assuming all images are the same size, set the width and height
        image_width = 2.5 * inch
        image_height = 2 * inch
        spacer_height = 0.1 * inch

        # Create rows of images with 2 images per row
        image_rows = []
        for i in range(0, len(image_paths), 2):
            row = []
            for j in range(2):
                if i + j < len(image_paths):
                    img = ReportLabImage(image_paths[i + j], image_width, image_height)
                    row.append(img)
                else:
                    # If there is no image for the second column of the last row, add a blank string.
                    row.append('')
            image_rows.append(row)

        # Create a table with the image rows
        image_table = Table(image_rows)

        # Define the style for the table, if needed
        image_table_style = TableStyle([
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            # Add more styles if needed
        ])

        # Apply the style to the table
        image_table.setStyle(image_table_style)

        # Add the table to the elements list
        elements.append(image_table)
        
        # Build the PDF document
        doc.build(elements)
        buffer.seek(0)
        response = HttpResponse(buffer, content_type='application/pdf')
        response['Content-Disposition'] = 'attachment; filename="breast_health_report.pdf"'
        return response
# imran code
class GetJwtToken(APIView):
    global check_token
    permission_classes = ()
    serializer_class = LoginSerializer

    def post(self, request):
        username = request.data.get("username")
        password = request.data.get("password")
        # print(username, password, 'fafaf')
        # user = authenticate(request=request, email=email, password=password)
        user = authenticate(username=username, password=password)
        print(user, 'user')
        if user:
            serializer = LoginSerializer(
                data=request.data, context={"request": request}
            )
            try:
                serializer.is_valid(raise_exception=True)
            except Exception as e:
                raise InvalidToken(e.args[0])
            print(serializer.validated_data['access'], 'TESTING TOKEN')
            save_token(serializer.validated_data['access'], user)
            
            return Response(serializer.validated_data, status=status.HTTP_200_OK)
        
        return Response(
            {"message": "Invalid data"}, status=status.HTTP_401_UNAUTHORIZED
        )


def generate_random_patient_id(length=10):
    characters = string.digits
    return ''.join(random.choice(characters) for _ in range(length))

# def generate_random_patient_id(length=10):
#     return ''.join((random.randint(0, 9)) for _ in range(length))

class  ImageprocessViewAPI(ModelViewSet):
    global token_value
    serializer_class = UploadImgeSerializer
    queryset = ""

    def create(self, request, *args, **kwargs):
        
        serializer = self.get_serializer(data=request.data)
    
        serializer.is_valid(raise_exception=True)
        name = serializer.validated_data.get('name')
        print(name)


# reportgenerator
def generate_report(request, api_responses,report_data):
    response = HttpResponse(content_type='application/pdf')
    response['Content-Disposition'] = 'attachment; filename="report.pdf"'
    

    # doc = SimpleDocTemplate(response, pagesize=letter)
    # story = []

    # styles = getSampleStyleSheet()
    # report_title = Paragraph("Eye Report", styles['Title'])
    # story.append(report_title)
    # story.append(Spacer(1, 20))

    # # Get the template and render it
    # template = get_template('doctorpanel/report_template.html')
    # context = {
    #     'api_responses': api_responses,
    #     'report_data': report_data,
    # }
    # rendered_template = template.render(context)
    
    # # Convert the HTML content to ReportLab elements
    # story.append(KeepTogether(Paragraph(rendered_template, styles['Normal'])))

    # doc.build(story)
    c = canvas.Canvas(response, pagesize=letter)
    styles = getSampleStyleSheet()
        # Load the logo image
    logo_path = "static/images/report_logo.png"
    logo = ImageReader(logo_path)
    # Draw a white rectangle as the background of the logo
    c.setFillColor(white)
    c.rect(0.5*inch, letter[1] - 0.5*inch, 3*inch, 0.5*inch, fill=True, stroke=False)
    # Draw the logo over the white rectangle (adjust dimensions as needed)
    c.drawImage(logo,0.5*inch, letter[1] - 0.8*inch, width=3*inch, height=0.5*inch)
    # Add your content here using the canvas.drawString(), canvas.drawParagraph(), etc.
    #writing Headings
    # Add text under the logo
    text = "Eye Screen Service"
    c.setFont("Helvetica-Bold", 18)
    c.setFillColor(black)
    c.drawString(0.5*inch, letter[1] - 1.3*inch, text)
    second_line_text = "Diabetic Retinopathy Screening Report"
    c.setFont("Helvetica", 12)
    c.drawString(0.5*inch, letter[1] - 1.6*inch, second_line_text)


    #Drawing Line
    c.setStrokeColor(HexColor("#208295"))  # Use a blue color for the line
    c.setLineWidth(2)  # Set a thinner line width
    c.line(0.5*inch, letter[1] - 1.8*inch, 8*inch, letter[1] - 1.8*inch)

    #***Line below Box***
    c.setStrokeColor(HexColor("#208295"))  # Use a blue color for the line
    c.setLineWidth(17) 
    c.line(0.5*inch, letter[1] - 3.3*inch, 8*inch, letter[1] - 3.3*inch) # line used for heading fundus images used in examination

    #creatingbox
    box_width = 7.5*inch
    box_height = 1*inch
    box_x = 0.5*inch
    box_y = letter[1] - 3.0*inch
    box_border_width = 1


    c.setStrokeColor(HexColor("#000000"))  # Black color for the border
    c.setLineWidth(box_border_width)
    c.rect(box_x, box_y, box_width, box_height, fill=False, stroke=True)

    #Patient details
    patient_mrn = report_data['patient_id']
    current_datetime = datetime.now()
    imaging_datetime = current_datetime.strftime('%a %d %b %H:%M:%S %Y')
    patient_name = report_data['patient_name']
    report_datetime = current_datetime.strftime('%a %d %b %H:%M:%S %Y')
    doctor_name = report_data['username']

    # Add fields inside the box
    c.setFont("Helvetica-Bold", 12)
    c.setFillColor(black)
    field_x_left = box_x + 10
    field_x_right = box_x + box_width - 250
    field_y = box_y + 10

    c.drawString(field_x_left, field_y + 40, f"Patient MRN: {patient_mrn}")
    c.drawString(field_x_right - 30, field_y + 40, f"Imaging Date-Time: {imaging_datetime}")

    c.drawString(field_x_left, field_y + 20, f"Patient Name: {patient_name}")
    c.drawString(field_x_right - 30, field_y + 20, f"Report Date-Time: {report_datetime}")

    c.drawString(field_x_left, field_y + 1, f"Doctor Name: {doctor_name}")

    #************Reports Start************

    text = "FUNDUS IMAGES USED IN EXAMINATION"
    c.setFont("Helvetica-Bold", 18)
    c.setFillColor(white)
    c.drawCentredString(4.3*inch, letter[1] - 3.4*inch, text)

    #Box for generating Images

    box_width = 7.5*inch
    box_height = 3.77*inch
    box_x = 0.5*inch
    box_y = letter[1] - 7.2*inch
    box_border_width = 1


    c.setStrokeColor(HexColor("#000000"))  # Black color for the border
    c.setLineWidth(box_border_width)
    c.rect(box_x, box_y, box_width, box_height, fill=False, stroke=True)
    
    # Load the two images
    image1_path = report_data.get('left_eye_image_path', '')
    image2_path = report_data.get('right_eye_image_path', '')
    # print(image1_path)
    # print(image2_path)
    # print(report_data)


    image1 = None
    image2 = None
    

    if image1_path:
        image1 = ImageReader(image1_path)

    if image2_path:
        image2 = ImageReader(image2_path)

    # Calculate image positions and dimensions
    image_width = 2.0*inch
    image_height = 2.0*inch
    image1_x = 1.1*inch
    image_y = letter[1] - 6.3*inch
    image2_x = image1_x + image_width + 0.5*inch

    # Draw the images on the canvas
    if image1:
        c.drawImage(image1, image1_x, image_y, width=image_width, height=image_height)

    if image2:
        c.drawImage(image2, image2_x + 20, image_y, width=image_width, height=image_height)

    #*************ICD codes text**************
    #ICD codes fetching
    left_eye_icd = report_data['left_result']
    right_eye_icd = report_data['right_result']
    #Left Image
    if image1:
        c.setFont("Helvetica-Bold", 10)
        c.setFillColor(green)
        c.drawCentredString(2.5*inch, letter[1] - 6.6*inch, f"OS(LE): {left_eye_icd}")

    #Right Image ICD code
    if image2:
        c.setFont("Helvetica-Bold", 10)
        c.setFillColor(green)
        c.drawCentredString(6.2*inch, letter[1] - 6.6*inch, f"OS(RE): {right_eye_icd}")

    #conditional line
    text = "*Image orientation and labeling is for reference purpose only and should not be used for diagnostic purpose"
    c.setFont("Helvetica", 8)
    c.setFillColor(black)
    c.drawString(0.8*inch, letter[1] - 7*inch, text)

    #*********Diagnosis***********
    #creating a label for heading
    c.setStrokeColor(HexColor("#208295"))  # Use a blue color for the line
    c.setLineWidth(17) 
    c.line(0.5*inch, letter[1] - 7.5*inch, 8*inch, letter[1] - 7.5*inch)
    
    #diagnosis heading
    text = "DIAGNOSIS AND ICD-10 CODES"
    c.setFont("Helvetica-Bold", 18)
    c.setFillColor(white)
    c.drawCentredString(4.3*inch, letter[1] - 7.6*inch, text)
    red_text_style = ParagraphStyle(
    "RedText",
    parent=styles["Normal"],
    textColor=colors.red  # Set the text color to red
    )
    icd_desc_list = json.loads(report_data['icd_desc'])
    icd_codes_list = json.loads(report_data['icd_codes'])
    last_digit = report_data['diabetes_type'][-1]
    filtered_icd_codes = [code for code in icd_codes_list if code.endswith(last_digit)]
    print(filtered_icd_codes, "CHECK",icd_codes_list)
    if len(icd_codes_list) == 2:
        last_digit = report_data['diabetes_type'][-1]
        # filtered_icd_codes = [code for code in icd_codes_list if code[-2] == last_digit]
    elif len(icd_codes_list) == 1:
        filtered_icd_codes = [code for code in icd_codes_list]
    #Daignosis table
    table_data = [ 
        ("Condition", "Condition 2"),
        (f"Diagnosis {filtered_icd_codes}", [Paragraph(desc, styles["Normal"]) for desc in icd_desc_list]),
        ("Screening Result", Paragraph(report_data['summary'],
               red_text_style))
    ]
    colWidths = [1.9*inch, 5.6*inch, 5.6*inch] 

    table_style = [
        ("BACKGROUND", (0, 0), (0, -1), HexColor("#208295")),
        ("TEXTCOLOR", (0, 0), (0, -1), white),
        ("ALIGN", (0, 0), (0, -1), "CENTER"),
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"), 
        ("BOTTOMPADDING", (0, 0), (0, -1), 10),                         #bottom pading
        ("BACKGROUND", (1, 0), (-1, -1), HexColor("#ECECEC")),
        ("GRID", (0, 0), (-1, -1), 1, black), 
        ("ALIGN", (1,0), (-1,-1), "CENTER"),
    ]

    # Create the table
    table = Table(table_data, colWidths=colWidths, rowHeights=[0.4*inch, 0.7*inch, 0.5*inch])
    table.setStyle(TableStyle(table_style))

    # Draw the table on the canvas
    table_x = 0.5*inch
    table_y = letter[1] - 9.3*inch
    table.wrapOn(c, 0, 0)
    table.drawOn(c, table_x, table_y)

    #Recomendation heading
    c.setStrokeColor(HexColor("#208295"))  # Use a blue color for the line
    c.setLineWidth(17) 
    c.line(0.5*inch, letter[1] - 9.72*inch, 8*inch, letter[1] - 9.72*inch)

    text = "RECOMMENDATION"
    c.setFont("Helvetica-Bold", 18)
    c.setFillColor(white)
    c.drawCentredString(4.3*inch, letter[1] - 9.82*inch, text)

    #BOX FOR RECOMMENDATION
    box_rwidth = 7.5*inch
    box_rheight = 1*inch
    box_rx = 0.5*inch
    box_ry = letter[1] - 10.6*inch
    box_rborder_width = 1

    c.setStrokeColor(HexColor("#000000"))  # Black color for the border
    c.setLineWidth(box_rborder_width)
    c.rect(box_rx, box_ry, box_rwidth, box_rheight, fill=False, stroke=True)

    c.setFont("Helvetica-Bold", 11)
    c.setFillColor(black)
    field_rx_left = box_rx + 10
    field_ry = box_ry + 10
    current_year = current_datetime.year
    is_leap_year = (current_year % 4 == 0 and current_year % 100 != 0) or (current_year % 400 == 0)

# Calculate the number of days in the year (adjust for leap years)
    days_in_a_year = 366 if is_leap_year else 365

# Calculate the follow-up date (next year, one day before current)
    followup_datetime = current_datetime + timedelta(days=days_in_a_year + 1)
    formatted_followup_date = followup_datetime.strftime('%d-%b-%Y')
    point_1 = f"1. Consult an Ophthalmologist within 12 months, preferably before {formatted_followup_date}"
    
    before_date = current_datetime + timedelta(days=days_in_a_year - 8)
    formatted_before_date = before_date.strftime('%d-%b-%Y')
    after_date = current_datetime + timedelta(days=days_in_a_year + 8)
    formatted_after_date = after_date.strftime('%d-%b-%Y')
    point_2 = f"2. Get your eyes screened for DR on {report_data['username']} at Clinic between {formatted_before_date} & {formatted_after_date}"
    c.drawString(field_rx_left, field_ry + 20, f"{point_1}")
    c.drawString(field_rx_left, field_ry + 5, f"{point_2}")
    c.showPage()
    #***************************** page 1 ends ****************************

    #Disclamer
    c.setStrokeColor(HexColor("#208295"))  # Use a blue color for the line
    c.setLineWidth(17) 
    c.line(0.5*inch, letter[1] - 0.5*inch, 8*inch, letter[1] - 0.5*inch)

    text = "DISCLAIMER"
    c.setFont("Helvetica-Bold", 18)
    c.setFillColor(white)
    c.drawCentredString(4.3*inch, letter[1] - 0.6*inch, text)

    #PARAGRAPH

    content = ("This report is generated by an Artificial Intelligence software, an initiative of BioScan AI Innovative\n"
            "Solutions, and is for informational purposes only. This report is indicative and assistive in nature and is\n"
            "not a replacement for medical advice, diagnosis or treatment. Accrodingly, before taking any action upon such information, we encourage you to consult with the appropriate professionals, We do not provide any medical/ health advice.")
    # Create a Paragraph object with the 'Normal' style
    modified_style = styles['Normal']

    modified_style.fontSize = 8
    modified_style.leading = 10
    paragraph = Paragraph(content, style=modified_style)

    # Convert inch measurements to points
    x_pos = 0.5 * inch  # X position in inches (1.5 inches)
    y_pos = 2.44 * inch  # Y position in inches (10 inches)
    width = 7 * inch   # Width in inches (5 inches)
    height = 1 * inch  # Height in inches (4 inches)

    # Draw the paragraph on the canvas
    paragraph.wrapOn(c, width, height)
    paragraph.drawOn(c, x_pos, letter[1] - y_pos)

    #DISCLAIMER IMAGE
    # image1_path = "D:/maskottchen/zia-project/pdf_report/disclaimer.png"
    # image1_path = "dristieye/static/images/doctor.png"
    # image1 = ImageReader(image1_path)

    # Calculate image positions and dimensions
    image_width = 6.5*inch
    image_height = 4.0*inch
    image1_x = 0.8*inch
    image_y = letter[1] - 6.6*inch

    # Draw the images on the canvas
    # c.drawImage(image1, image1_x, image_y, width=image_width, height=image_height)
 
   # Save the canvas and finalize the PDF
    
    c.showPage()
    c.save()

    # subject = 'Verify your email address'
    # message = 'Your Report is Ready'
    # print(request.user.email,'EMAIL')
    # send_mail(subject, message, 'boredstuff2021@gmail.com', [request.user.email])
    # Send an email with the PDF attached

    # Create a PDF file
    # c = canvas.Canvas(pdf_file)
    
    subject = 'Your Report is Ready'
    message = 'Your Report is Ready'
    from_email = 'boredstuff2021@gmail.com'
    recipient_email = request.user.email 

    # Create an email message with the PDF as an attachment
    email = EmailMessage(subject, message, from_email, [recipient_email])
    email.attach('report.pdf', response.getvalue(), 'application/pdf')
    # Send the email
    email.send()
    return response
    
        



# Pending Request (doctor)
def pendingRequest(request):
    response = requests.get('http://127.0.0.1:8000/credit_requset_list/')
    doctor_id=[]
    doctor_names=[]
    req_date= []
    re_cre_val = []
    
    # Check if the request was successful (status code 200)
    if response.status_code == 200:
        # Parse the JSON response
        data = response.json()
        # data = json.loads(response.text)
        print(data)
        data = json.loads(data)
        # print(type(data), 'TYPE')
        for item in data:
            # Access the 'fields' dictionary within each object and then the 'doctor' field
            doctor = item['fields']['doctor']
            # Retrieve the doctor object from the database using the ID
            doctor = Doctor.objects.get(id=doctor)

            # Access the 'fullname' attribute of the doctor object
            # doctor_name = doctor.fullname

            # doctor_names.append(doctor_name)

            r_date = item['fields']['request_date']
            r_credit = item['fields']['credit_value']
            doctor_id.append(doctor)
            req_date.append(r_date)
            re_cre_val.append(r_credit)
            # print(doctor_id,re_cre_val, req_date)
        data = list(zip(doctor_id, req_date, re_cre_val))
    return render(request,'doctorpanel/pendingRequest.html', {'data': data,})

# Success Request (doctor)
def successRequest(request):
    return render(request,'doctorpanel/successRequest.html')

# Rejected Request (doctor)
def rejectedRequest(request):
    return render(request,'doctorpanel/rejectedRequest.html')

# Search Reports (doctor)
def searchReports(request):
    return render(request,'doctorpanel/searchReports.html')

# Chart (doctor)
def charts(request):
    patients_by_day = list(patients_by_weekday())
    print(patients_by_day)
    context={
        'patients_by_day':patients_by_day
    }
    return render(request,'doctorpanel/charts.html', context)

# Request Credit (doctor)
def requestCredit(request):
    if request.method == 'POST':
        
        form = CreditRequestForm(request.POST)
        if form.is_valid():
            # Get the currently logged-in doctor
            doctor = request.user
            
            # Extract credit_value from the form
            credit_value = form.cleaned_data['credit_value']
            
            # Create a CreditRequest associated with the logged-in doctor
            CreditRequest.objects.create(doctor=doctor, credit_value=credit_value)
            
            user = request.user

            if user.is_superuser:
                return redirect('admin_home')
            # Redirect to a success page or wherever you need to go
            return redirect('doctorDashboard')
                
    else:
        form = CreditRequestForm()

    return render(request, 'doctorpanel/requestCredit.html', {'form': form})
    # return render(request,'doctorpanel/requestCredit.html')

def error(request):
    return render(request,'error.html')

def mail_error(request):
    return render(request,'mail_error.html')

def general_error(request):
    return render(request,'general_error.html')