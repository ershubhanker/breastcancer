from django.shortcuts import render, redirect
from django.contrib import messages 
from .models import Image
from PIL import Image as PILImage
import base64
import json
from django.core.files.base import ContentFile
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import os
from django.conf import settings
import cv2
import numpy as np
from django.core.files.storage import FileSystemStorage
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from .temperature_cal import get_temperature_from_pixel
from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login,logout
from .forms import DoctorRegistrationForm,DoctorLoginForm
from django.views.decorators.csrf import csrf_exempt
  
# signup page
def register(request):
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
    return render(request, 'myapp/register.html', {'form': form})

# login page
def login(request):
    if request.method == 'POST':
        form = DoctorLoginForm(data=request.POST)
        if form.is_valid():
            user = form.get_user()
            login(request, user)
            if user.is_staff:
                return redirect('admin_home')  # Redirect to staff-specific page
            else:
                return redirect('index')  # Redirect to regular home page
        else:
            messages.error(request, 'Invalid Username or Password. Please try again.')
    else:
        form = DoctorLoginForm()
    return render(request, 'myapp/login.html', {'form': form})

# logout page
def user_logout(request):
    logout(request)
    return redirect('login')


# imran code
def upload_image(request):
    if request.method == 'POST':
        # Get the name of the image from the POST data. The name attributes of the input fields in the HTML are the keys.
        image_field_name = next(iter(request.FILES))
        image_file = request.FILES.get(image_field_name)

        if image_file:
            # Save the image under the appropriate title
            Image.objects.create(title=image_field_name.replace('_', ' ').capitalize(), image=image_file)
            messages.success(request, "Image uploaded successfully.") 
            return redirect('upload-image')

    # If not POST or if no file is selected, render the page with the form
    return render(request, 'myapp/upload-image.html')

def home(request):
    return render(request, 'myapp/home.html')  # Make sure this template path is correct.

def annotate_image(request):
    images = Image.objects.all()
    images = Image.objects.filter(is_annotated=False)
    return render(request, 'myapp/annotate-image.html', {'images': images})
@csrf_exempt
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
def thermal_image_view(request):
    # Handling the AJAX request for temperature calculation
    if request.method == 'GET' and all(k in request.GET for k in ('x', 'y', 'displayWidth', 'displayHeight')):
        x = float(request.GET['x'])
        y = float(request.GET['y'])
        display_width = float(request.GET['displayWidth'])
        display_height = float(request.GET['displayHeight'])

        # Load the thermal image to get its actual dimensions
        thermal_image_path = os.path.join(settings.MEDIA_ROOT, 'new_directory/', 'callibaration_image.png')
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
        # Directory where annotated images are stored
        annotated_images_dir = os.path.join(settings.MEDIA_ROOT, 'uploads', 'new_directory')

        # List all files in the annotated images directory
        annotated_image_files = os.listdir(annotated_images_dir)

        annotated_image_urls = [
        os.path.join(settings.MEDIA_URL, 'uploads/', 'new_directory/', filename)
        for filename in annotated_image_files if filename.endswith('.png')
        ]

        # Render the template with the image URLs
        return render(request, 'myapp/thermal-image.html', {'annotated_images': annotated_image_urls})

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
# def analyze_thermal_image(image_path, output_path):
#     image = cv2.imread(image_path)
#     # Convert to grayscale
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
#     # Convert to HSV for red color detection
#     hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
#     # Thresholds to identify the black boundary and the red hotspots
#     _, black_mask = cv2.threshold(hsv[:, :, 2], 50, 255, cv2.THRESH_BINARY_INV)
#     _, red_mask = cv2.threshold(hsv[:, :, 0], 150, 255, cv2.THRESH_BINARY)

#     # Find contours in the black mask
#     contours, _ = cv2.findContours(black_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     breast_contour = max(contours, key=cv2.contourArea)

#     # Create a mask from the breast contour
#     breast_mask = np.zeros_like(hsv[:, :, 0])
#     cv2.drawContours(breast_mask, [breast_contour], -1, 255, -1)

#     # Use the breast mask to isolate red regions within the black boundary
#     red_within_breast = cv2.bitwise_and(red_mask, red_mask, mask=breast_mask)

#     # Find contours of the red areas within the breast mask
#     red_contours, _ = cv2.findContours(red_within_breast, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#     # Convert to HSV for red color detection
#     hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

#     # Define the range for red color
#     # Adjust the range as necessary for your specific image
#     lower_red = np.array([0, 120, 70])
#     upper_red = np.array([10, 255, 255])
#     lower_red2 = np.array([0, 120, 70])
#     upper_red2 = np.array([10, 255, 255])

#     # Create masks for the red color
#     mask_red1 = cv2.inRange(hsv, lower_red, upper_red)
#     mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
#     mask_red = cv2.add(mask_red1, mask_red2)

#     # Use the breast mask to isolate red regions within the black boundary
#     red_within_breast = cv2.bitwise_and(mask_red, mask_red, mask=breast_mask)

#     # Find contours of the red areas within the breast mask
#     red_contours, _ = cv2.findContours(red_within_breast, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#     # Draw contours on the original image around the red areas
#     cv2.drawContours(image, red_contours, -1, (255, 0, 0), 2)  # Blue lines for visibility

#     # Save the resulting image instead of displaying it
#     cv2.imwrite(output_path, image)

def calculate_thermal_image(image_path, calibration_image_path):
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
        def rgb_to_temperature(pixel_color, color_temp_mapping):
            pixel_color_np = np.array(pixel_color, dtype=np.int16)
            min_distance = float('inf')
            closest_temp = None
            for color, temp in color_temp_mapping:
                distance = np.linalg.norm(pixel_color_np - np.array(color, dtype=np.int16))
                if distance < min_distance:
                    min_distance = distance
                    closest_temp = temp
            return closest_temp

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
        
        # Display the results
        for cross in crosses_info:
            diff = crosses_info[0]['temperature']-crosses_info[1]['temperature']
            avg = (crosses_info[0]['temperature']+crosses_info[1]['temperature'])/2
            ans = round(diff/avg,2) * 100
            return diff,ans

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
    hotspot_extent = round((hotspot_area / roi_area) * 100, 2) if roi_area > 0 else 0
    percentage = calculate_percentage(image_path)
    temperature_difference, shape_description = calculate_temperature_difference(hsv, calibration_column)
    aerolar_symmetry, aerolar_difference = aerolar_difference(image_path, calibration_image_path)
    parameters = {
        "NumberofHotspots": num_hotspots,
        "TemperatureDifferenceAroundBoundaries": temperature_difference,
        "HotspotShapes": shape_description,
        "ExtentofHotspots": hotspot_extent,
        "MaxHotspotTemperature": max_temp,
        "MinHotspotTemperature": min_temp,
        "MeanROITemperature": mean_temp,
        "PercentageofROI": percentage,
        "temperaturedifference":round(aerolar_difference,2),
        "AerolarSymmetry" : round(aerolar_symmetry,2)
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
def thermal_parameters(request):
    threshold_value = request.session.get('threshold_value', 34)
    # Directory where the original images are stored
    input_directory_path = os.path.join(settings.MEDIA_ROOT, 'uploads', 'new_directory')

    # Directory where you want to save the processed images
    processed_directory_path = os.path.join(settings.MEDIA_ROOT, 'uploads', 'processed_thermal_image')

    # Calibration image path
    calibration_image_path = os.path.join(settings.MEDIA_ROOT, 'new_directory', 'callibaration_image.png')

    # Create the processed images directory if it doesn't exist
    if not os.path.exists(processed_directory_path):
        os.makedirs(processed_directory_path)
    
    thermal_data = []

    for filename in os.listdir(input_directory_path):
        if filename.endswith('.png'):  # Assuming you want to process PNG images
            input_image_path = os.path.join(input_directory_path, filename)
            output_image_name = f'processed_{filename}'
            output_image_path = os.path.join(processed_directory_path, output_image_name)

            # Process the image and save it in the new directory
            analyze_thermal_image(input_image_path, output_image_path,calibration_image_path,threshold_value)

            # Calculate thermal parameters
            parameters = calculate_thermal_image(input_image_path, calibration_image_path)
            # Adding the threshold value inside the parameters
            parameters['threshold'] = threshold_value 

            # Create the URL for the processed image
            processed_image_url = os.path.join(settings.MEDIA_URL, 'uploads', 'processed_thermal_image', output_image_name)

            thermal_data.append({
                'processed_image_url': processed_image_url,
                'parameters': parameters
            })

    context = {'thermal_data': thermal_data}
    return render(request, 'myapp/thermal_parameters.html', context)

# def thermal_parameters(request):
#     # Directory where the original images are stored
#     input_directory_path = os.path.join(settings.MEDIA_ROOT, 'uploads', 'new_directory')
    
#     # Directory where you want to save the processed images
#     processed_directory_path = os.path.join(settings.MEDIA_ROOT, 'uploads', 'processed_thermal_image')
    
    
#     # Create the processed images directory if it doesn't exist
#     if not os.path.exists(processed_directory_path):
#         os.makedirs(processed_directory_path)
    
#     processed_images_urls = []
    
#     for filename in os.listdir(input_directory_path):
#         if filename.endswith('.png'):  # Assuming you want to process PNG images
#             input_image_path = os.path.join(input_directory_path, filename)
#             output_image_name = f'processed_{filename}'
#             output_image_path = os.path.join(processed_directory_path, output_image_name)

#             # Process the image and save it in the new directory
#             analyze_thermal_image(input_image_path, output_image_path)
#             # Create the URL for the processed image
#             processed_image_url = os.path.join(settings.MEDIA_URL, 'uploads', 'processed_thermal_image', output_image_name)
#             processed_images_urls.append(processed_image_url)

#     context = {'processed_images_urls': processed_images_urls}
#     return render(request, 'myapp/thermal_parameters.html', context)