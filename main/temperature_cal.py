# import numpy as np
# import cv2
# import matplotlib.pyplot as plt

# image = cv2.imread('E:/maskotten/breast images/Breast Sample Images/Breast Sample Images/breastcancerbackend/python temperature/annotated_images/1_front.png') 
# cv2.imshow('Original', image) 
# cv2.waitKey(0) 
  
# # Use the cvtColor() function to grayscale the image 
# gray= cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
# cv2.imshow('Original', gray) 
# cv2.waitKey(0)   



# # Load the calibration image
# calibration_image_path = 'E:/maskotten/breast images/Breast Sample Images/Breast Sample Images/breastcancerbackend/python temperature/annotated_images/callibaration_image.png'
# calibration_image = cv2.imread(calibration_image_path)

# # Load the thermal image
# thermal_image_path = 'E:/maskotten/breast images/Breast Sample Images/Breast Sample Images/breastcancerbackend/python temperature/annotated_images/1_front.png'
# thermal_image = cv2.imread(thermal_image_path)

# # Assuming the calibration image is a vertical gradient of colors corresponding to the temperature scale.
# # We will extract the column of pixels that runs down the center of this gradient and use it to map colors to temperatures.

# # Get the middle column of pixels in the calibration image
# middle_column = calibration_image[:, calibration_image.shape[1] // 2, :]

# # Get the temperature values from the top and bottom of the calibration scale
# top_temperature = 37.62  # Extracted from the top of the calibration image
# bottom_temperature = 25.78  # Extracted from the bottom of the calibration image
# temperature_range = top_temperature - bottom_temperature

# # Function to map a pixel's RGB values to a temperature
# def rgb_to_temperature(rgb_pixel):
#     # Calculate the differences array
#     differences = np.sqrt(np.sum((middle_column - rgb_pixel) ** 2, axis=1))
#     # Find the index of the smallest difference
#     index_of_smallest_difference = np.argmin(differences)
#     # Map the index to a temperature
#     temperature = top_temperature - (temperature_range * (index_of_smallest_difference / len(middle_column)))
#     return temperature

# # Function to handle the mouse click on the thermal image
# def handle_mouse_click(event, x, y, flags, param):
#     if event == cv2.EVENT_LBUTTONDOWN:
#         # Get the RGB values of the clicked pixel
#         clicked_pixel = thermal_image[y, x, :]
#         # Convert the RGB values to temperature
#         temperature = rgb_to_temperature(clicked_pixel)
#         print(f"Clicked Position: ({x},{y}) - Temperature: {temperature:.2f}°C")

# # Set up the OpenCV window and mouse callback function
# cv2.namedWindow('Thermal Image')
# cv2.setMouseCallback('Thermal Image', handle_mouse_click)

# # Display the image and wait for a click
# cv2.imshow('Thermal Image', thermal_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
import numpy as np
import cv2

def get_closest_temperature_index(clicked_pixel, gradient_column):
    differences = np.sqrt(np.sum((gradient_column - clicked_pixel) ** 2, axis=1))
    return np.argmin(differences)

def get_temperature_from_pixel(x, y, calibration_image_path, thermal_image_path, top_temperature, bottom_temperature):
    # Load images
    calibration_image = cv2.imread(calibration_image_path)
    thermal_image = cv2.imread(thermal_image_path)

    if calibration_image is None or thermal_image is None:
        print("Error: Unable to load images.")
        return None

    # Extract a column of pixels from the middle of the calibration image
    middle_column = calibration_image[:, int(calibration_image.shape[1] / 2), :]
    # Get RGB value of the clicked pixel in the thermal image
    clicked_pixel = thermal_image[y, x, :]
    print(f"Clicked Pixel (RGB): {clicked_pixel}")
    # Find the closest matching color in the calibration image
    index = get_closest_temperature_index(clicked_pixel, middle_column)
    print("The index: ",index)
    # Map the index to a temperature
    temperature_range = top_temperature - bottom_temperature
    temperature = top_temperature - (temperature_range * (index / len(middle_column)))

    return temperature
