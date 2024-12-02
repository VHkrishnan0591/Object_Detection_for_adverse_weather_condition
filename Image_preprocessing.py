import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import image_dehazer

class ImageDehazer():
    def __init__(self):
        pass
    
    def image_enhancer(self,image):
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

        # Split the LAB channels
        l, a, b = cv2.split(lab)

        # Apply CLAHE to the L (lightness) channel
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l_clahe = clahe.apply(l)

        # Merge the enhanced L channel back with a and b
        lab_clahe = cv2.merge((l_clahe, a, b))
        return cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)
        
    def increase_brightness(self,img, value=50):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)  # Convert to HSV
        h, s, v = cv2.split(hsv)
        v = cv2.add(v, value)  # Increase brightness
        v = np.clip(v, 0, 255)  # Ensure values stay within valid range
        final_hsv = cv2.merge((h, s, v))
        return cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    
    def dehazer(self,image):
        HazeImg = image
        HazeCorrectedImg, HazeMap = image_dehazer.remove_haze(HazeImg)	
        return cv2.cvtColor(HazeCorrectedImg, cv2.COLOR_BGR2RGB)
    
class ImageDerained():
    def __init__(self):
        pass

    def guided_filter(self,I, p, r, eps):
    
        # Ensure the input is in float format
        I = I.astype(np.float32) / 255.0
        p = p.astype(np.float32) / 255.0

        # Compute the means of I, p, and I*p
        mean_I = cv2.boxFilter(I, ddepth=-1, ksize=(r, r))
        mean_p = cv2.boxFilter(p, ddepth=-1, ksize=(r, r))
        mean_Ip = cv2.boxFilter(I * p, ddepth=-1, ksize=(r, r))

        # Compute covariance of I and p: cov_Ip = E(Ip) - E(I)E(p)
        cov_Ip = mean_Ip - mean_I * mean_p

        # Compute the mean of I^2
        mean_II = cv2.boxFilter(I * I, ddepth=-1, ksize=(r, r))

        # Compute variance of I: var_I = E(I^2) - (E(I))^2
        var_I = mean_II - mean_I * mean_I

        # Compute the coefficients a and b
        a = cov_Ip / (var_I + eps)
        b = mean_p - a * mean_I

        # Compute the mean of a and b
        mean_a = cv2.boxFilter(a, ddepth=-1, ksize=(r, r))
        mean_b = cv2.boxFilter(b, ddepth=-1, ksize=(r, r))

        # Compute the output q
        q = mean_a * I + mean_b

        # Scale the output back to the original image range
        q = (q * 255).clip(0, 255).astype(np.uint8)

        return q
    
    def edge_enhancement(self,low_freq_image, omega):
        # Compute the gradients using the Sobel operator
        gradient_x = cv2.Sobel(low_freq_image, cv2.CV_64F, 1, 0, ksize=5)  # Gradient in the x-direction
        gradient_y = cv2.Sobel(low_freq_image, cv2.CV_64F, 0, 1, ksize=5)  # Gradient in the y-direction

        # Compute the gradient magnitude (edge strength)
        gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
        gradient_magnitude = np.power(gradient_magnitude, 1.5)

        # Normalize the gradient magnitude to the range [0, 255]
        gradient_magnitude = (gradient_magnitude / np.max(gradient_magnitude) * 255).astype(np.uint8)

        # Apply the enhancement formula: I*_LF = I_LF + ω * ∇I_LF
        enhanced_image = low_freq_image.astype(np.float32) + omega * gradient_magnitude

        # Clip the values to ensure they remain in the valid range [0, 255]
        enhanced_image = np.clip(enhanced_image, 0, 255).astype(np.uint8)

        for _ in range(3):  # Apply 3 iterations for stronger effect
            gradient_x = cv2.Sobel(enhanced_image, cv2.CV_64F, 1, 0, ksize=3)
            gradient_y = cv2.Sobel(enhanced_image, cv2.CV_64F, 0, 1, ksize=3)
            gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
            gradient_magnitude = (gradient_magnitude / np.max(gradient_magnitude) * 255).astype(np.uint8)
            enhanced_image = enhanced_image.astype(np.float32) + omega * gradient_magnitude
            enhanced_image = np.clip(enhanced_image, 0, 255).astype(np.uint8)

        return enhanced_image

    def derainer(self,image): 
        input_image = image
        radius = 10  # Size of the local window
        epsilon = 0.3 ** 2  # Regularization parameter

        # Apply the guided filter to get the low-frequency component
        low_freq_image = self.guided_filter(input_image, input_image, radius, epsilon)

        # Extract the high-frequency component by subtracting the low-frequency part
        high_freq_image = cv2.subtract(input_image, low_freq_image)

        # Enhancing the low-frequency image
        enhanced_low_freq_img = self.edge_enhancement(low_freq_image,0.1)

        # Apply the guided filter to get the low-frequency component
        high_freq_bg_part = self.guided_filter(enhanced_low_freq_img,high_freq_image, radius, epsilon)
        recovered_image = cv2.add(low_freq_image,high_freq_bg_part)

        # Recovering lost details through image minimisation and weighted sum
        if input_image.shape != recovered_image.shape:
            print("Input and recovered images must have the same dimensions.")
        else: 
            print("they are at same shape")
            corrected_image = np.minimum(recovered_image, input_image)
        
        if corrected_image.shape != recovered_image.shape:
            print("Input and recovered images must have the same dimensions.")
        else: 
            print("they are at same shape")
            dst = cv2.addWeighted(corrected_image,0.8,recovered_image,0.2,0)

        # Apply the guided filter to get the low-frequency component
        final_derained_image = self.guided_filter(corrected_image,dst, radius, epsilon)

        return cv2.cvtColor(final_derained_image, cv2.COLOR_BGR2RGB)
        
