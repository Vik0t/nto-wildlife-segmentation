import cv2
import numpy as np

# Load the image
image = cv2.imread('govno/0a21f6a0d0885367a46601cbc8b3d854.JPG', cv2.IMREAD_GRAYSCALE)

# Apply histogram equalization to improve contrast
equalized_image = cv2.equalizeHist(image)

# Apply a Gaussian blur to reduce noise
blurred_image = cv2.GaussianBlur(equalized_image, (5, 5), 0)

# Apply sharpening filter
kernel = np.array([[0, -1, 0], [-1, 5,-1], [0, -1, 0]])
sharpened_image = cv2.filter2D(blurred_image, -1, kernel)

# Save the enhanced image
cv2.imwrite('enhanced_image.jpg', sharpened_image)

# Display the original and enhanced images
cv2.imshow('Original Image', image)
cv2.imshow('Enhanced Image', sharpened_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
