from PIL import Image, ImageEnhance
import numpy as np
import cv2
import random

def gaussian_noise(image, mean=0, std=10):
    noise = np.random.normal(mean, std, image.shape)
    noisy_image = image + noise
    return np.clip(noisy_image, 0, 255)

def salt_and_pepper(image, amt=0.01):
    noisy = np.copy(image)
    pixels = int(amt * image.size)
    coords = [(random.randint(0, image.shape[0]-1), random.randint(0, image.shape[1]-1)) for n in range(pixels)]

    for x, y in coords:
        noisy[x, y] = 0 if random.random() < 0.5 else 255
    return noisy

def compression(image, quality=30):
    pil_image = Image.fromarray(image.astype(np.uint8))
    pil_image.save('temp.png', 'JPEG', quality=quality)
    return np.array(Image.open('temp.png').convert('L'))

def blur(image, kernel_size=3):
    return cv2.GaussianBlur(image.astype(np.uint8), (kernel_size, kernel_size), 0)

def histogram_equalization(image):
    return cv2.equalizeHist(image.astype(np.uint8))

def rotate(image, angle=90):
    height, width = image.shape
    matrix = cv2.getRotationMatrix2D((width/2, height/2), angle, 1)
    return cv2.warpAffine(image, matrix, (width, height), borderMode=cv2.BORDER_REFLECT)
