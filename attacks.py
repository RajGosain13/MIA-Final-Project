from PIL import Image, ImageEnhance
import numpy as np
import cv2

def gaussian_noise(image, mean=0, std=10):
    noise = np.random.normal(mean, std, image.shape)
    noisy_image = image + noise
    return np.clip(noisy_image, 0, 255)

def crop(image, ratio=0.1):
    length, width = image.shape
    crop_length = int(length*ratio)
    crop_width = int(width*ratio)

    return image[crop_length:length-crop_length, crop_width:width-crop_width]

def compression(image, quality=30):
    pil_image = Image.fromarray(image.astype(np.uint8))
    pil_image.save('temp.png', 'JPEG', quality=quality)
    return np.array(Image.open('temp.png').convert('L'))

def blur(image, kernel_size=3):
    return cv2.GaussianBlur(image.astype(np.uint8), (kernel_size, kernel_size), 0)

def resize(image, scale=0.5):
    length, width = image.shape
    resized = cv2.resize(image, (int(width*scale), int(length*scale)), interpolation=cv2.INTER_AREA)
    return cv2.resize(resized, (width, length), interpolation=cv2.INTER_LINEAR)

def change_contrast(image, factor=0.5):
    pil_img = Image.fromarray(image.astype(np.uint8))
    enhancer = ImageEnhance.Contrast(pil_img)
    return np.array(enhancer.enhance(factor=factor).convert('L'))

