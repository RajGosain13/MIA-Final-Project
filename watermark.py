import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def generate(user, length):
    random_seed = sum([ord(c) for c in user])
    rng = np.random.default_rng(random_seed)
    return rng.normal(0, 1, length)

def embed(image, watermark, alpha=0.1, block_size=32):
    image = image.astype(np.float64)
    length, width = image.shape
    watermarked_image = np.copy(image)

    index = 0
    for i in range(0, length, block_size):
        for j in range(0, width, block_size):
            if index >= len(watermark):
                break

            block = image[i:i+block_size, j:j+block_size]
            if block.shape != (block_size, block_size):
                continue

            dft = np.fft.fft2(block)
            magnitude = np.abs(dft)
            phase = np.angle(dft)

            peak = np.unravel_index(np.argmax(magnitude), block.shape)
            magnitude[peak] *= (1 + alpha * watermark[index])

            dft_mod = magnitude * np.exp(1j * phase)
            watermarked_block = np.real(np.fft.ifft2(dft_mod))

            watermarked_image[i:i+block_size, j:j+block_size] = watermarked_block
            index += 1

    return watermarked_image

def extract(original, watermarked, alpha=0.1, block_size=32):
    original = original.astype(np.float64)
    watermarked = watermarked.astype(np.float64)
    length, width = original.shape

    extracted = []

    for i in range(0, length, block_size):
        for j in range(0, width, block_size):
            block_orig = original[i:i+block_size, j:j+block_size]
            block_water = watermarked[i:i+block_size, j:j+block_size]

            if block_orig.shape != (block_size, block_size):
                continue

            dft_orig = np.fft.fft2(block_orig)
            dft_water = np.fft.fft2(block_water)

            mag_orig = np.abs(dft_orig)
            mag_water = np.abs(dft_water)

            peak = np.unravel_index(np.argmax(mag_orig), block_orig.shape)
            v = mag_orig[peak]
            v_star = mag_water[peak]

            x_i = ((v_star / v) - 1) / alpha
            extracted.append(x_i)
    
    return np.array(extracted)
