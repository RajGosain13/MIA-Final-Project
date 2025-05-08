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

def similarity(x, x_star):
    x = np.sign(x)
    x_star = np.sign(x_star)
    return np.dot(x, x_star) / len(x)

def snr(original, watermarked):
    noise = original - watermarked
    return 10 * np.log10(np.sum(np.power(original, 2) / np.sum(np.power(noise, 2))))


def mid_frequency_coords(block_size, margin=5):
    center = block_size // 2
    coords = []
    for u in range(block_size):
        for v in range(block_size):
            dist = np.sqrt((u - center)**2 + (v - center)**2)
            if margin < dist < center:
                coords.append((u, v))
    return coords

def embed_robust(image, user_id, alpha=0.1, block_size=32, eps=1e-3):
    image = image.astype(np.float64)
    watermarked_image = np.copy(image)
    length, width = image.shape

    used_coords = []
    mid_coords = mid_frequency_coords(block_size)

    for i in range(0, length, block_size):
        for j in range(0, width, block_size):
            block = image[i:i+block_size, j:j+block_size]
            if block.shape != (block_size, block_size):
                continue

            dft = np.fft.fft2(block)
            magnitude = np.abs(dft)
            phase = np.angle(dft)

            for u, v in mid_coords:
                if magnitude[u, v] > eps:
                    used_coords.append((i, j, u, v))
                    break

    watermark = generate(user_id, len(used_coords))

    for index, (i, j, u, v) in enumerate(used_coords):
        block = image[i:i+block_size, j:j+block_size]
        dft = np.fft.fft2(block)
        magnitude = np.abs(dft)
        phase = np.angle(dft)

        magnitude[u, v] *= (1 + alpha * watermark[index])
        dft_mod = magnitude * np.exp(1j * phase)
        watermarked_block = np.real(np.fft.ifft2(dft_mod))

        watermarked_image[i:i+block_size, j:j+block_size] = watermarked_block

    return watermarked_image, watermark, used_coords

def extract_robust(original, watermarked, used_coords, alpha=0.1, block_size=32, eps=1e-3):
    original = original.astype(np.float64)
    watermarked = watermarked.astype(np.float64)

    extracted = []

    for i, j, u, v in used_coords:
        block_orig = original[i:i+block_size, j:j+block_size]
        block_water = watermarked[i:i+block_size, j:j+block_size]

        dft_orig = np.fft.fft2(block_orig)
        dft_water = np.fft.fft2(block_water)

        v_orig = np.abs(dft_orig[u, v])
        v_water = np.abs(dft_water[u, v])

        if v_orig < eps:
            extracted.append(0.0)
        else:
            x_i = ((v_water / v_orig) - 1) / alpha
            extracted.append(x_i)

    return np.array(extracted)

def embed_advanced(image, user_id, alpha=0.1, block_size=32, reps=3, eps=1e-3):
    image = image.astype(np.float64)
    watermarked = np.copy(image)
    length, width = image.shape

    mid_coords = mid_frequency_coords(block_size)
    all_blocks = []

    for i in range(0, length, block_size):
        for j in range(0, width, block_size):
            block = image[i:i+block_size, j:j+block_size]
            if block.shape != (block_size, block_size):
                continue
            for u, v in mid_coords:
                if np.abs(np.fft.fft2(block)[u, v]) > eps:
                    all_blocks.append((i, j, u, v))
                    break

    bits = len(all_blocks) // reps
    watermark = generate(user_id, bits)
    used_coords = []

    for bit_index in range(bits):
        for k in range(reps):
            block_index = bit_index * reps + k
            i, j, u, v = all_blocks[block_index]
            block = image[i:i+block_size, j:j+block_size]
            dft = np.fft.fft2(block)

            mag = np.abs(dft)
            phase = np.angle(dft)
            mag_norm = mag / (np.linalg.norm(mag) + 1e-10)
            mag_log = np.log1p(mag_norm)

            mag_log[u, v] *= (1 + alpha * watermark[bit_index])
            mag_mod = np.expm1(mag_log)
            mag_mod *= np.linalg.norm(mag) / (np.linalg.norm(mag_mod) + 1e-10)

            dft_mod = mag_mod * np.exp(1j * phase)
            watermarked_block = np.real(np.fft.ifft2(dft_mod))
            watermarked[i:i+block_size, j:j+block_size] = watermarked_block

            used_coords.append((i, j, u, v, bit_index))

    return watermarked, watermark, used_coords, reps

def extract_advanced(original, watermarked, used_coords, alpha=0.1, block_size=32, reps=3):
    original = original.astype(np.float64)
    watermarked = watermarked.astype(np.float64)

    extracted = {}
    for i, j, u, v, bit_index in used_coords:
        block_o = original[i:i+block_size, j:j+block_size]
        block_w = watermarked[i:i+block_size, j:j+block_size]

        dft_o = np.fft.fft2(block_o)
        dft_w = np.fft.fft2(block_w)

        mag_o = np.abs(dft_o)
        mag_w = np.abs(dft_w)

        norm_o = mag_o / (np.linalg.norm(mag_o) + 1e-10)
        norm_w = mag_w / (np.linalg.norm(mag_w) + 1e-10)

        log_o = np.log1p(norm_o)
        log_w = np.log1p(norm_w)

        v_orig = log_o[u, v]
        v_water = log_w[u, v]

        if v_orig < 1e-8:
            x_i = 0.0
        else:
            x_i = ((v_water / v_orig) - 1) / alpha

        extracted.setdefault(bit_index, []).append(x_i)

    # Majority vote for each bit
    result = []
    for idx in sorted(extracted.keys()):
        votes = np.sign(extracted[idx])
        final = np.sign(np.sum(votes))
        result.append(final)

    return np.array(result)
