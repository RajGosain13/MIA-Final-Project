import cv2
import numpy as np
import hashlib


def text_to_seed(text):
    return int(hashlib.sha256(text.encode()).hexdigest(), 16) % (2**32)

def watermark(input, watermark, output, points=100, strength=10):
    image = cv2.imread(input, cv2.IMREAD_GRAYSCALE)

    rows, cols = image.shape
    dft = np.fft.fft2(image)
    shifted = np.fft.fftshift(dft)

    seed = text_to_seed(watermark)
    rng = np.random.default_rng(seed)

    for _ in range(points):
        r = rng.integers(0, rows // 2)
        c = rng.integers(0, cols // 2)
        value = rng.choice([-1, 1]) * strength

        shifted[r, c] += value
        shifted[rows - r - 1, cols - c - 1] += value


    dft_ishift = np.fft.ifftshift(shifted)
    img_back = np.fft.ifft2(dft_ishift)
    img_back = np.abs(img_back)
    img_back = np.clip(img_back, 0, 255).astype(np.uint8)

    cv2.imwrite(output, img_back)

    combined = np.hstack((image, img_back))
    cv2.imshow("Original (left) vs Watermarked (right)", combined)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def show_watermark(image):
    img = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
    dft = np.fft.fft2(img)
    dft_shifted = np.fft.fftshift(dft)
    magnitude_spectrum = 20 * np.log(np.abs(dft_shifted) + 1)

    mag_norm = cv2.normalize(magnitude_spectrum, None, 0, 255, cv2.NORM_MINMAX)
    mag_norm = mag_norm.astype(np.uint8)

    cv2.imshow("DFT Magnitude Spectrum", magnitude_spectrum.astype(np.uint8))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def show_watermark_locations(image_path, watermark_text, points=100):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    dft = np.fft.fft2(img)
    shifted = np.fft.fftshift(dft)
    magnitude = 20 * np.log(np.abs(shifted) + 1)
    mag_norm = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    mag_color = cv2.cvtColor(mag_norm, cv2.COLOR_GRAY2BGR)

    # Recreate watermark coordinates
    rows, cols = img.shape
    seed = text_to_seed(watermark_text)
    rng = np.random.default_rng(seed)

    for _ in range(points):
        r = rng.integers(0, rows // 2)
        c = rng.integers(0, cols // 2)

        # Draw red dots at watermark locations
        mag_color = cv2.circle(mag_color, (c, r), radius=3, color=(0, 0, 255), thickness=-1)
        mag_color = cv2.circle(mag_color, (cols - c - 1, rows - r - 1), radius=3, color=(0, 0, 255), thickness=-1)

    cv2.imshow("Watermark Locations in DFT", mag_color)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Comment 1
watermark('RajAndSofia.jpeg', "Sofia", "RajAndSofiaWatermarked.jpeg")
show_watermark('RajAndSofiaWatermarked.jpeg')
show_watermark_locations('RajAndSofiaWatermarked.jpeg', 'Sofia')