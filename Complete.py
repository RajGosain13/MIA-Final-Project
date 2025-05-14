import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import watermark as w
import attacks as a
from skimage.transform import resize

image = np.array(Image.open('RajAndSofiaTest.jpeg').convert('L'))
image = resize(image, (320, 320), anti_aliasing=True)
length, width = image.shape
block_size = 32
alpha = 0.25

user_id = 'Raj Gosain'
num_blocks = (length // block_size) * (width // block_size)

watermark_normal = w.generate(user_id, num_blocks)
watermarked_normal, coords_normal = w.embed(image, watermark_normal, alpha, block_size)
Image.fromarray(np.clip(watermarked_normal, 0, 255).astype(np.uint8)).save('RajAndSofiaWatermarked.jpeg')

attacks_normal = {
    'Gaussian': a.gaussian_noise(watermarked_normal),
    'Salt and Pepper': a.salt_and_pepper(watermarked_normal),
    'Compression': a.compression(watermarked_normal),
    'Blurred': a.blur(watermarked_normal),
    'Histogram Equalization': a.histogram_equalization(watermarked_normal),
    'Rotation': a.rotate(watermarked_normal)
}

for name, attacked_image in attacks_normal.items():
    watermark_extracted = w.extract(image, attacked_image, alpha, block_size)
    similarity = w.similarity(watermark_normal, watermark_extracted)
    snr = w.snr(image, attacked_image)

    print(f"[{name}] Similarity: {similarity:.4f}, SNR: {snr:.2f} dB") 
    
    plt.imshow(attacked_image, cmap='gray')
    plt.title(f"{name}\nSimilarity: {similarity:.4f}, SNR: {snr:.2f} dB")
    plt.axis('off')
    plt.show()
    

watermarked_robust, watermark_robust, coords_robust, used_robust = w.embed_robust(image, user_id, alpha, block_size)
Image.fromarray(np.clip(watermarked_robust, 0, 255).astype(np.uint8)).save('RajAndSofiaWatermarked.jpeg')

attacks_robust = {
    'Gaussian': a.gaussian_noise(watermarked_robust),
    'Salt and Pepper': a.salt_and_pepper(watermarked_robust),
    'Compression': a.compression(watermarked_robust),
    'Blurred': a.blur(watermarked_robust),
    'Histogram Equalization': a.histogram_equalization(watermarked_robust),
    'Rotation': a.rotate(watermarked_robust)
}

for name, attacked_image in attacks_robust.items():
    watermark_extracted = w.extract_robust(image, attacked_image, coords_robust)
    similarity = w.similarity(watermark_robust, watermark_extracted)
    snr = w.snr(image, attacked_image)

    print(f"[{name}] Similarity: {similarity:.4f}, SNR: {snr:.2f} dB") 
    
    plt.imshow(attacked_image, cmap='gray')
    plt.title(f"{name}\nSimilarity: {similarity:.4f}, SNR: {snr:.2f} dB")
    plt.axis('off')
    plt.show()
    


watermarked_advanced, watermark_advanced, coords_adv, reps, used_adv = w.embed_advanced(image, user_id=user_id, alpha=alpha, reps=10)
Image.fromarray(np.clip(watermarked_advanced, 0, 255).astype(np.uint8)).save('RajAndSofiaWatermarked.jpeg')

attacks_advanced = {
    'Gaussian': a.gaussian_noise(watermarked_advanced),
    'Salt and Pepper': a.salt_and_pepper(watermarked_advanced),
    'Compression': a.compression(watermarked_advanced),
    'Blurred': a.blur(watermarked_advanced),
    'Histogram Equalization': a.histogram_equalization(watermarked_advanced),
    'Rotation': a.rotate(watermarked_advanced)
}

for name, attacked_image in attacks_advanced.items():
    watermark_extracted = w.extract_advanced(image, attacked_image, coords_adv, alpha=0.2)
    similarity = w.similarity(watermark_advanced, watermark_extracted)
    snr = w.snr(image, attacked_image)

    print(f"[{name}] Similarity: {similarity:.4f}, SNR: {snr:.2f} dB")
    plt.imshow(attacked_image, cmap='gray')
    plt.title(f"{name}\nSimilarity: {similarity:.4f}, SNR: {snr:.2f} dB")
    plt.axis('off')
    plt.show()
     

def compute_dft_magnitude(image):
    dft = np.fft.fft2(image)
    dft_shifted = np.fft.fftshift(dft)
    magnitude_spectrum = np.log1p(np.abs(dft_shifted))
    return magnitude_spectrum

original = np.array(Image.open('RajAndSofiaTest.jpeg').convert('L'), dtype=np.float64)
original = resize(image, (320, 320), anti_aliasing=True)
dft_o = compute_dft_magnitude(original)

dft_n = compute_dft_magnitude(watermarked_normal)
dft_r = compute_dft_magnitude(watermarked_robust)
dft_a = compute_dft_magnitude(watermarked_advanced)

diff_n = np.abs(dft_o - dft_n)
diff_r = np.abs(dft_o - dft_r)
diff_a = np.abs(dft_o - dft_a)

fig, axes = plt.subplots(3, 3, figsize=(10,10))
plt.subplot(3, 3, 1)
plt.imshow(dft_o, cmap='gray')
plt.title('Original DFT')
plt.axis('off')

plt.subplot(3, 3, 2)
plt.imshow(dft_n, cmap='gray')
plt.title('High Peak DFT')
plt.axis('off')

plt.subplot(3, 3, 3)
plt.imshow(dft_r, cmap='gray')
plt.title('Mid Frequency DFT')
plt.axis('off')

plt.subplot(3, 3, 4)
plt.imshow(dft_a, cmap='gray')
plt.title('Refined Mid Frequency DFT')
plt.axis('off')

plt.subplot(3, 3, 5)
plt.imshow(diff_n, cmap='gray')
plt.title('Difference High Peak')
plt.axis('off')

plt.subplot(3, 3, 6)
plt.imshow(diff_r, cmap='gray')
plt.title('Difference Mid Frequency')
plt.axis('off')

plt.subplot(3, 3, 7)
plt.imshow(diff_a, cmap='gray')
plt.title('Difference Refined Mid Frequency')
plt.axis('off')

plt.tight_layout()
plt.show()

def plot_fft_histogram(image, title="FFT Magnitude Histogram", bins=100):
    dft = np.fft.fft2(image)
    dft_shifted = np.fft.fftshift(dft)
    magnitude = np.abs(dft_shifted).flatten()

    magnitude = magnitude[magnitude > 0] 
    magnitude = np.log1p(magnitude)

    plt.figure(figsize=(8, 4))
    plt.hist(magnitude, bins=bins, color='steelblue', edgecolor='black')
    plt.title(title)
    plt.xlabel('Log Magnitude')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

'''
plot_fft_histogram(image, title="Original Image FFT Histogram")
plot_fft_histogram(watermarked_normal, title="High Peak Watermarked FFT Histogram")
plot_fft_histogram(watermarked_robust, title="Mid Frequency Watermarked FFT Histogram")
plot_fft_histogram(watermarked_advanced, title="Refined Mid Frequency Watermarked FFT Histogram")
'''

similarity_normal = w.similarity_at_coords(image, watermarked_normal, coords_normal, block_size)
print(similarity_normal)
plt.imshow(watermarked_normal, cmap='gray')
plt.title(f"High Peak Watermark\nSimilarity: {similarity_normal:.4f} dB")
plt.axis('off')
plt.show()


similarity_robust = w.similarity_at_coords(image, watermarked_robust, used_robust)
plt.imshow(watermarked_robust, cmap='gray')
plt.title(f"Mid Frequency Watermark\nSimilarity: {similarity_robust:.4f} dB")
plt.axis('off')
plt.show()

similarity_adv = w.similarity_at_coords(image, watermarked_advanced, used_adv)
plt.imshow(watermarked_advanced, cmap='gray')
plt.title(f"Refined Mid Frequency Watermark\nSimilarity: {similarity_adv:.7f} dB")
plt.axis('off')
plt.show()