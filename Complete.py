import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import watermark as w
import attacks as a

image = np.array(Image.open('RajAndSofiaTest.jpeg').convert('L'))
length, width = image.shape
block_size = 32
alpha = 0.05

user_id = 'Raj Gosain'
num_blocks = (length // block_size) * (width // block_size)

watermark_normal = w.generate(user_id, num_blocks)
watermarked_normal = w.embed(image, watermark_normal, alpha, block_size)
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

watermarked_robust, watermark_robust, user_coords = w.embed_robust(image, user_id, alpha, block_size)
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
    watermark_extracted = w.extract_robust(image, attacked_image, user_coords)
    similarity = w.similarity(watermark_robust, watermark_extracted)
    snr = w.snr(image, attacked_image)

    print(f"[{name}] Similarity: {similarity:.4f}, SNR: {snr:.2f} dB") 
    plt.imshow(attacked_image, cmap='gray')
    plt.title(f"{name}\nSimilarity: {similarity:.4f}, SNR: {snr:.2f} dB")
    plt.axis('off')
    plt.show()


watermarked_advanced, watermark_advanced, coords, reps = w.embed_advanced(image, user_id=user_id, alpha=alpha, reps=10)
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
    watermark_extracted = w.extract_advanced(image, attacked_image, coords, alpha=0.2, reps=10)
    similarity = w.similarity(watermark_advanced, watermark_extracted)
    snr = w.snr(image, attacked_image)

    print(f"[{name}] Similarity: {similarity:.4f}, SNR: {snr:.2f} dB") 
    plt.imshow(attacked_image, cmap='gray')
    plt.title(f"{name}\nSimilarity: {similarity:.4f}, SNR: {snr:.2f} dB")
    plt.axis('off')
    plt.show()

