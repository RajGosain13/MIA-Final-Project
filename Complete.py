import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import watermark as w
import attacks as a

image = np.array(Image.open('RajAndSofiaTest.jpeg').convert('L'))
length, width = image.shape
block_size = 32
alpha = 0.25

user_id = 'Raj Gosain'
num_blocks = (length // block_size) * (width // block_size)

watermarked, watermark, used_coords = w.embed_robust(image, user_id=user_id, alpha=0.2)
Image.fromarray(np.clip(watermarked, 0, 255).astype(np.uint8)).save('RajAndSofiaWatermarked.jpeg')

attacks = {
    'Gaussian': a.gaussian_noise(watermarked),
    'Compression': a.compression(watermarked),
    'Blurred': a.blur(watermarked),
    'Resize': a.resize(watermarked),
    'Contrast': a.change_contrast(watermarked)
}

for name, attacked_image in attacks.items():
    watermark_extracted = w.extract_robust(image, attacked_image, used_coords, alpha=0.2)
    similarity = w.similarity(watermark, watermark_extracted)
    snr = w.snr(image, attacked_image)

    print(f"[{name}] Similarity: {similarity:.4f}, SNR: {snr:.2f} dB") 
    plt.imshow(attacked_image, cmap='gray')
    plt.title(f"{name}\nSimilarity: {similarity:.4f}, SNR: {snr:.2f} dB")
    plt.axis('off')
    plt.show()