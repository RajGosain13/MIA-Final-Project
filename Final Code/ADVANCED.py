import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import watermark as w
import attacks as a
from skimage.transform import resize

# Open and resize
image = np.array(Image.open('RajAndSofiaTest.jpeg').convert('L'))
image = resize(image, (320, 320), anti_aliasing=True)
image *= 255
length, width = image.shape

# Watermark parameters
block_size = 32
alpha = 0.25
user_id = 'Raj Gosain'
num_blocks = (length // block_size) * (width // block_size)

# Apply watermark
WMimage, WM, used_coords, _, _ = w.embed_advanced(image, user_id, alpha, block_size)
Image.fromarray(np.clip(WMimage, 0, 255).astype(np.uint8)).save('RajAndSofiaWatermarked.jpeg')

# Print OG and watermarked image
fig, ax = plt.subplots(1,2)
ax[0].imshow(image,cmap='gray')
ax[0].set_title('Original')
ax[1].imshow(WMimage,cmap='gray')
ax[1].set_title(f'{chr(945)} = {alpha} Watermarked\nSNR:{w.snr(image,WMimage)}')
plt.show()

# Apply attacks
attacked = {
    'Gaussian': a.gaussian_noise(WMimage),
    'Salt and Pepper': a.salt_and_pepper(WMimage),
    'Compression': a.compression(WMimage),
    'Blurred': a.blur(WMimage),
    'Histogram Equalization': a.histogram_equalization(WMimage),
    'Rotation': a.rotate(WMimage)
}

# Extract and plot
fig, ax = plt.subplots(2,3)
ax_it = 0
for name, attacked_image in attacked.items():
    watermark_extracted = w.extract_advanced(image, attacked_image, used_coords, alpha, block_size)
    similarity = w.similarity(WM, watermark_extracted)

    print(f"[{name}] Similarity: {similarity:.4f}") 
    
    ax[ax_it//3,ax_it%3].imshow(attacked_image, cmap='gray')
    ax[ax_it//3,ax_it%3].set_title(f"{name}\nSimilarity: {similarity:.4f}")
    ax[ax_it//3,ax_it%3].axis('off')
    ax_it += 1

plt.show()
    


     

