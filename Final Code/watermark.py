import numpy as np

def generate(user, length):
    seed = sum(map(ord, user))
    rng = np.random.default_rng(seed)
    return rng.normal(0, 1, length)

def mid_frequency_coords(block_size, margin=5):
    center = block_size // 2
    coords = [
        (u, v) for u in range(block_size) for v in range(block_size)
        if margin < np.hypot(u - center, v - center) < center
    ]
    return coords

def embed(image, watermark, alpha=0.1, block_size=32):
    image = image.astype(np.float64)
    output = image.copy()
    idx = 0
    used_coords = []

    for i in range(0, image.shape[0], block_size):
        for j in range(0, image.shape[1], block_size):
            if idx >= len(watermark):
                return output, used_coords

            block = image[i:i+block_size, j:j+block_size]
            if block.shape != (block_size, block_size):
                continue

            dft = np.fft.fft2(block)
            mag, phase = np.abs(dft), np.angle(dft)
            peak = np.unravel_index(np.argmax(mag), mag.shape)
            mag[peak] *= 1 + alpha * watermark[idx]

            mod_block = np.real(np.fft.ifft2(mag * np.exp(1j * phase)))
            output[i:i+block_size, j:j+block_size] = mod_block
            used_coords.append((i, j))
            idx += 1

    return output, used_coords

def extract(original, watermarked, alpha=0.1, block_size=32):
    original = original.astype(np.float64)
    watermarked = watermarked.astype(np.float64)
    result = []

    for i in range(0, original.shape[0], block_size):
        for j in range(0, original.shape[1], block_size):
            orig_block = original[i:i+block_size, j:j+block_size]
            wm_block = watermarked[i:i+block_size, j:j+block_size]
            if orig_block.shape != (block_size, block_size):
                continue

            dft_o = np.fft.fft2(orig_block)
            dft_w = np.fft.fft2(wm_block)

            peak = np.unravel_index(np.argmax(np.abs(dft_o)), orig_block.shape)
            v_o = np.abs(dft_o[peak])
            v_w = np.abs(dft_w[peak])
            result.append(((v_w / v_o) - 1) / alpha)

    return np.array(result)

def similarity(x, x_star):
    numerator = np.abs(np.dot(x, x_star))
    denominator = np.sqrt(np.dot(x_star, x_star))
    return numerator/denominator

def snr(original, watermarked):
    noise = original - watermarked
    return 10 * np.log10(np.sum(original**2) / np.sum(noise**2))

def embed_robust(image, user_id, alpha=0.1, block_size=32, eps=1e-8):
    image = image.astype(np.float64)
    output = image.copy()
    coords = mid_frequency_coords(block_size)
    used = []

    for i in range(0, image.shape[0], block_size):
        for j in range(0, image.shape[1], block_size):
            block = image[i:i+block_size, j:j+block_size]
            if block.shape != (block_size, block_size): continue
            dft = np.fft.fft2(block)
            for u, v in coords:
                if np.abs(dft[u, v]) > eps:
                    used.append((i, j, u, v))
                    break

    watermark = generate(user_id, len(used))
    for idx, (i, j, u, v) in enumerate(used):
        block = image[i:i+block_size, j:j+block_size]
        dft = np.fft.fft2(block)
        mag, phase = np.abs(dft), np.angle(dft)
        mag[u, v] *= 1 + alpha * watermark[idx]
        output[i:i+block_size, j:j+block_size] = np.real(np.fft.ifft2(mag * np.exp(1j * phase)))
    block_coords = list({(i, j) for i, j, _, _ in used})
    return output, watermark, used, block_coords

def extract_robust(original, watermarked, used, alpha=0.1, block_size=32, eps=1e-8):
    original = original.astype(np.float64)
    watermarked = watermarked.astype(np.float64)
    result = []

    for i, j, u, v in used:
        bo = original[i:i+block_size, j:j+block_size]
        bw = watermarked[i:i+block_size, j:j+block_size]
        dft_o = np.fft.fft2(bo)
        dft_w = np.fft.fft2(bw)

        vo, vw = np.abs(dft_o[u, v]), np.abs(dft_w[u, v])
        result.append(((vw / vo - 1) / alpha) if vo > eps else 0.0)

    return np.array(result)


def embed_advanced(image, user_id, alpha=0.1, block_size=32, reps=3, eps=1e-8):
    image = image.astype(np.float64)
    output = image.copy()
    coords = mid_frequency_coords(block_size)
    usable = []

    for i in range(0, image.shape[0], block_size):
        for j in range(0, image.shape[1], block_size):
            block = image[i:i+block_size, j:j+block_size]
            if block.shape != (block_size, block_size): continue
            dft = np.fft.fft2(block)
            for u, v in coords:
                if np.abs(dft[u, v]) > eps:
                    usable.append((i, j, u, v))
                    break

    bits = len(usable) // reps
    watermark = generate(user_id, bits)
    used = []

    for b in range(bits):
        for r in range(reps):
            idx = b * reps + r
            i, j, u, v = usable[idx]
            block = image[i:i+block_size, j:j+block_size]
            dft = np.fft.fft2(block)
            mag = np.abs(dft)
            phase = np.angle(dft)

            mag /= np.linalg.norm(mag) + 1e-10
            mag_log = np.log1p(mag)
            mag_log[u, v] *= 1 + alpha * watermark[b]
            mod = np.expm1(mag_log)
            mod *= np.linalg.norm(np.abs(dft)) #/ (np.linalg.norm(mod) + 1e-10)

            output[i:i+block_size, j:j+block_size] = np.real(np.fft.ifft2(mod * np.exp(1j * phase)))
            used.append((i, j, u, v, b))
    block_coords = list({(i, j) for i, j, _, _, _ in used})
    return output, watermark, used, reps, block_coords

def extract_advanced(original, watermarked, used, alpha=0.1, block_size=32):
    extracted = {}
    for i, j, u, v, b in used:
        bo = original[i:i+block_size, j:j+block_size]
        bw = watermarked[i:i+block_size, j:j+block_size]
        dft_o = np.fft.fft2(bo)
        dft_w = np.fft.fft2(bw)

        mag_o, mag_w = np.abs(dft_o), np.abs(dft_w)
        log_o = np.log1p(mag_o / (np.linalg.norm(mag_o) + 1e-10))
        log_w = np.log1p(mag_w / (np.linalg.norm(mag_w) + 1e-10))

        x = ((log_w[u, v] / log_o[u, v]) - 1) / alpha if log_o[u, v] > 1e-8 else 0.0
        extracted.setdefault(b, []).append(x)

    return np.array([np.sign(np.sum(np.sign(extracted[b]))) for b in sorted(extracted)])

def similarity_at_coords(original, modified, coords, block_size=32):

    numerator = np.sum(np.multiply(original, modified))
    denominator = np.sqrt(np.sum(np.multiply(modified,modified)))
    return numerator/denominator



    '''
    dots = 0
    total = 0

    for i, j in coords:
        orig_block = original[i:i+block_size, j:j+block_size]
        mod_block = modified[i:i+block_size, j:j+block_size]
        
        # Optional: flatten blocks for dot product
        o = np.sign(orig_block.flatten())
        m = np.sign(mod_block.flatten())

        dots += np.dot(o, m)
        total += len(o)

    return dots / total if total > 0 else 0.0

    '''

