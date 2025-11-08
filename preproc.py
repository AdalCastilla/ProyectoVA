import cv2

import numpy as np

def resize_keep_aspect(img, max_side=720):
    h, w = img.shape[:2]
    s = max_side / max(h, w)
    return cv2.resize(img, (int(w*s), int(h*s)), interpolation=cv2.INTER_AREA) if s < 1 else img

def gaussian_denoise(img, ksize=3, sigma=0):
    ksize= max(1, ksize | 1)
    return cv2.GaussianBlur(img, (ksize, ksize), sigma)

def median_denoise(img, ksize=3):
    ksize = max(1, ksize | 1)  # impar
    return cv2.medianBlur(img, ksize)

def bilateral_denoise(img, d=5, sigmaColor=50, sigmaSpace=5):
    return cv2.bilateralFilter(img, d, sigmaColor, sigmaSpace)

def clahe_ycrcb(img, clip=2.0, tile=8):
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb)
    clahe = cv2.createCLAHE(clipLimit=float(clip), tileGridSize=(tile, tile))
    y2 = clahe.apply(y)
    return cv2.cvtColor(cv2.merge([y2, cr, cb]), cv2.COLOR_YCrCb2BGR)

def clahe_lab(img, clip=2.0, tile=8):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=float(clip), tileGridSize=(tile, tile))
    l2 = clahe.apply(l)
    return cv2.cvtColor(cv2.merge([l2, a, b]), cv2.COLOR_LAB2BGR)

def gamma_correct(img, gamma=1.0):
    gamma = max(0.1, float(gamma))
    inv = 1.0 / gamma
    table = (np.linspace(0, 1, 256) ** inv * 255).astype(np.uint8)
    return cv2.LUT(img, table)

def gray_world_white_balance(img):
    b, g, r = cv2.split(img.astype(np.float32) + 1e-6)
    mean = (b.mean() + g.mean() + r.mean()) / 3.0
    b *= mean / b.mean(); g *= mean / g.mean(); r *= mean / r.mean()
    out = cv2.merge([b, g, r])
    return np.clip(out, 0, 255).astype(np.uint8)

def unsharp_mask(img, ksize=5, amount=1.0, threshold=0):
    blur = cv2.GaussianBlur(img, (ksize | 1, ksize | 1), 0)
    sharp = cv2.addWeighted(img, 1 + amount, blur, -amount, 0)
    if threshold <= 0:
        return sharp
    mask = (np.abs(sharp.astype(np.int16) - img.astype(np.int16)) < threshold)
    np.copyto(sharp, img, where=mask)
    return sharp