import numpy as np
from PIL import Image

def center_crop(img_path, crop_size=224):
    # Step 1: load image
    image = Image.open(img_path).convert("RGB")

    # Step 2: center crop
    w, h = image.size
    left = (w - crop_size) // 2
    top = (h - crop_size) // 2
    right = left + crop_size
    bottom = top + crop_size
    image = image.crop((left, top, right, bottom))  # PIL Image, size (224, 224)

    # Step 3: to_numpy
    image = np.array(image).astype(np.float32) / 255.0  # (H, W, C)

    # Step 4: norm
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])
    image = (image - mean) / std  # (H, W, C)
    image = image.transpose(2, 0, 1) # (C, H, W)
    return image[None] # (1, C, H, W)


def resize_short_side(img_path, target_size=224,patch_size=14):
    # Step 1: load image
    image = Image.open(img_path).convert("RGB")
    w,h=image.size

    # Step 2: resize so that the shorter side == target_size
    # and more, ensure both sides are multiples of patch size, e.g., 14
    scale=target_size/min(w,h)
    new_w=w*scale
    new_h=h*scale
    
    #强制调整为 patch_size (14) 的整数倍
    new_w=int(new_w/patch_size)*patch_size
    new_h=int(new_h/patch_size)*patch_size
    
    # 避免缩放后尺寸为0
    new_w = max(new_w, patch_size)
    new_h = max(new_h, patch_size)
    
    image = image.resize((new_w, new_h), resample=Image.BICUBIC)

    # Step 3: to_numpy
    image = np.array(image).astype(np.float32) / 255.0  # (H, W, C)

    # Step 4: norm
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])
    image = (image - mean) / std  # (H, W, C)
    image = image.transpose(2, 0, 1) # (C, H, W)
    return image[None] # (1, C, H, W)