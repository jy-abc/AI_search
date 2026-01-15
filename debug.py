import numpy as np
import os
from dinov2_numpy import Dinov2Numpy
from preprocess_image import center_crop

# 加载模型
weights = np.load("vit-dinov2-base.npz")
vit = Dinov2Numpy(weights)

# 提取特征
print("Extracting features...")
cat_pixel_values = center_crop("./demo_data/cat.jpg")
cat_feat = vit(cat_pixel_values) # (1, 768)

dog_pixel_values = center_crop("./demo_data/dog.jpg")
dog_feat = vit(dog_pixel_values) # (1, 768)

#合并特征
my_features = np.concatenate([cat_feat, dog_feat], axis=0) # (2, 768)

#加载参考文件并对比
ref_path = "./demo_data/cat_dog_feature.npy"
if os.path.exists(ref_path):
    print(f"Loading reference from {ref_path}...")
    ref_features = np.load(ref_path)
    
    # 检查形状是否一致
    if my_features.shape != ref_features.shape:
        print(f"Shape mismatch! Mine: {my_features.shape}, Ref: {ref_features.shape}")
    else:
        # 计算最大绝对误差
        diff = np.abs(my_features - ref_features)
        max_diff = np.max(diff)
        mean_diff = np.mean(diff)
        
        print(f"Max difference: {max_diff}")
        print(f"Mean difference: {mean_diff}")
        
        # 判断结果是否合理
        if mean_diff < 0.005:
            print("Success! The difference is within tolerance.")
        else:
            print("Warning: The difference is too large. Check implementation.")
else:
    print(f"Reference file {ref_path} not found. Cannot verify correctness.")