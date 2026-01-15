import os
import requests
import pandas as pd
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image
import time

from dinov2_numpy import Dinov2Numpy
from preprocess_image import resize_short_side

# --- 配置区 ---
CSV_PATH = "data.csv"
IMAGES_DIR = "./images" 
INDEX_DIR = "./index"
MAX_WORKERS = 8     
TARGET_COUNT = 1000000
BATCH_SIZE = 100

# 创建目录
os.makedirs(IMAGES_DIR, exist_ok=True)
os.makedirs(INDEX_DIR, exist_ok=True)

# 检查点文件路径
CHECKPOINT_FILE = os.path.join(INDEX_DIR, "checkpoint.txt")
FEATURES_FILE = os.path.join(INDEX_DIR, "features.npz")  

def load_checkpoint():
    """加载检查点，返回已处理的数量"""
    if os.path.exists(CHECKPOINT_FILE):
        try:
            with open(CHECKPOINT_FILE, 'r') as f:
                return int(f.read().strip())
        except:
            return 0
    return 0

def save_checkpoint(count):
    """保存检查点"""
    with open(CHECKPOINT_FILE, 'w') as f:
        f.write(str(count))

def download_image(url, save_path):
    """下载图片并保存"""
    try:
        response = requests.get(url, timeout=(5, 10), headers={'User-Agent': 'Mozilla/5.0'})
        if response.status_code == 200:
            with open(save_path, 'wb') as f:
                f.write(response.content)
            return True
    except Exception as e:
        return False
    return False

def extract_and_save_features(vit, images_data, features_file, metadata_list):
    """提取特征并保存到npz文件"""
    if not images_data:
        return
    
    all_features = []
    all_metadata = []
    all_labels = []  # 存储标签,便于通过特征找到相应图片
    
    for img_path, meta in tqdm(images_data, desc="提取特征中", leave=False):
        try:
            # 预处理图片
            pixel_values = resize_short_side(img_path)
            # 提取特征
            feat = vit(pixel_values)
            # 归一化
            feat = feat / (np.linalg.norm(feat, axis=-1, keepdims=True) + 1e-9)
            
            all_features.append(feat)
            all_metadata.append(meta)
            
            # 从图片路径提取标签（索引）
            filename = os.path.basename(img_path)
            # 去掉扩展名，提取数字部分
            label = int(filename.split('.')[0])
            all_labels.append(label)
            
        except Exception as e:
            # 提取失败，删除图片
            try:
                os.remove(img_path)
            except:
                pass
            continue
    
    if all_features:
        # 转换为numpy数组
        features_array = np.vstack(all_features).astype(np.float32)
        metadata_array = np.array(all_metadata, dtype=object)
        labels_array = np.array(all_labels, dtype=np.int32)  
        
        # 保存到npz文件
        if os.path.exists(features_file):
            # 加载现有数据
            existing_data = np.load(features_file, allow_pickle=True)
            existing_features = existing_data['features']
            existing_metadata = existing_data['metadata']
            existing_labels = existing_data['labels'] 
            
            # 合并数据
            features_array = np.vstack([existing_features, features_array])
            metadata_array = np.concatenate([existing_metadata, metadata_array])
            labels_array = np.concatenate([existing_labels, labels_array])  
        
        # 保存npz文件
        np.savez_compressed(
            features_file,
            features=features_array,
            metadata=metadata_array,
            labels=labels_array  
        )
        
        print(f"已保存 {len(all_features)} 个特征到 {features_file}")
    
    return len(all_features)

def build_gallery():
    """主函数：构建图片库"""
    # 加载数据
    df = pd.read_csv(CSV_PATH).head(TARGET_COUNT)
    
    # 加载模型权重
    weights = np.load("vit-dinov2-base.npz")
    vit = Dinov2Numpy(weights)
    
    # 加载检查点
    processed_count = load_checkpoint()
    print(f"从检查点恢复，已处理: {processed_count} 张")
    
    # 准备任务列表（跳过已处理的）
    tasks = []
    for idx in range(processed_count, min(len(df), TARGET_COUNT)):
        row = df.iloc[idx]
        save_path = os.path.join(IMAGES_DIR, f"{idx:06d}.jpg")
        tasks.append((idx, row['image_url'], save_path, row.to_dict()))
    
    if not tasks:
        print("所有图片已处理完成！")
        return
    
    print(f"开始处理 {len(tasks)} 张图片...")
    
    start_time = time.time()
    batch_images_data = []  # 存储批量图片数据用于特征提取
    successful_downloads = 0
    
    with tqdm(total=len(tasks), desc="下载图片") as pbar:
        for i in range(0, len(tasks), MAX_WORKERS):
            # 准备当前批次的任务
            batch_tasks = tasks[i:i + MAX_WORKERS]
            
            # 使用线程池下载当前批次的图片
            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                future_to_task = {}
                for task in batch_tasks:
                    idx, url, path, meta = task
                    future = executor.submit(download_image, url, path)
                    future_to_task[future] = (idx, path, meta)
                
                # 处理下载结果
                for future in as_completed(future_to_task):
                    idx, path, meta = future_to_task[future]
                    try:
                        if future.result():
                            # 下载成功，添加到批量处理列表
                            batch_images_data.append((path, meta))
                            successful_downloads += 1
                    except Exception as e:
                        # 下载失败，删除可能存在的损坏文件
                        if os.path.exists(path):
                            try:
                                os.remove(path)
                            except:
                                pass
                    
                    pbar.update(1)
                    pbar.set_postfix({
                        "已下载": successful_downloads,
                        "成功率": f"{(successful_downloads/(i+pbar.n+1))*100:.1f}%"
                    })
            
            # 每下载完一批，就提取特征
            if batch_images_data:
                # 提取特征并保存
                extracted_count = extract_and_save_features(
                    vit, batch_images_data, FEATURES_FILE, []
                )
                
                # 更新处理计数
                processed_count += len(batch_tasks)
                save_checkpoint(processed_count)
                
                # 清空当前批次数据
                batch_images_data = []
                
                # 显示统计信息
                elapsed_time = time.time() - start_time
                avg_time = elapsed_time / successful_downloads if successful_downloads > 0 else 0
                print(f"\n进度: {processed_count}/{min(len(df), TARGET_COUNT)} | "
                      f"成功: {successful_downloads} | "
                      f"平均耗时: {avg_time:.2f}s/张 | "
                      f"总耗时: {elapsed_time:.1f}s")
    
    # 处理最后一批数据
    if batch_images_data:
        extract_and_save_features(vit, batch_images_data, FEATURES_FILE, [])
        processed_count += len(batch_images_data)
        save_checkpoint(processed_count)
    
    # 最终统计
    print("\n" + "="*50)
    print("处理完成！")
    print(f"总图片数: {min(len(df), TARGET_COUNT)}")
    print(f"成功下载并提取特征: {successful_downloads}")
    
    # 加载并显示最终特征文件信息
    if os.path.exists(FEATURES_FILE):
        data = np.load(FEATURES_FILE, allow_pickle=True)
        print(f"特征文件: {FEATURES_FILE}")
        print(f"特征维度: {data['features'].shape}")
        print(f"标签数量: {len(data['labels'])}")  
        print(f"图片保存在: {IMAGES_DIR}")

if __name__ == "__main__":
    build_gallery()