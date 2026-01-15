import numpy as np
import os
from dinov2_numpy import Dinov2Numpy
from preprocess_image import resize_short_side
import random

class ImageSearcher:
    def __init__(self, model_path="vit-dinov2-base.npz", index_path="./index/features.npz"):
        # 首先设置图片目录
        self.images_dir = "images"
        if not os.path.exists(self.images_dir):
            os.makedirs(self.images_dir)
            print(f"创建图片目录: {self.images_dir}")
        
        # 获取所有存在的图片文件
        self.existing_images = self._get_existing_images()
        print(f"实际存在的图片文件: {len(self.existing_images)} 个")
        
        # 加载模型
        weights = np.load(model_path)
        self.vit = Dinov2Numpy(weights)
        
        # 加载索引库
        if os.path.exists(index_path):
            data = np.load(index_path, allow_pickle=True)
            self.gallery_features = data['features']
            self.metadata = data['metadata']
            self.gallery_labels = data['labels']  
            print(f"成功加载索引库，共 {len(self.gallery_features)} 条记录。")
            print(f"标签范围: {self.gallery_labels.min()} 到 {self.gallery_labels.max()}")
        else:
            raise FileNotFoundError(f"索引文件 {index_path} 不存在")

    def _get_existing_images(self):
        """获取所有实际存在的图片文件"""
        existing_images = []
        if os.path.exists(self.images_dir):
            for f in os.listdir(self.images_dir):
                if f.lower().endswith('.jpg'):
                    existing_images.append(f)
        
        # 按文件名排序
        existing_images.sort()
        return existing_images

    def get_random_images(self, count=50):
        """获取随机图片"""
        if not self.existing_images:
            return []
        
        # 随机选择图片
        count = min(count, len(self.existing_images))
        selected_images = random.sample(self.existing_images, count)
        
        # 返回文件名列表
        return selected_images

    def _get_image_filename(self, label):
        """根据标签获取图片文件名"""
        return f"{label:06d}.jpg"

    def search(self, query_image_path, top_k=10):
        """检索图片"""
        if self.gallery_features is None:
            return []

        try:
            # 1. 预处理并提取查询图特征
            pixel_values = resize_short_side(query_image_path)
            query_feat = self.vit(pixel_values)  # (1, 768)
            
            # 2. 特征归一化
            query_feat = query_feat[0]  # 去掉batch维度 -> (768,)
            query_feat = query_feat / (np.linalg.norm(query_feat) + 1e-9)
            
            # 3. 计算余弦相似度
            similarities = np.dot(self.gallery_features, query_feat).flatten()
            
            # 4. 获取前 K 个结果
            top_indices = np.argsort(similarities)[::-1][:top_k]
            
            results = []
            for feature_idx in top_indices:
                # 获取对应的标签
                label = int(self.gallery_labels[feature_idx])
                
                # 根据标签生成图片文件名
                image_id = self._get_image_filename(label)
                
                # 检查图片是否存在
                image_path = os.path.join(self.images_dir, image_id)
                image_exists = os.path.exists(image_path)
                
                # 获取元数据
                meta = {}
                if self.metadata is not None and feature_idx < len(self.metadata):
                    if isinstance(self.metadata[feature_idx], dict):
                        meta = self.metadata[feature_idx]
                    elif self.metadata[feature_idx] is not None:
                        meta = {"data": str(self.metadata[feature_idx])}
                
                # 计算相似度百分比
                score = float(similarities[feature_idx])
                similarity_percent = min(max(score * 100, 0), 100)
                
                results.append({
                    "image_id": image_id,          
                    "feature_index": feature_idx,  
                    "label": label,                
                    "score": score,
                    "similarity_percent": similarity_percent,
                    "exists": image_exists,
                    "metadata": meta,
                    "rank": len(results) + 1
                })
            

            print(f"\n=== 搜索结果统计 ===")
            found_count = len([r for r in results if r['exists']])
            print(f"总结果数: {len(results)}")
            print(f"找到图片文件: {found_count}")
            print(f"缺失图片文件: {len(results) - found_count}")
            
            print("前5个结果详情:")
            for i, r in enumerate(results[:5]):
                status = "✓" if r['exists'] else "✗"
                print(f"  {i+1}. [{status}] 文件:{r['image_id']} 标签:{r['label']} 相似度:{r['similarity_percent']:.2f}%")
            
            return results
            
        except Exception as e:
            print(f"搜索过程中出错: {e}")
            import traceback
            traceback.print_exc()
            return []