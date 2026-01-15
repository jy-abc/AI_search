import os
from flask import Flask, render_template, request, send_from_directory, url_for, jsonify
from werkzeug.utils import secure_filename
from image_retrieval import ImageSearcher
import time
import random

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['IMAGES_FOLDER'] = 'images'

# 确保文件夹存在
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# 初始化搜索器
searcher = ImageSearcher()

@app.route('/')
def index():
    """首页 - 展示图库"""
    # 获取随机50张图片用于首页显示
    gallery_images = get_random_images(50)
    return render_template('index.html', gallery_images=gallery_images)

@app.route('/search', methods=['GET', 'POST'])
def search_page():
    """检索页面"""
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('search.html', error="未选择文件")
        
        file = request.files['file']
        if file.filename == '':
            return render_template('search.html', error="文件名为空")

        if file:
            try:
                # 生成唯一文件名
                timestamp = int(time.time())
                filename = f"{timestamp}_{secure_filename(file.filename)}"
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                
                print(f"\n=== 开始搜索 ===")
                print(f"查询图片: {filename}")
                
                # 执行检索
                results = searcher.search(filepath, top_k=10)
                
                # 获取上传图片的URL
                query_img_url = url_for('uploaded_file', filename=filename)
                
                return render_template('search.html', 
                                       query_img_url=query_img_url,
                                       results=results)
                
            except Exception as e:
                print(f"处理错误: {e}")
                return render_template('search.html', error=f"处理失败: {str(e)}")
    
    # GET请求，显示空白的检索页面
    return render_template('search.html')

@app.route('/gallery/<filename>')
def get_gallery_image(filename):
    """提供图片访问"""
    try:
        # 安全检查
        if '..' in filename or '/' in filename:
            return "无效文件名", 400
        
        # 构造完整路径
        filepath = os.path.join(app.config['IMAGES_FOLDER'], filename)
        
        # 检查文件是否存在
        if not os.path.exists(filepath):
            print(f"图片文件不存在: {filepath}")
            return "图片不存在", 404
        
        return send_from_directory(app.config['IMAGES_FOLDER'], filename)
        
    except Exception as e:
        print(f"提供图片错误: {e}")
        return "内部错误", 500

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/api/random_images')
def get_random_images_api():
    """获取随机图片的API接口"""
    count = request.args.get('count', 50, type=int)
    gallery_images = get_random_images(count)
    return jsonify({
        'success': True,
        'count': len(gallery_images),
        'images': gallery_images
    })

def get_random_images(count=50):
    """获取随机图片列表"""
    images = []
    
    if os.path.exists(app.config['IMAGES_FOLDER']):
        # 获取所有图片文件
        all_files = []
        for f in os.listdir(app.config['IMAGES_FOLDER']):
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.gif')):
                all_files.append(f)
        
        # 随机选择指定数量的图片
        if all_files:
            # 如果请求的数量大于实际数量，使用实际数量
            count = min(count, len(all_files))
            selected_files = random.sample(all_files, count)
            
            for filename in selected_files:
                images.append({
                    'filename': filename,
                    'url': url_for('get_gallery_image', filename=filename)
                })
    
    return images

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)