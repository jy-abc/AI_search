# AI Image Search Web Application



## Project Overview

This project builds a complete end-to-end image retrieval system. It utilizes the DINOv2 Vision Transformer model to extract image features, implements "search by image" functionality through feature vector similarity computation, and develops an interactive web interface.



## Installation and Running

1. Install Dependencies

```bash
cd AI_search
# Create virtual environment
python -m venv ll_env
# Activate virtual environment
ll_env\Scripts\activate     # Windows
source ll_env/bin/activate  # Linux/Mac
# Install dependencies
pip install -r requirements.txt
```

2.Run

```bash
python app.py
```



## Project Structure

```tex
AI_search/
├── app.py                    # Backend main file
├── image_retrieval.py        # Core image retrieval class
├── dinov2_numpy.py           # DINOv2 model implementation
├── preprocess_image.py       # Image cropping
├── demo_data/      		  # demo
├── build_index.py            # Offline database creation
├── debug.py                  # Test file
├── requirements.txt       	  # Configuration file
└── templates/                # HTML templates
```



## Feature Overview

- Gallery Preview
  - Homepage randomly displays 50 offline images
  - Click any image to view enlarged preview
  - Support random refresh via button

- Interactive Retrieval
  - Supports uploading JPG/PNG/GIF format images
  - Real-time preview of uploaded images
  - Returns top 10 most similar matching results
