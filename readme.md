
# Janus AI Image Generator

![Janus AI](static/janus-logo.png)

## Overview

Janus AI Image Generator is a web application that leverages DeepSeek AI's powerful Janus models to generate high-quality images from text descriptions. This application supports multiple resolutions (1K, 2K, and 4K), batch generation, and various creative settings, all within an intuitive user interface.

The application uses a Python Flask backend to handle image generation and a modern HTML/CSS/JS frontend for the user interface.

## Features

- **Text-to-Image Generation**: Create images from detailed text descriptions
- **Multiple Resolution Support**: Generate images in 1K, 2K, or 4K resolution
- **Batch Processing**: Generate up to 16 images in a single request (depending on resolution)
- **Creative Controls**: Adjust guidance scale and temperature for creative control
- **Real-time Progress Tracking**: Monitor generation progress with a progress bar
- **High-quality Upscaling**: Advanced upscaling techniques for higher resolution outputs
- **Image Management**: View, download individual images or all generated images at once
- **Responsive Design**: Works on desktop and mobile devices

## Installation

### Prerequisites

- Python 3.8 or higher (tested with Python 3.11.9)
- CUDA-compatible GPU (optional, but recommended for faster generation)
- At least 8GB of RAM (16GB+ recommended for 2K/4K images)
- ~5GB of disk space for model files

### Step 1: Clone the repository

```bash
git clone https://github.com/yourusername/janus-image-generator.git
cd janus-image-generator
```

### Step 2: Set up the environment

```bash
# Create and activate a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
python setup.py
```

### Step 3: Download the model

If you haven't already downloaded a Janus model, you can use the `download_model.py` script:

```bash
python download_model.py --model Janus-Pro-1B
```

Available model options:
- `Janus-Pro-1B` (Default, best balance of quality and speed)
- `Janus-Pro-7B` (Higher quality, but requires more GPU memory)
- `Janus-1.3B` (Smaller model, faster)
- `JanusFlow-1.3B` (Alternative architecture)

### Step 4: Run the application

```bash
python gpu.py
```

The application will be available at http://localhost:5000

## Usage Guide

### Generating Images

1. **Enter a description**: Type a detailed description of the image you want to generate in the text area.
   - Be specific about details like style, colors, lighting, and composition
   - Example: "A stunning winter landscape with snow-covered pine trees, a frozen lake, and a small log cabin with warm lights in the windows, photorealistic style"

2. **Select generation settings**:
   - **Number of Images**: How many images to generate (1-16)
   - **Resolution**: Select from 1K (standard), 2K (high resolution), or 4K (ultra HD)
   - **Guidance Scale**: Controls how closely the model follows your prompt (higher = more faithful to the prompt)
   - **Creativity**: Controls the randomness of generation (higher = more creative, lower = more deterministic)

3. **Generate Images**: Click the "Generate Images" button and wait for the process to complete.

### Managing Generated Images

- **View Full-size**: Click on any thumbnail to view the full-resolution image
- **Download Individual Images**: While viewing a full-size image, click "Download" to save it
- **Download All Images**: Click "Download All" to save all generated images
- **Start New Generation**: Click "New Generation" to clear the results and start over

## Configuration

### Model Settings

The application uses the local model in the `janus_pro1b` folder by default. To use a different model:

1. Download the desired model using the `download_model.py` script
2. Edit `gpu.py` and update the `model_path` variable in the `load_model()` function to point to your model folder

### Advanced Settings

Edit the `gpu.py` file to modify these advanced settings:

- **GPU Memory Management**: Adjust batch sizes and parallelism based on your GPU memory
- **Image Quality**: Modify upscaling algorithms and JPEG quality settings
- **Server Configuration**: Change host address and port number

## Troubleshooting

### Common Issues

#### "No module named 'cv2'"
```
pip install opencv-python
```

#### "No module named 'janus'"
```
pip install git+https://github.com/deepseek-ai/Janus.git
```

#### "CUDA out of memory"
- Lower the resolution in the UI
- Generate fewer images at once
- Use a smaller model (e.g., Janus-Pro-1B instead of Janus-Pro-7B)

#### "Error loading model"
- Ensure the model path in `gpu.py` matches your folder structure
- Check that you have all the model files downloaded correctly

#### Slow Image Generation
- Use a GPU if available
- Close other GPU-intensive applications
- For CPU-only setups, generate fewer images and use 1K resolution


## Credits and License

- **Janus Model**: Created by [DeepSeek AI](https://github.com/deepseek-ai/Janus)
- **Janus License**: The use of Janus models is subject to the [DeepSeek Model License](https://github.com/deepseek-ai/Janus/blob/main/LICENSE-MODEL)
- **Application Code**: MIT License

### Citation

If you use this application for academic or research purposes, please cite the original Janus papers:

```
@article{chen2025janus,
  title={Janus-Pro: Unified Multimodal Understanding and Generation with Data and Model Scaling},
  author={Chen, Xiaokang and Wu, Zhiyu and Liu, Xingchao and Pan, Zizheng and Liu, Wen and Xie, Zhenda and Yu, Xingkai and Ruan, Chong},
  journal={arXiv preprint arXiv:2501.17811},
  year={2025}
}

@article{wu2024janus,
  title={Janus: Decoupling visual encoding for unified multimodal understanding and generation},
  author={Wu, Chengyue and Chen, Xiaokang and Wu, Zhiyu and Ma, Yiyang and Liu, Xingchao and Pan, Zizheng and Liu, Wen and Xie, Zhenda and Yu, Xingkai and Ruan, Chong and others},
  journal={arXiv preprint arXiv:2410.13848},
  year={2024}
}
```

## Contact

For questions or support, please [open an issue](https://github.com/yourusername/janus-image-generator/issues) on GitHub.
