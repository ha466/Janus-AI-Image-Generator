# Janus AI Image Generator: Where Pixels Party!

![Janus AI](static/janus-logo.png)  
*Our mascot Janus - two faces because one's always judging your prompt*

## Overview

Janus AI Image Generator is like a magic lamp for your weirdest ideas, except instead of a genie, you get GPUs working overtime. We've trained some very sleep-deprived AI models (they prefer "chronically caffeinated") to turn your text into images. Describe it, and we'll generate it - whether that's "a majestic unicorn eating tacos" or "my sleep paralysis demon in business casual."

## Features

- **Text-to-Image Generation**: Turn your wildest dreams into pixels (we won't judge... much)
- **Multiple Resolution Support**: 1K for memes, 4K for art so sharp it could cut reality
- **Batch Processing**: Generate 16 images before you finish saying "CUDA out of memory"
- **Creative Controls**: Because sometimes you want a photorealistic cat, sometimes you want a cubist giraffe
- **Real-time Progress Tracking**: Watch the progress bar move slower than your motivation on Monday
- **High-quality Upscaling**: Making your images clearer than your life choices
- **Image Management**: Download button included - your parents will finally believe you're an "artist"
- **Responsive Design**: Works on desktop, mobile, and your smart fridge (because why not?)

## Installation: The Quest Begins

### Prerequisites

- Python 3.8+ (We'd say 3.6 but that's like using a flip phone)
- CUDA-compatible GPU (Or a potato and infinite patience)
- 8GB RAM (16GB if you want to generate more than stick figures)
- 5GB disk space (Less than your meme folder)

### Step 1: Clone Like It's 1999

```bash
git clone https://github.com/ha466/Janus-AI-Image-Generator.git
cd janus-image-generator
```

### Step 2: Environment Setup (Not the Jungle Kind)

```bash
# Create a virtual environment so we don't fight with your other Python projects
python -m venv venv
source venv/bin/activate  # Windows users: venv\Scripts\activate (good luck)

# Install dependencies - this may take longer than your last relationship
python setup.py
```

### Step 3: Download Our Digital Brainchildren

```bash
python download_model.py --model Janus-Pro-1B
```

Model options explained:  
- `Janus-Pro-1B`: Your reliable Honda Civic of AI models  
- `Janus-Pro-7B`: The Hulk Hogan of models (needs GPU muscles)  
- `Janus-1.3B`: For when you're in a hurry and quality is optional  
- `JanusFlow-1.3B`: The abstract expressionist of the group  

### Step 4: Launch the Magic

```bash
python gpu.py
```

Now visit http://localhost:5000 and try not to crash your browser with "shrek in space"

## Usage Guide: From Mundane to Masterpiece

### Step 1: Describe Your Vision

Channel your inner Shakespeare:  
Good: "A cybernetic samurai squirrel wielding a lightsaber made of cheese"  
Bad: "thing with stuff" (Our AI will judge you silently)

### Step 2: Tweak the Knobs

- **Number of Images**: How many attempts you need before getting it right  
- **Resolution**: 1K = Instagram, 4K = "I need to see every pore"  
- **Guidance Scale**: 1 = "Do whatever", 10 = "HELICOPTER PARENT MODE"  
- **Creativity**: 1 = Paint-by-numbers, 10 = Toddler on sugar  

### Step 3: Generate and Wait

Click the button and:  
1. Watch the progress bar  
2. Question your life choices  
3. Consider making coffee  
4. Get surprised when it actually works  

## Troubleshooting: Panic Mode

### "No module named 'cv2'"
```
pip install opencv-python
# If that fails, try sacrificing a USB drive to the tech gods
```

### "CUDA out of memory"
- Generate fewer images than your GPU's self-esteem can handle  
- Switch to 1K resolution (Your 8K monitor will forgive you)  
- Try the model that thinks it's 2010 ("Janus-1.3B")  

### "Error loading model"
- Did you unzip it?  
- Did you look at it funny?  
- Try turning it off and on again (classic)  

### Slow Generation?
- GPU: "I'm speed!"  
- CPU: "I'm... contemplating the universe..."  
Pro tip: Use generation time to:  
1. Learn a new language  
2. Write your novel  
3. Question why you didn't buy a better GPU  

## License & Credits

- **Janus Models**: Made with ❤️ and ☕️ by DeepSeek AI  
- **App Code**: MIT License - do whatever, just don't blame us if it creates a black hole  
- **You**: Legendary for reading this far  

### Citation (For Academic Flex)

```bibtex
@article{janus2024,
  title={How We Made GPUs Cry},
  author={All Night Coders},
  journal={Journal of Questionable Life Choices},
  year={2024}
}
```

## Need Help?

- GitHub Issues: Our digital therapy couch  
- Email: support@janus.ai (Responses may include cat pictures)  
- Carrier Pigeon: Please attach SSD with error logs  

Now go forth and generate things that would confuse Van Gogh! 🎨🚀
