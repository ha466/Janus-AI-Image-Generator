import os
import torch
import numpy as np
import PIL.Image
from PIL import Image
from flask import Flask, request, jsonify, send_from_directory
from transformers import AutoModelForCausalLM
from janus.models import MultiModalityCausalLM, VLChatProcessor
from threading import Thread
import time

# Try to import OpenCV, with fallback if not available
try:
    import cv2
    OPENCV_AVAILABLE = True
except ImportError:
    print("OpenCV (cv2) not found. Using Pillow for image processing instead.")
    OPENCV_AVAILABLE = False

app = Flask(__name__, static_folder='static')

# Make sure we have directories to save generated images
os.makedirs('static/generated_images', exist_ok=True)
os.makedirs('static/generated_images/thumbnails', exist_ok=True)

# Global variables for model and processor
vl_gpt = None
vl_chat_processor = None
generation_status = {}

def load_model():
    global vl_gpt, vl_chat_processor
    
    # Fixed path name - exactly match your folder name
    model_path = "janus_pro1b"  # Local folder containing the model
    
    try:
        # Verify GPU is available
        if not torch.cuda.is_available():
            print("ERROR: CUDA is not available. This script requires GPU.")
            return False
            
        # Initialize the processor and tokenizer
        print(f"Loading processor from {model_path}...")
        vl_chat_processor = VLChatProcessor.from_pretrained(model_path)
        
        # Load the model with GPU optimizations
        print(f"Loading model from {model_path}...")
        
        # Enable GPU memory optimization
        torch.cuda.empty_cache()
        
        # Load with bfloat16 precision for better performance
        vl_gpt = AutoModelForCausalLM.from_pretrained(
            model_path, 
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map="auto"  # Automatically distribute model across GPUs if multiple
        )
        
        print(f"Model loaded to GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        print(f"Current allocated memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        
        vl_gpt = vl_gpt.eval()
        
        print("Model loaded successfully in GPU-optimized mode")
        return True
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return False


def upscale_image(img, target_size):
    """
    Upscale an image to a target size using the best available method.
    """
    # Convert to PIL Image if it's a numpy array
    if isinstance(img, np.ndarray):
        img_pil = Image.fromarray(img)
    else:
        img_pil = img
        
    current_size = img_pil.size
    
    # If OpenCV is available and upscaling is significant, use it for better quality
    if OPENCV_AVAILABLE and target_size[0] / current_size[0] > 2:
        # First upscale using Pillow to an intermediate size
        intermediate_size = (current_size[0] * 2, current_size[1] * 2)
        img_pil = img_pil.resize(intermediate_size, Image.LANCZOS if hasattr(Image, 'LANCZOS') else Image.BICUBIC)
        
        # Convert to OpenCV format
        img_cv = np.array(img_pil)
        img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)
        
        try:
            # Try to use OpenCV's super resolution if available
            if hasattr(cv2, 'dnn_superres'):
                sr = cv2.dnn_superres.DnnSuperResImpl_create()
                path = "models/EDSR_x2.pb"
                
                if os.path.exists(path):
                    sr.readModel(path)
                    sr.setModel("edsr", 2)
                    img_cv = sr.upsample(img_cv)
                else:
                    # Fallback to bicubic interpolation
                    img_cv = cv2.resize(img_cv, target_size, interpolation=cv2.INTER_CUBIC)
            else:
                # If dnn_superres is not available
                img_cv = cv2.resize(img_cv, target_size, interpolation=cv2.INTER_CUBIC)
        except Exception as e:
            print(f"OpenCV upscaling error: {str(e)}. Using fallback method.")
            img_cv = cv2.resize(img_cv, target_size, interpolation=cv2.INTER_CUBIC)
        
        img_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
        return img_cv
    else:
        # For smaller upscaling or if OpenCV is not available, use Pillow
        resample_method = Image.LANCZOS if hasattr(Image, 'LANCZOS') else Image.BICUBIC
        img_pil = img_pil.resize(target_size, resample_method)
        return np.array(img_pil)


@torch.inference_mode()
def generate_image_async(prompt_text, image_id, resolution="1K", cfg_weight=7.0, temperature=0.8, num_images=1):
    global vl_gpt, vl_chat_processor, generation_status
    
    # Record start time for performance tracking
    start_time = time.time()
    print(f"[INFO] Starting GPU image generation for ID: {image_id}")
    print(f"[INFO] Prompt: \"{prompt_text}\"")
    
    generation_status[image_id].update({
        "status": "processing", 
        "completed": 0, 
        "total": num_images,
        "started_at": start_time,
        "prompt": prompt_text
    })
    
    try:
        # Ensure model is loaded
        if vl_gpt is None or vl_chat_processor is None:
            success = load_model()
            if not success:
                generation_status[image_id] = {"status": "error", "message": "Failed to load model"}
                return None
        
        # For GPU optimization, determine max parallel images based on resolution
        if resolution.upper() == "4K":
            max_images = 4
        elif resolution.upper() == "2K":
            max_images = 8
        else:
            max_images = 16
            
        num_images = min(num_images, max_images)
        
        # Clear CUDA cache before generation
        torch.cuda.empty_cache()
        
        # Get the correct device
        device = next(vl_gpt.parameters()).device
        print(f"Using device: {device} for generating {num_images} images at {resolution} resolution")
        
        # Prepare conversation format - use role format that matches model version
        conversation = [
            {
                "role": "<|User|>",
                "content": prompt_text,
            },
            {"role": "<|Assistant|>", "content": ""},
        ]
        
        # Apply SFT format and add image start tag
        sft_format = vl_chat_processor.apply_sft_template_for_multi_turn_prompts(
            conversations=conversation,
            sft_format=vl_chat_processor.sft_format,
            system_prompt="",
        )
        
        # Try different image tags based on model version
        if hasattr(vl_chat_processor, 'image_start_tag'):
            image_start_tag = vl_chat_processor.image_start_tag
        elif "<begin_of_image>" in vl_chat_processor.tokenizer.get_vocab():
            image_start_tag = "<begin_of_image>"
        else:
            image_start_tag = "<image>"
        
        prompt = sft_format + image_start_tag
        print(f"Generated prompt: {prompt}")
        
        # Image generation parameters
        parallel_size = num_images
        image_token_num_per_image = 576
        img_size = 384
        patch_size = 16
        
        # Encode the prompt
        input_ids = vl_chat_processor.tokenizer.encode(prompt)
        input_ids = torch.LongTensor(input_ids).to(device)
        
        # Create tokens for conditional and unconditional guidance
        tokens = torch.zeros((parallel_size*2, len(input_ids)), dtype=torch.int).to(device)
        for i in range(parallel_size*2):
            tokens[i, :] = input_ids
            if i % 2 != 0:
                tokens[i, 1:-1] = vl_chat_processor.pad_id
        
        # Get token embeddings
        inputs_embeds = vl_gpt.language_model.get_input_embeddings()(tokens)
        
        # Initialize generated tokens
        generated_tokens = torch.zeros((parallel_size, image_token_num_per_image), dtype=torch.int).to(device)
        
        # Generate tokens one by one
        past_key_values = None
        
        # On GPU we can process all tokens in one go
        for i in range(image_token_num_per_image):
            outputs = vl_gpt.language_model.model(
                inputs_embeds=inputs_embeds, 
                use_cache=True, 
                past_key_values=past_key_values
            )
            past_key_values = outputs.past_key_values
            hidden_states = outputs.last_hidden_state
            
            # Get logits and apply classifier-free guidance
            logits = vl_gpt.gen_head(hidden_states[:, -1, :])
            logit_cond = logits[0::2, :]
            logit_uncond = logits[1::2, :]
            
            logits = logit_uncond + cfg_weight * (logit_cond - logit_uncond)
            probs = torch.softmax(logits / temperature, dim=-1)
            
            # Sample next token
            next_token = torch.multinomial(probs, num_samples=1)
            generated_tokens[:, i] = next_token.squeeze(dim=-1)
            
            # Prepare embedding for next step
            next_token = torch.cat([next_token.unsqueeze(dim=1), next_token.unsqueeze(dim=1)], dim=1).view(-1)
            img_embeds = vl_gpt.prepare_gen_img_embeds(next_token)
            inputs_embeds = img_embeds.unsqueeze(dim=1)
            
            # Update progress every 10%
            if (i+1) % (image_token_num_per_image // 10) == 0:
                progress = (i+1) / image_token_num_per_image
                generation_status[image_id]["progress"] = progress
                print(f"[INFO] Generation progress: {progress*100:.1f}%")
        
        # Decode the generated tokens to images
        dec = vl_gpt.gen_vision_model.decode_code(
            generated_tokens.to(dtype=torch.int), 
            shape=[parallel_size, 8, img_size//patch_size, img_size//patch_size]
        )
        dec = dec.to(torch.float32).cpu().numpy().transpose(0, 2, 3, 1)
        
        # Post-process and save images
        dec = np.clip((dec + 1) / 2 * 255, 0, 255)
        image_paths = []
        
        # Define resolution parameters
        if resolution.upper() == "2K":
            target_size = (2048, 2048)
        elif resolution.upper() == "4K":
            target_size = (4096, 4096)
        else:  # Default 1K
            target_size = (1024, 1024)
        
        for i in range(parallel_size):
            # Update status
            generation_status[image_id]["completed"] = i
            
            # Create base image
            visual_img = np.zeros((img_size, img_size, 3), dtype=np.uint8)
            visual_img[:, :] = dec[i]
            
            # Save thumbnail (original resolution)
            thumb_filename = f"thumbnails/img_{image_id}_{i}.jpg"
            thumb_path = os.path.join('static/generated_images', thumb_filename)
            PIL.Image.fromarray(visual_img).save(thumb_path)
            
            # Upscale if higher resolution requested
            if resolution.upper() != "1K":
                # Apply upscaling
                print(f"Upscaling image {i+1}/{parallel_size} to {target_size}")
                upscaled_img = upscale_image(visual_img, target_size)
                
                # Save high-res version
                filename = f"img_{image_id}_{i}.jpg"
                save_path = os.path.join('static/generated_images', filename)
                PIL.Image.fromarray(upscaled_img.astype(np.uint8)).save(save_path, quality=95)
            else:
                # For 1K, just resize the original
                upscaled_img = upscale_image(visual_img, (1024, 1024))
                filename = f"img_{image_id}_{i}.jpg"
                save_path = os.path.join('static/generated_images', filename)
                PIL.Image.fromarray(upscaled_img.astype(np.uint8)).save(save_path)
            
            image_paths.append({
                "thumbnail": f"/static/generated_images/{thumb_filename}",
                "full": f"/static/generated_images/{filename}"
            })
            
            print(f"[INFO] Saving image {i+1}/{parallel_size}")
        
        # Add timing information for performance monitoring
        end_time = time.time()
        generation_time = end_time - start_time
        generation_status[image_id] = {
            "status": "complete", 
            "images": image_paths,
            "completed": parallel_size,
            "total": parallel_size,
            "prompt": prompt_text,
            "generation_time": generation_time
        }
        
        # Clear cache after generation is complete
        torch.cuda.empty_cache()
        
        print(f"[INFO] Generation complete for ID: {image_id}")
        print(f"[INFO] Generation time: {generation_time:.2f} seconds")
        
        return image_paths
        
    except Exception as e:
        import traceback
        print(f"[ERROR] Error in image generation: {str(e)}")
        print(traceback.format_exc())
        generation_status[image_id] = {"status": "error", "message": str(e)}
        return None


@app.route('/api/generate', methods=['POST'])
def api_generate():
    data = request.json
    prompt = data.get('prompt', '')
    
    # Get generation parameters
    resolution = data.get('resolution', '1K')  # 1K, 2K, or 4K
    cfg_weight = float(data.get('cfg_weight', 7.0))
    temperature = float(data.get('temperature', 0.8))
    
    # Check if this is a manual confirmation or initial request
    confirmed = data.get('confirmed', False)
    
    if not prompt:
        return jsonify({"error": "No prompt provided"}), 400
    
    try:
        # Generate a unique ID for this generation or use existing
        image_id = data.get('image_id', str(int(time.time() * 1000)))
        
        # Determine max images based on resolution
        if resolution.upper() == "4K":
            max_images = 4
        elif resolution.upper() == "2K":
            max_images = 8
        else:
            max_images = 16
            
        num_images = min(int(data.get('num_images', 1)), max_images)
        
        # If this is the initial request (not confirmed), add to pending queue
        if not confirmed:
            # Create entry but don't start processing
            generation_status[image_id] = {
                "status": "pending",  # Change from "processing" to "pending"
                "prompt": prompt,
                "settings": {
                    "resolution": resolution,
                    "cfg_weight": cfg_weight,
                    "temperature": temperature,
                    "num_images": num_images
                },
                "created_at": time.time(),
            }
            
            # Return ID for confirmation
            return jsonify({
                "success": True,
                "message": "Generation queued and awaiting confirmation",
                "id": image_id,
                "status": "pending"
            })
        
        # If confirmed, start the generation process
        else:
            # Get settings from the pending request
            pending_info = generation_status.get(image_id, {})
            if not pending_info or pending_info.get('status') != 'pending':
                return jsonify({"error": "Invalid or expired request ID"}), 400
                
            # Extract settings from pending request
            settings = pending_info.get('settings', {})
            resolution = settings.get('resolution', resolution)
            cfg_weight = settings.get('cfg_weight', cfg_weight)
            temperature = settings.get('temperature', temperature)
            num_images = settings.get('num_images', num_images)
            
            # Start generation in a background thread
            thread = Thread(target=generate_image_async, 
                           args=(prompt, image_id, resolution, cfg_weight, temperature, num_images))
            thread.daemon = True
            thread.start()
            
            # Update status
            generation_status[image_id]["status"] = "processing"
            generation_status[image_id]["started_at"] = time.time()
            
            # Log the start of generation
            print(f"[INFO] Started manual generation for prompt: \"{prompt}\"")
            print(f"[INFO] Generation ID: {image_id}")
            print(f"[INFO] Settings: Resolution={resolution}, CFG={cfg_weight}, Temp={temperature}, Images={num_images}")
            
            return jsonify({
                "success": True,
                "message": "Generation confirmed and started",
                "id": image_id,
                "status": "processing"
            })
            
    except Exception as e:
        print(f"[ERROR] Error in generation request: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/status/<image_id>', methods=['GET'])
def check_status(image_id):
    """Check the status of an ongoing generation"""
    if image_id in generation_status:
        return jsonify(generation_status[image_id])
    else:
        return jsonify({"status": "not_found"}), 404


@app.route('/')
def index():
    return send_from_directory('static', 'index.html')


@app.route('/favicon.ico')
def favicon():
    return send_from_directory('static', 'favicon.ico')


@app.route('/static/<path:path>')
def serve_static(path):
    return send_from_directory('static', path)


# Add a new route for manual confirmation of pending generations
@app.route('/api/confirm/<image_id>', methods=['POST'])
def confirm_generation(image_id):
    """Manually confirm a pending generation"""
    if image_id not in generation_status:
        return jsonify({"error": "Generation ID not found"}), 404
        
    # Check if it's a pending generation
    generation_info = generation_status[image_id]
    if generation_info.get('status') != 'pending':
        return jsonify({"error": "Generation is not in pending state"}), 400
        
    # Get the details from the pending generation
    prompt = generation_info.get('prompt', '')
    settings = generation_info.get('settings', {})
    
    # Create request with confirmation flag
    confirmation_request = {
        'prompt': prompt,
        'image_id': image_id,
        'confirmed': True,
        'resolution': settings.get('resolution', '1K'),
        'cfg_weight': settings.get('cfg_weight', 7.0),
        'temperature': settings.get('temperature', 0.8),
        'num_images': settings.get('num_images', 1)
    }
    
    # Update request.json for the api_generate function
    original_json = request.json
    request.json = confirmation_request
    response = api_generate()
    request.json = original_json
    return response


if __name__ == '__main__':
    print(f"Python version: {os.sys.version}")
    
    # Try to load the model at startup
    print("Attempting to load model at startup...")
    load_model()
    
    # Create models directory for upscaler models
    os.makedirs('models', exist_ok=True)
    
    # Start the Flask app - optimized for GPU workloads
    app.run(debug=False, host='0.0.0.0', port=5000) 