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
app_config = {
    'model_path': 'janus_pro1b',
    'max_history': 30,
    'example_prompts': [
        "A photo-realistic image of a futuristic city with flying cars and neon lights",
        "A serene Japanese garden with cherry blossoms, wooden bridge and lake reflection",
        "A cybernetic samurai warrior in business casual outfit eating tacos",
        "A majestic dragon perched on a misty mountain peak at sunset",
        "A cozy interior of a modern living room with large panoramic windows"
    ]
}

def load_model():
    global vl_gpt, vl_chat_processor
    model_path = app_config['model_path']

    try:
        if not torch.cuda.is_available():
            print("ERROR: CUDA is not available. This script requires GPU.")
            return False

        print(f"Loading processor from {model_path}...")
        vl_chat_processor = VLChatProcessor.from_pretrained(model_path)

        print(f"Loading model from {model_path}...")
        torch.cuda.empty_cache()

        vl_gpt = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )

        print(f"Model loaded to GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

        vl_gpt = vl_gpt.eval()
        print("Model loaded successfully in GPU-optimized mode")
        return True
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return False


def upscale_image(img, target_size):
    if isinstance(img, np.ndarray):
        img_pil = Image.fromarray(img)
    else:
        img_pil = img

    current_size = img_pil.size

    if OPENCV_AVAILABLE and target_size[0] / current_size[0] > 2:
        intermediate_size = (current_size[0] * 2, current_size[1] * 2)
        img_pil = img_pil.resize(intermediate_size, Image.LANCZOS if hasattr(Image, 'LANCZOS') else Image.BICUBIC)

        img_cv = np.array(img_pil)
        img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)

        try:
            if hasattr(cv2, 'dnn_superres'):
                sr = cv2.dnn_superres.DnnSuperResImpl_create()
                path = "models/EDSR_x2.pb"
                if os.path.exists(path):
                    sr.readModel(path)
                    sr.setModel("edsr", 2)
                    img_cv = sr.upsample(img_cv)
                else:
                    img_cv = cv2.resize(img_cv, target_size, interpolation=cv2.INTER_CUBIC)
            else:
                img_cv = cv2.resize(img_cv, target_size, interpolation=cv2.INTER_CUBIC)
        except Exception as e:
            print(f"OpenCV upscaling error: {str(e)}. Using fallback method.")
            img_cv = cv2.resize(img_cv, target_size, interpolation=cv2.INTER_CUBIC)

        img_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
        return img_cv
    else:
        resample_method = Image.LANCZOS if hasattr(Image, 'LANCZOS') else Image.BICUBIC
        img_pil = img_pil.resize(target_size, resample_method)
        return np.array(img_pil)


@torch.inference_mode()
def generate_image_async(prompt_text, image_id, resolution="1K", cfg_weight=7.0, temperature=0.8, num_images=1):
    global vl_gpt, vl_chat_processor, generation_status

    start_time = time.time()
    print(f"[INFO] Starting GPU image generation for ID: {image_id}")
    print(f"[INFO] Prompt: \"{prompt_text}\"")

    generation_status[image_id].update({
        "status": "processing",
        "completed": 0,
        "total": num_images,
        "started_at": start_time,
        "progress": 0.0,
        "prompt": prompt_text
    })

    if len(generation_status) > app_config['max_history']:
        oldest_key = min(generation_status.keys(), key=lambda k: generation_status[k].get('started_at', generation_status[k].get('created_at', 0)))
        del generation_status[oldest_key]

    try:
        if vl_gpt is None or vl_chat_processor is None:
            success = load_model()
            if not success:
                generation_status[image_id].update({"status": "error", "message": "Failed to load model"})
                return None

        if resolution.upper() == "4K":
            max_images = 4
        elif resolution.upper() == "2K":
            max_images = 8
        else:
            max_images = 16

        num_images = min(max(1, num_images), max_images)
        torch.cuda.empty_cache()
        device = next(vl_gpt.parameters()).device

        conversation = [
            {"role": "<|User|>", "content": prompt_text},
            {"role": "<|Assistant|>", "content": ""},
        ]

        sft_format = vl_chat_processor.apply_sft_template_for_multi_turn_prompts(
            conversations=conversation,
            sft_format=vl_chat_processor.sft_format,
            system_prompt="",
        )

        if hasattr(vl_chat_processor, 'image_start_tag'):
            image_start_tag = vl_chat_processor.image_start_tag
        elif "<begin_of_image>" in vl_chat_processor.tokenizer.get_vocab():
            image_start_tag = "<begin_of_image>"
        else:
            image_start_tag = "<image>"

        prompt = sft_format + image_start_tag
        parallel_size = num_images
        image_token_num_per_image = 576
        img_size = 384
        patch_size = 16

        input_ids = vl_chat_processor.tokenizer.encode(prompt)
        input_ids = torch.LongTensor(input_ids).to(device)

        tokens = torch.zeros((parallel_size*2, len(input_ids)), dtype=torch.int).to(device)
        for i in range(parallel_size*2):
            tokens[i, :] = input_ids
            if i % 2 != 0:
                tokens[i, 1:-1] = vl_chat_processor.pad_id

        inputs_embeds = vl_gpt.language_model.get_input_embeddings()(tokens)
        generated_tokens = torch.zeros((parallel_size, image_token_num_per_image), dtype=torch.int).to(device)

        past_key_values = None

        for i in range(image_token_num_per_image):
            if generation_status.get(image_id, {}).get("status") == "cancelled":
                print(f"[INFO] Generation {image_id} cancelled by user.")
                return None

            outputs = vl_gpt.language_model.model(
                inputs_embeds=inputs_embeds,
                use_cache=True,
                past_key_values=past_key_values
            )
            past_key_values = outputs.past_key_values
            hidden_states = outputs.last_hidden_state

            logits = vl_gpt.gen_head(hidden_states[:, -1, :])
            logit_cond = logits[0::2, :]
            logit_uncond = logits[1::2, :]

            logits = logit_uncond + cfg_weight * (logit_cond - logit_uncond)
            probs = torch.softmax(logits / temperature, dim=-1)

            next_token = torch.multinomial(probs, num_samples=1)
            generated_tokens[:, i] = next_token.squeeze(dim=-1)

            next_token = torch.cat([next_token.unsqueeze(dim=1), next_token.unsqueeze(dim=1)], dim=1).view(-1)
            img_embeds = vl_gpt.prepare_gen_img_embeds(next_token)
            inputs_embeds = img_embeds.unsqueeze(dim=1)

            if (i + 1) % 10 == 0 or (i + 1) == image_token_num_per_image:
                progress = round((i + 1) / image_token_num_per_image, 2)
                generation_status[image_id]["progress"] = progress

        dec = vl_gpt.gen_vision_model.decode_code(
            generated_tokens.to(dtype=torch.int),
            shape=[parallel_size, 8, img_size//patch_size, img_size//patch_size]
        )
        dec = dec.to(torch.float32).cpu().numpy().transpose(0, 2, 3, 1)
        dec = np.clip((dec + 1) / 2 * 255, 0, 255)
        image_paths = []

        if resolution.upper() == "2K":
            target_size = (2048, 2048)
        elif resolution.upper() == "4K":
            target_size = (4096, 4096)
        else:
            target_size = (1024, 1024)

        for i in range(parallel_size):
            generation_status[image_id]["completed"] = i + 1

            visual_img = np.zeros((img_size, img_size, 3), dtype=np.uint8)
            visual_img[:, :] = dec[i]

            thumb_filename = f"thumbnails/img_{image_id}_{i}.jpg"
            thumb_path = os.path.join('static/generated_images', thumb_filename)
            PIL.Image.fromarray(visual_img).save(thumb_path)

            if resolution.upper() != "1K":
                upscaled_img = upscale_image(visual_img, target_size)
                filename = f"img_{image_id}_{i}.jpg"
                save_path = os.path.join('static/generated_images', filename)
                PIL.Image.fromarray(upscaled_img.astype(np.uint8)).save(save_path, quality=95)
            else:
                upscaled_img = upscale_image(visual_img, (1024, 1024))
                filename = f"img_{image_id}_{i}.jpg"
                save_path = os.path.join('static/generated_images', filename)
                PIL.Image.fromarray(upscaled_img.astype(np.uint8)).save(save_path)

            image_paths.append({
                "thumbnail": f"/static/generated_images/{thumb_filename}",
                "full": f"/static/generated_images/{filename}"
            })

        end_time = time.time()
        generation_time = round(end_time - start_time, 2)

        generation_status[image_id] = {
            "id": image_id,
            "status": "complete",
            "images": image_paths,
            "completed": parallel_size,
            "total": parallel_size,
            "prompt": prompt_text,
            "started_at": start_time,
            "completed_at": end_time,
            "generation_time": generation_time,
            "settings": {
                "resolution": resolution,
                "cfg_weight": cfg_weight,
                "temperature": temperature,
                "num_images": parallel_size
            }
        }

        torch.cuda.empty_cache()
        return image_paths

    except Exception as e:
        import traceback
        print(f"[ERROR] Error in image generation: {str(e)}")
        print(traceback.format_exc())

        end_time = time.time()
        generation_status[image_id] = {
            "id": image_id,
            "status": "error",
            "message": str(e),
            "prompt": prompt_text,
            "started_at": start_time,
            "error_at": end_time,
            "settings": {
                "resolution": resolution,
                "cfg_weight": cfg_weight,
                "temperature": temperature,
                "num_images": num_images
            }
        }
        return None


@app.route('/api/generate', methods=['POST'])
def api_generate():
    data = request.json or {}
    prompt = data.get('prompt', '').strip()

    if not prompt:
        return jsonify({"error": "No prompt provided"}), 400

    resolution = str(data.get('resolution', '1K')).upper()
    if resolution not in ['1K', '2K', '4K']:
        resolution = '1K'

    try:
        cfg_weight = float(data.get('cfg_weight', 7.0))
        cfg_weight = max(1.0, min(20.0, cfg_weight))
    except (ValueError, TypeError):
        cfg_weight = 7.0

    try:
        temperature = float(data.get('temperature', 0.8))
        temperature = max(0.1, min(2.0, temperature))
    except (ValueError, TypeError):
        temperature = 0.8

    try:
        num_images = int(data.get('num_images', 1))
        num_images = max(1, min(16, num_images))
    except (ValueError, TypeError):
        num_images = 1

    confirmed = data.get('confirmed', True)
    image_id = str(data.get('image_id') or int(time.time() * 1000))

    try:
        generation_status[image_id] = {
            "id": image_id,
            "status": "processing" if confirmed else "pending",
            "prompt": prompt,
            "progress": 0.0,
            "completed": 0,
            "total": num_images,
            "settings": {
                "resolution": resolution,
                "cfg_weight": cfg_weight,
                "temperature": temperature,
                "num_images": num_images
            },
            "created_at": time.time(),
            "started_at": time.time() if confirmed else None
        }

        if confirmed:
            thread = Thread(target=generate_image_async,
                           args=(prompt, image_id, resolution, cfg_weight, temperature, num_images))
            thread.daemon = True
            thread.start()

            return jsonify({
                "success": True,
                "message": "Generation started",
                "id": image_id,
                "status": "processing"
            })
        else:
            return jsonify({
                "success": True,
                "message": "Generation queued and awaiting confirmation",
                "id": image_id,
                "status": "pending"
            })

    except Exception as e:
        print(f"[ERROR] Error in generation request: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/status/<image_id>', methods=['GET'])
def check_status(image_id):
    if image_id in generation_status:
        return jsonify(generation_status[image_id])
    else:
        return jsonify({"status": "not_found"}), 404


@app.route('/api/cancel/<image_id>', methods=['POST'])
def cancel_generation(image_id):
    if image_id in generation_status:
        generation_status[image_id]["status"] = "cancelled"
        generation_status[image_id]["message"] = "Cancelled by user"
        return jsonify({"success": True, "message": f"Generation {image_id} cancelled"})
    return jsonify({"error": "Generation ID not found"}), 404


@app.route('/api/history', methods=['GET'])
def get_history():
    sorted_history = sorted(
        [status for status in generation_status.values() if 'started_at' in status or 'created_at' in status],
        key=lambda x: x.get('started_at') or x.get('created_at', 0),
        reverse=True
    )
    return jsonify(sorted_history)


@app.route('/api/clear_history', methods=['POST'])
def clear_history():
    global generation_status
    keys_to_remove = [k for k, v in generation_status.items() if v.get('status') in ['complete', 'error', 'cancelled']]
    for k in keys_to_remove:
        del generation_status[k]
    return jsonify({"success": True, "cleared_count": len(keys_to_remove)})


@app.route('/api/system', methods=['GET'])
def get_system_info():
    gpu_info = {}
    gpu_available = torch.cuda.is_available()
    if gpu_available:
        try:
            gpu_info = {
                "name": torch.cuda.get_device_name(0),
                "memory_total": round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2),
                "memory_used": round(torch.cuda.memory_allocated() / 1e9, 2),
            }
        except Exception:
            gpu_info = {"name": "CUDA GPU Detected", "memory_total": 0, "memory_used": 0}

    return jsonify({
        "python_version": os.sys.version.split()[0],
        "torch_version": torch.__version__,
        "gpu_available": gpu_available,
        "gpu_info": gpu_info,
        "model_path": app_config['model_path'],
        "model_loaded": vl_gpt is not None
    })


@app.route('/')
def index():
    return send_from_directory('static', 'index.html')


@app.route('/favicon.ico')
def favicon():
    return send_from_directory('static', 'favicon.ico')


@app.route('/static/<path:path>')
def serve_static(path):
    return send_from_directory('static', path)


@app.route('/api/example_prompts', methods=['GET'])
def get_example_prompts():
    return jsonify(app_config['example_prompts'])


@app.route('/api/confirm/<image_id>', methods=['POST'])
def confirm_generation(image_id):
    if image_id not in generation_status:
        return jsonify({"error": "Generation ID not found"}), 404

    generation_info = generation_status[image_id]
    if generation_info.get('status') != 'pending':
        return jsonify({"error": "Generation is not in pending state"}), 400

    prompt = generation_info.get('prompt', '')
    settings = generation_info.get('settings', {})

    confirmation_request = {
        'prompt': prompt,
        'image_id': image_id,
        'confirmed': True,
        'resolution': settings.get('resolution', '1K'),
        'cfg_weight': settings.get('cfg_weight', 7.0),
        'temperature': settings.get('temperature', 0.8),
        'num_images': settings.get('num_images', 1)
    }

    with app.test_request_context(json=confirmation_request):
        return api_generate()


if __name__ == '__main__':
    print(f"Python version: {os.sys.version}")
    os.makedirs('models', exist_ok=True)
    app.run(debug=False, host='0.0.0.0', port=5000)
