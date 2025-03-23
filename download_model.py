import os
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from janus.models import MultiModalityCausalLM, VLChatProcessor

def download_model(model_name="Janus-Pro-1B", output_path=None, use_auth_token=None, confirm=False):
    """
    Download a Janus model from Hugging Face.
    
    Args:
        model_name: Name of the model to download (Janus-Pro-1B, Janus-Pro-7B, Janus-1.3B, or JanusFlow-1.3B)
        output_path: Local directory to save the model (defaults to model name)
        use_auth_token: HuggingFace token for downloading gated models
        confirm: Whether to proceed with download or just show information
    """
    # Map model name to HF path
    model_map = {
        "Janus-Pro-1B": "deepseek-ai/Janus-Pro-1B",
        "Janus-Pro-7B": "deepseek-ai/Janus-Pro-7B",
        "Janus-1.3B": "deepseek-ai/Janus-1.3B",
        "JanusFlow-1.3B": "deepseek-ai/JanusFlow-1.3B"
    }
    
    if model_name not in model_map:
        raise ValueError(f"Unknown model: {model_name}. Choose from: {', '.join(model_map.keys())}")
    
    hf_model_path = model_map[model_name]
    
    # Set output path
    if output_path is None:
        output_path = model_name.lower().replace("-", "_")
    
    print(f"[INFO] Model: {model_name}")
    print(f"[INFO] HuggingFace path: {hf_model_path}")
    print(f"[INFO] Will be saved to: {output_path}")
    
    # If confirm is False, just show information and exit
    if not confirm:
        print("\n[INFO] This is a dry run. To download the model, add --confirm flag.")
        return None
    
    # Create output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)
    
    # Download processor
    print("[INFO] Downloading processor...")
    processor = VLChatProcessor.from_pretrained(hf_model_path, use_auth_token=use_auth_token)
    processor.save_pretrained(output_path)
    print("[INFO] Processor downloaded and saved!")
    
    # Download model
    print("[INFO] Downloading model (this may take a while)...")
    model = AutoModelForCausalLM.from_pretrained(
        hf_model_path, 
        trust_remote_code=True,
        local_files_only=False,
        use_auth_token=use_auth_token
    )
    model.save_pretrained(output_path)
    print("[INFO] Model downloaded and saved!")
    
    print(f"[INFO] Successfully downloaded {model_name} to {output_path}")
    return output_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download a Janus model from Hugging Face")
    parser.add_argument(
        "--model", 
        type=str, 
        default="Janus-Pro-1B",
        choices=["Janus-Pro-1B", "Janus-Pro-7B", "Janus-1.3B", "JanusFlow-1.3B"],
        help="Model to download"
    )
    parser.add_argument(
        "--output", 
        type=str, 
        help="Output directory (defaults to lowercase model name with underscores)"
    )
    parser.add_argument(
        "--token", 
        type=str, 
        help="HuggingFace token for downloading gated models"
    )
    parser.add_argument(
        "--confirm", 
        action="store_true",
        help="Confirm and proceed with download"
    )
    
    args = parser.parse_args()
    download_model(args.model, args.output, args.token, args.confirm) 