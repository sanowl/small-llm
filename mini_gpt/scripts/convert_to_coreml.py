# scripts/convert_to_coreml.py

import torch
import coremltools as ct
from models.mini_gpt import MiniGPT
from tokenizers.tokenizer import load_trained_tokenizer

def main():
    # Configuration
    model_path = 'outputs/models/best_mini_gpt.pth'
    tokenizer_path = 'outputs/tokenizer/'
    coreml_model_path = 'outputs/models/MiniGPT.mlmodel'
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    # Load tokenizer
    tokenizer = load_trained_tokenizer(tokenizer_path)
    vocab_size = len(tokenizer)
    
    # Initialize model
    model = MiniGPT(
        vocab_size=vocab_size,
        embed_size=512,
        num_heads=8,
        hidden_dim=2048,
        num_layers=12,
        max_seq_length=512,
        dropout=0.1
    ).to(device)
    
    # Load model weights
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # Create an example input
    example_input = torch.randint(0, vocab_size, (1, 512)).to(device)
    
    # Trace the model with TorchScript
    traced_model = torch.jit.trace(model, example_input)
    traced_model_path = 'outputs/models/mini_gpt_traced.pt'
    traced_model.save(traced_model_path)
    
    # Convert to Core ML
    coreml_model = ct.convert(
        traced_model,
        inputs=[ct.TensorType(shape=example_input.shape, name="input_ids")],
        outputs=[ct.TensorType(name="logits")]
    )
    
    # Save the Core ML model
    coreml_model.save(coreml_model_path)
    print(f"Core ML model saved to {coreml_model_path}")

if __name__ == "__main__":
    main()
