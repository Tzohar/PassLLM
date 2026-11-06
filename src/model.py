import torch
import torch.nn as nn

class LoRALayer(nn.Module):
    """
    Implements the LoRA adapter layer for inference.
    Math: W_new = W_old + (B @ A) * scaling
    """

    def __init__(self, original_layer, rank=16, alpha=32):
        super().__init__()

        self.original_layer = original_layer
        self.original_layer.weight.requires_grad = False

        # Get dimensions: d_in (input size) and d_out (output size)
        d_in = original_layer.in_features
        d_out = original_layer.out_features

        # Matrix A: Compresses (d_in x r)
        # Matrix B: Expands (r x d_out)
        self.lora_a = nn.Parameter(torch.zeros(d_in, rank), requires_grad=False)
        self.lora_b = nn.Parameter(torch.zeros(rank, d_out), requires_grad=False)

        # Scaling factor
        self.scaling = alpha / rank

    def forward(self, x):
        # x shape: [Batch, Sequence_Length, d_in]
        
        # Path 1: The Original Weights
        original_output = self.original_layer(x)
        
        # Path 2: The LoRA Path (A -> B)
        low_rank_output = (x.to(self.lora_a.dtype) @ self.lora_a) @ self.lora_b   

        # Combine: y = y_1 + (y_2 * scaling)
        return original_output + (low_rank_output.to(original_output.dtype) * self.scaling)