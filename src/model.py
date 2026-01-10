import torch
import torch.nn as nn
import math

class LoRALayer(nn.Module):
    """
    Implements the 'LoRA fine tuning' block 
    Math: W_new = W_old + (B @ A) * scaling
    """

    def __init__(self, original_layer, rank=16, alpha=32, dropout=0.05):
        super().__init__()

        # We keep the original pre-trained layer but freeze it to save memory
        # It will not be updated during training, because we only train the LoRA parameters
        self.original_layer = original_layer
        self.original_layer.weight.requires_grad = False

        # Get dimensions: d_in (input size) and d_out (output size)
        # d_out is the number of rows, d_in is the number of columns in the weight matrix
        d_in = original_layer.in_features
        d_out = original_layer.out_features

        # To compress data (d_in -> r) and expand it back (r -> d_out)
        # Matrix A: Compresses (d_in x r)
        # Matrix B: Expands (r x d_out)
        self.lora_a = nn.Parameter(torch.zeros(d_in, rank)) # Matrix A
        self.lora_b = nn.Parameter(torch.zeros(rank, d_out)) # Matrix B

        # This constant controls how much influence the new path has
        # We're dividng alpha by rank to keep the scale consistent
        self.scaling = alpha / rank

        # Dropout (For regularization, standard in LoRA)
        # It prevents overfitting by randomly zeroing some inputs during training
        self.dropout = nn.Dropout(p=dropout)

        # We start with A=Random and B=Zero.
        # This ensures the starting output is (Original + 0), so we don't break the model.
        self.reset_parameters()

    def reset_parameters(self):
        # Initialize A with Kaiming Uniform (Gaussian noise)
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))
        # Initialize B with Zeros
        nn.init.zeros_(self.lora_b)

    def forward(self, x):
        # x shape: [Batch, Sequence_Length, d_in]
        
        # Path 1: The Original Frozen Weights
        # Equation: y_1 = W * x
        original_output = self.original_layer(x)
        
        # Path 2: The LoRA Path (A -> B)
        # Equation: y_2 = (x * A) * B
        # We apply dropout first, then multiply by A, then B
        # And we cast input 'x' to match LoRA weights, Float16 -> Float32
        x_dropped = self.dropout(x)
        low_rank_output = (x_dropped.to(self.lora_a.dtype) @ self.lora_a) @ self.lora_b   

        # Combine: y = y_1 + (y_2 * scaling)
        # This matches the "Element-wise addition" circle in Figure 2
        # And we cast back to original output dtype to avoid issues, Float32 -> Float16
        return original_output + (low_rank_output.to(original_output.dtype) * self.scaling)