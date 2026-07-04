# Vision Transformer built from scratch in PyTorch (Colab), trained on CIFAR-10. 

After reading "Attention Is All You Need" (Vaswani et al., 2017), I implemented the ViT architecture from scratch to understand the internals, how attention works and how tensor shapes transform as an image passes through the model. Trained from scratch on CIFAR-10, it reaches 84.4% test accuracy (d_model=256, 6 layers, 8 heads). This is below pretrained-ViT/ResNet numbers (~93%), which is expected: ViTs lack the spatial inductive bias of CNNs and are data-hungry, so a from-scratch ViT on a small dataset like CIFAR-10 trades accuracy for architectural transparency.

## Components implemented from scratch:
- Scaled dot-product attention — QKᵀ scaled by √dₖ before softmax, to prevent softmax saturation and keep gradients healthy.
- Multi-head attention — multiple heads attending to different learned relationships in parallel.
- Positional encoding — injects token-order information, since attention is otherwise permutation-invariant.
- Encoder block, patch embedding, and the full ViT.

## Validation:
My MultiheadAttention was verified against PyTorch's nn.MultiheadAttention, matching to a max difference of 1e-07 (floating-point equivalent).
