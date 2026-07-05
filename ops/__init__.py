# backbone.ops package
# Submodules (attn_interface, conv, hyena_se, hyena_x, embedding, etc.) require
# CUDA/Triton/flash-attn at import time. Import them lazily where needed
# rather than eagerly here, so `import backbone.ops` works on CPU-only envs.
