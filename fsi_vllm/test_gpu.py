#!/usr/bin/env python3
"""
Test GPU detection for MI300X
"""

try:
    import torch
    print(f"PyTorch version: {torch.__version__}")
    
    if torch.cuda.is_available():
        print(f"✅ CUDA/ROCm available: {torch.cuda.is_available()}")
        print(f"GPU count: {torch.cuda.device_count()}")
        
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            props = torch.cuda.get_device_properties(i)
            print(f"  Memory: {props.total_memory / (1024**3):.1f} GB")
            print(f"  Compute capability: {props.major}.{props.minor}")
        
        # Test tensor creation on GPU
        try:
            x = torch.randn(1000, 1000).cuda()
            y = torch.randn(1000, 1000).cuda()
            z = torch.mm(x, y)
            print("✅ GPU tensor operations working")
        except Exception as e:
            print(f"❌ GPU tensor operations failed: {e}")
    else:
        print("❌ No GPU detected")
        
except ImportError as e:
    print(f"❌ PyTorch not available: {e}")
