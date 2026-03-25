import torch
import time


print(f"PyTorch Version: {torch.__version__}")
cuda_availability = torch.cuda.is_available()
print(f"Cuda available: {cuda_availability}")

if cuda_availability:
    
    print(f"Current device: {torch.cuda.current_device()}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Device capability: {torch.cuda.get_device_capability(0)}")
    prop = torch.cuda.get_device_properties(0)
    print(f"Architecture: {prop.major}.{prop.minor}")
    print(f"Processors (SM): {prop.multi_processor_count}")
    print(f"Total memory: {prop.total_memory / 1024**2:.2f} MB")
    
    
    
    print("\n Testing matrix multiplication via GPU...")
    size = 5000
    a = torch.randn(size, size, device='cuda')
    b = torch.randn(size, size, device='cuda')
    
    start_time = time.time()
    
    c = torch.matmul(a, b)
    
    torch.cuda.synchronize() 
    end_time = time.time()
    
    print(f"Test finished")
    print(f"Execution time: {end_time - start_time:.4f} seconds.")
else:
    print("\n [Warning] PyTorch is not detecting any GPU. Check your drivers and try again. If you do not count with a cuda compatible GPU then that might be the problem")