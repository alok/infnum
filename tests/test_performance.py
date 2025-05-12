import time
from infnum import ε

def benchmark_operations():
    # Create test numbers
    x = sum(2 * ε**i for i in range(100))
    y = sum(3 * ε**i for i in range(100))
    
    # Benchmark addition
    start = time.time()
    for _ in range(1000):
        _ = x + y
    add_time = time.time() - start
    
    # Benchmark multiplication
    start = time.time()
    for _ in range(1000):
        _ = x * y
    mul_time = time.time() - start
    
    # Benchmark inversion
    start = time.time()
    for _ in range(100):
        _ = ~x
    inv_time = time.time() - start
    
    print(f"Addition time: {add_time:.3f}s")
    print(f"Multiplication time: {mul_time:.3f}s")
    print(f"Inversion time: {inv_time:.3f}s")

if __name__ == "__main__":
    benchmark_operations() 