import time
import torch
import torchvision.models as models

# Measure inference latency on CPU
device = torch.device('cpu')
model = models.resnet18()
model.fc = torch.nn.Linear(model.fc.in_features, 5)
model.eval()

dummy_image = torch.randn(1, 3, 224, 224)

# Warmup
for _ in range(5):
    _ = model(dummy_image)

# Benchmark 50 runs
times = []
with torch.no_grad():
    for _ in range(50):
        t0 = time.perf_counter()
        _ = model(dummy_image)
        times.append((time.perf_counter() - t0) * 1000)

avg_time = sum(times) / len(times)
print(f"Average CPU Inference Latency per image: {avg_time:.2f} ms")
print(f"Throughput: {1000 / avg_time:.1f} frames/sec")
