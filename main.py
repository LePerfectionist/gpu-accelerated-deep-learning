import torch
from models.cnn import CNN
from utils.data_loader import get_dataloaders
from utils.train_eval import train_model, evaluate_model

def benchmark(device_label, device, epochs=5, data_batchsize=64):
    print(f"\nRunning on {device_label}...")
    model = CNN()
    train_loader, test_loader = get_dataloaders(data_batchsize)

    train_time = train_model(model, train_loader, device, epochs)

    accuracy = None
    if(epochs > 1):
        accuracy = evaluate_model(model, test_loader, device)
        print(f"{device_label} Training Time: {train_time:.2f} seconds")
        print(f"{device_label} Accuracy: {accuracy:.2f}%")

    return train_time, accuracy

if __name__ == "__main__":
    print("Torch CUDA available: ", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("Torch CUDA device name:", torch.cuda.get_device_name(0))

    if not torch.cuda.is_available():
        print("CUDA not available, benchmarking only on CPU.")

    cpu_device = torch.device("cpu")
    gpu_device = torch.device("cuda") if torch.cuda.is_available() else cpu_device

    results = []


    # Run benchmarks
    cpu_time, cpu_acc = benchmark("CPU", cpu_device, epochs=5, data_batchsize=64)
    if torch.cuda.is_available():
        gpu_time, gpu_acc = benchmark("GPU", gpu_device, epochs=5, data_batchsize=64)

    #Testing over different batchszies

    for batch_size in [32, 128, 256, 512]:
        for device in ["CPU", "GPU"]:

            training_time, _ = benchmark(
                device,
                cpu_device if device=="CPU" else gpu_device,
                epochs=1,
                data_batchsize=batch_size
            )
            results.append(
                {
                    'device': device,
                    'batch_size': batch_size,
                    'time' : training_time

                }
            )


    print("\nBenchmark Results (1 epoch):")
    print(f"{'Device':<6} | {'Batch Size':<10} | {'Time (s)':<8}")
    print("-" * 35)
    for r in results:
        print(f"{r['device']:<6} | {r['batch_size']:<10} | {r['time']:<8}")
