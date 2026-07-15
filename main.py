import marimo

__generated_with = "0.23.14"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # PyTorch Quantization
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Basic Setup
    """)
    return


@app.cell
def _():
    import os
    import time
    import warnings
    import numpy as np
    import torch
    import torch.nn as nn
    from torchvision import datasets, transforms, models
    from torch.utils.data import DataLoader, Subset

    # ignores irrelevant warning, see: https://github.com/pytorch/pytorch/issues/149829
    warnings.filterwarnings(
        "ignore",
        message=".*TF32 acceleration on top of oneDNN is available for Intel GPUs. The current Torch version does not have Intel GPU Support.*",
    )

    # ignores irrelevant warning, see: https://github.com/tensorflow/tensorflow/issues/77293
    warnings.filterwarnings("ignore", message=".*erase_node(.*) on an already erased node.*")

    print(f"PyTorch Version: {torch.__version__}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device used: {device.type}")

    skip_cpu = False  # change to True to skip the slow checks on CPU
    print(f"Should skip CPU evaluations: {skip_cpu}")
    return (
        DataLoader,
        Subset,
        datasets,
        device,
        models,
        nn,
        np,
        os,
        skip_cpu,
        time,
        torch,
        transforms,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Get CIFAR-10 train and test sets
    """)
    return


@app.cell
def _(DataLoader, Subset, datasets, transforms):
    transform = transforms.Compose(
        [transforms.Resize(32), transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )

    train_dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
    train_loader = DataLoader(
        datasets.CIFAR10(root="./data", train=True, download=True, transform=transform),
        batch_size=128,
        shuffle=True,
        num_workers=0,
    )

    test_dataset = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)
    test_loader = DataLoader(
        datasets.CIFAR10(root="./data", train=False, download=True, transform=transform),
        batch_size=128,
        shuffle=False,
        num_workers=0,
        drop_last=True,
    )

    calibration_dataset = Subset(train_dataset, range(256))
    calibration_loader = DataLoader(calibration_dataset, batch_size=128, shuffle=False)
    return calibration_loader, test_loader, train_loader


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Adjust ResNet18 network for CIFAR-10 dataset
    """)
    return


@app.cell
def _(device, models, nn):
    def get_resnet18_for_cifar10():
        """
        Returns a ResNet-18 model adjusted for CIFAR-10:
        - 3x3 conv with stride 1
        - No max pooling
        - 10 output classes
        """
        model = models.resnet18(weights=None, num_classes=10)
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()

        return model.to(device)

    return (get_resnet18_for_cifar10,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Define Train and Evaluate Functions
    """)
    return


@app.cell
def _(device, nn, os, test_loader, torch):
    def train(model, loader, epochs, lr=0.01, save_path="model.pth", silent=False):
        """
        Trains a model with SGD and cross-entropy loss.
        Loads from save_path if it exists.
        """

        try:
            model.train()
        except NotImplementedError:
            torch.ao.quantization.move_exported_model_to_train(model)

        if os.path.exists(save_path):
            if not silent:
                print(f"Model already trained. Loading from {save_path}")
            model.load_state_dict(torch.load(save_path), strict=True)
            return

        # no saved model found. training from given model state

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)

        for epoch in range(epochs):
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                loss = criterion(model(x), y)
                loss.backward()
                optimizer.step()
            if not silent:
                print(f"Epoch {epoch + 1}: loss={loss.item():.4f}")
                evaluate(model, f"Epoch {epoch + 1}")
                try:
                    model.train()
                except NotImplementedError:
                    torch.ao.quantization.move_exported_model_to_train(model)

        if save_path:
            torch.save(model.state_dict(), save_path)
            if not silent:
                print(f"Training complete. Model saved to {save_path}")

    def evaluate(model, tag):
        """
        Evaluates the model on test_loader and prints accuracy.
        """

        try:
            model.eval()
        except NotImplementedError:
            model = torch.ao.quantization.move_exported_model_to_eval(model)

        model.to(device)
        correct = total = 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                preds = model(x).argmax(1)
                correct += (preds == y).sum().item()
                total += y.size(0)
        accuracy = correct / total
        print(f"Accuracy ({tag}): {accuracy * 100:.2f}%")

    return evaluate, train


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Define helper functions to measure latency
    """)
    return


@app.cell
def _(np, os, time, torch):
    class Timer:
        """
        A simple timer utility for measuring elapsed time in milliseconds.

        Supports both GPU and CPU timing:
        - If CUDA is available, uses torch.cuda.Event for accurate GPU timing.
        - Otherwise, falls back to wall-clock CPU timing via time.time().

        Methods:
            start(): Start the timer.
            stop(): Stop the timer and return the elapsed time in milliseconds.
        """

        def __init__(self):
            self.use_cuda = torch.cuda.is_available()
            if self.use_cuda:
                self.starter = torch.cuda.Event(enable_timing=True)
                self.ender = torch.cuda.Event(enable_timing=True)

        def start(self):
            if self.use_cuda:
                self.starter.record()
            else:
                self.start_time = time.time()

        def stop(self):
            if self.use_cuda:
                self.ender.record()
                torch.cuda.synchronize()
                return self.starter.elapsed_time(self.ender)  # ms
            else:
                return (time.time() - self.start_time) * 1000  # ms

    def estimate_latency(model, example_inputs, repetitions=50):
        """
        Returns avg and std inference latency (ms) over given runs.
        """

        timer = Timer()
        timings = np.zeros((repetitions, 1))

        # warm-up
        for _ in range(5):
            _ = model(example_inputs)

        with torch.no_grad():
            for rep in range(repetitions):
                timer.start()
                _ = model(example_inputs)
                elapsed = timer.stop()
                timings[rep] = elapsed

        return np.mean(timings), np.std(timings)

    def estimate_latency_full(model, tag, skip_cpu):
        """
        Prints model latency on GPU and (optionally) CPU.
        """

        # estimate latency on CPU
        if not skip_cpu:
            example_input = torch.rand(128, 3, 32, 32).cpu()
            model.cpu()
            latency_mu, latency_std = estimate_latency(model, example_input)
            print(f"Latency ({tag}, on CPU): {latency_mu:.2f} ± {latency_std:.2f} ms")

        # estimate latency on GPU
        example_input = torch.rand(128, 3, 32, 32).cuda()
        model.cuda()
        latency_mu, latency_std = estimate_latency(model, example_input)
        print(f"Latency ({tag}, on GPU): {latency_mu:.2f} ± {latency_std:.2f} ms")

    def print_size_of_model(model, tag=""):
        """
        Prints model size (MB).
        """

        torch.save(model.state_dict(), "temp.p")
        size_mb_full = os.path.getsize("temp.p") / 1e6
        print(f"Size ({tag}): {size_mb_full:.2f} MB")
        os.remove("temp.p")

    return estimate_latency_full, print_size_of_model


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Train full model
    """)
    return


@app.cell
def _():
    import subprocess

    subprocess.run(["mkdir", "-p", "models"], capture_output=True, text=True)
    return


@app.cell
def _(get_resnet18_for_cifar10, train, train_loader):
    model_to_quantize = get_resnet18_for_cifar10()
    train(model_to_quantize, train_loader, epochs=15, save_path="./models/full_model.pth")
    return (model_to_quantize,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Evaluate full model
    """)
    return


@app.cell
def _(
    estimate_latency_full,
    evaluate,
    model_to_quantize,
    print_size_of_model,
    skip_cpu,
):
    print_size_of_model(model_to_quantize, "full")

    # evaluate full accuracy
    evaluate(model_to_quantize, "full")

    # estimate full model latency
    estimate_latency_full(model_to_quantize, "full", skip_cpu)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Post Training Quantization (PTQ)

    The basic flow is as follow:

    1. Export the model to to a stable, backend-agnostic format that’s suitable for transformations, optimizations, and deployment.
    2. Define the quantizer that will prepare the model for quantization. Here I used the X86 for CPU deployments, but there is a simple variant that works better for mobile and edge devices working on ARM CPUs.
    3. Preparing the model for quantization. For example, folding batch-norm into preceding conv2d operators, and inserting observers in appropriate places to collect activation statistics needed for calibration.
    4. Running inference on calibration data to collect activation statistics
    5. Converts calibrated model to a quantized model. While the quantized model already takes less space, it is not yet optimized for the final deployment.
    """)
    return


@app.cell
def _(calibration_loader, device, model_to_quantize, torch):
    from packaging import version

    from torch.ao.quantization.quantize_pt2e import (
        prepare_pt2e,
        convert_pt2e,
    )

    import torch.ao.quantization.quantizer.x86_inductor_quantizer as xiq
    from torch.ao.quantization.quantizer.x86_inductor_quantizer import X86InductorQuantizer

    # batch of 128 images, each with 3 color channels and 32x32 resolution (CIFAR-10)
    example_inputs = (torch.rand(128, 3, 32, 32).to(device),)

    # export the model to a standardized format before quantization
    if version.parse(torch.__version__) >= version.parse("2.5"):  # for pytorch 2.5+
        exported_model = torch.export.export_for_training(
            model_to_quantize, example_inputs
        ).module()
    else:  # for pytorch 2.4
        from torch._export import capture_pre_autograd_graph

        exported_model = capture_pre_autograd_graph(model_to_quantize, example_inputs)

    # quantization setup for X86 Inductor Quantizer
    quantizer = X86InductorQuantizer()
    quantizer.set_global(xiq.get_default_x86_inductor_quantization_config())

    # preparing for PTQ by folding batch-norm into preceding conv2d operators, and inserting observers in appropriate places
    prepared_model = prepare_pt2e(exported_model, quantizer)

    # run inference on calibration data to collect activation stats needed for activation quantization
    def calibrate(model, data_loader):
        torch.ao.quantization.move_exported_model_to_eval(model)
        with torch.no_grad():
            for image, target in data_loader:
                model(image.to(device))

    calibrate(prepared_model, calibration_loader)

    # converts calibrated model to a quantized model
    quantized_model = convert_pt2e(prepared_model)

    # export again to remove unused weights after quantization
    if version.parse(torch.__version__) >= version.parse("2.5"):  # for pytorch 2.5+
        quantized_model = torch.export.export_for_training(quantized_model, example_inputs).module()
    else:  # for pytorch 2.4
        quantized_model = capture_pre_autograd_graph(quantized_model, example_inputs)
    return (quantized_model,)


@app.cell
def _(
    estimate_latency_full,
    evaluate,
    print_size_of_model,
    quantized_model,
    skip_cpu,
):
    # get quantized model size
    print_size_of_model(quantized_model, "quantized")

    # evaluate quantized accuracy
    evaluate(quantized_model, "quantized")

    # estimate quantized model latency
    estimate_latency_full(quantized_model, "quantized", skip_cpu)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Compile PTQ Model
    """)
    return


@app.cell
def _(os, quantized_model, torch):
    # 1. Tell the C++ compiler where to find the conda-packaged CUDA headers
    if "CONDA_PREFIX" in os.environ:
        cuda_include = os.path.join(os.environ["CONDA_PREFIX"], "targets/x86_64-linux/include")
        os.environ["CPLUS_INCLUDE_PATH"] = cuda_include

    # 2. Tell the linker where to find libcuda.so in WSL
    wsl_lib_path = "/usr/lib/wsl/lib"
    if os.path.exists(wsl_lib_path):
        existing_lib_path = os.environ.get("LIBRARY_PATH", "")
        os.environ["LIBRARY_PATH"] = (
            f"{wsl_lib_path}:{existing_lib_path}" if existing_lib_path else wsl_lib_path
        )

    # enable the use of the C++ wrapper for TorchInductor which reduces Python overhead
    import torch._inductor.config as config

    config.cpp_wrapper = True

    # compiles quantized model to generate optimized model
    with torch.no_grad():
        optimized_model = torch.compile(quantized_model)
    return (optimized_model,)


@app.cell
def _(
    estimate_latency_full,
    evaluate,
    optimized_model,
    print_size_of_model,
    skip_cpu,
):
    # get optimized model size
    print_size_of_model(optimized_model, "optimized")

    # evaluate optimized accuracy
    evaluate(optimized_model, "optimized")

    # estimate optimized model latency
    estimate_latency_full(optimized_model, "optimized", skip_cpu)
    return


if __name__ == "__main__":
    app.run()
