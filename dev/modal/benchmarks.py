from pathlib import Path

import modal

ROOT_PATH = Path(__file__).parent.parent.parent
REMOTE_ROOT_PATH = "/root/liger-kernel"
PYTHON_VERSION = "3.12"

image = modal.Image.debian_slim(python_version=PYTHON_VERSION).pip_install("uv")

app = modal.App("liger_benchmarks", image=image)

# mount: add local files to the remote container
repo = image.add_local_dir(ROOT_PATH, remote_path=REMOTE_ROOT_PATH)


@app.function(gpu="H100!", image=repo, timeout=60 * 90)
def liger_benchmarks():
    import os
    import subprocess

    subprocess.run(
        ["uv pip install -e '.[dev]' --system"],
        check=True,
        shell=True,
        cwd=REMOTE_ROOT_PATH,
    )
    subprocess.run(["make run-benchmarks"], check=True, shell=True, cwd=REMOTE_ROOT_PATH)

    file_path = Path(REMOTE_ROOT_PATH) / "benchmark" / "data" / "all_benchmark_data.csv"
    print(f"Checking if file exists at: {file_path}")
    print(f"File exists: {os.path.exists(file_path)}")

    if not os.path.exists(file_path):
        print("Listing directory contents:")
        data_dir = file_path.parent
        if os.path.exists(data_dir):
            print(f"Contents of {data_dir}:")
            print(os.listdir(data_dir))
        else:
            print(f"Data directory {data_dir} does not exist")
        raise FileNotFoundError(f"Benchmark data file not found at {file_path}")

    with open(file_path, "rb") as f:
        data = f.read()
        print(f"Successfully read {len(data)} bytes of data")
        return data


@app.local_entrypoint()
def main():
    try:
        # Run the benchmarks and get the data
        print("Starting benchmark run...")
        benchmark_data = liger_benchmarks.remote()

        if not benchmark_data:
            raise ValueError("No data received from remote function")

        # Save the data locally
        local_data_path = ROOT_PATH / "benchmark" / "data" / "all_benchmark_data.csv"
        print(f"Attempting to save data to: {local_data_path}")

        local_data_path.parent.mkdir(parents=True, exist_ok=True)

        with open(local_data_path, "wb") as f:
            f.write(benchmark_data)

        print(f"Successfully saved {len(benchmark_data)} bytes to: {local_data_path}")

    except Exception as e:
        print(f"Error occurred: {str(e)}")
        raise
