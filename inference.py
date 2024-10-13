import torch
import torch.multiprocessing as mp
from torch.multiprocessing import Queue
import os
from tqdm import tqdm
import argparse

def worker(gpu_id, input_queue, output_queue, model_path):
    # Set the device for this process
    device = torch.device(f'cuda:{gpu_id}')

    # Load the model onto the assigned GPU
    model = torch.jit.load(model_path).to(device)
    # model.eval()  # Set model to evaluation mode

    while True:
        try:
            # Get the next input from the queue
            input_data = input_queue.get(timeout=5)
            if input_data is None:
                # Exit signal received
                break

            # Move input data to the GPU
            input_data = input_data.to(device)

            # Process the input data
            with torch.no_grad():
                output_data = model(input_data)
                # Move output back to CPU if needed
                output_data = output_data.cpu()

            # Put the result into the output queue
            output_queue.put(output_data)

        except Exception as e:
            print(f"Exception in worker {os.getpid()}: {e}")
            break

def parse_arguments():
    parser = argparse.ArgumentParser(description="Process inputs using a TorchScript model with multiprocessing.")

    # Add arguments
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the TorchScript model file.')
    parser.add_argument('--input_path', type=str, required=True,
                        help='Path to the input data file or directory.')
    parser.add_argument('--num_procs_per_gpu', type=int, default=2,
                        help='Number of processes to run per GPU.')
    parser.add_argument('--num_gpus', type=int, default=None,
                        help='Number of GPUs to use. Default uses all available GPUs.')
    parser.add_argument('--output_path', type=str, default='results.pt',
                        help='Path to save the output results.')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Number of samples per batch for processing.')
    # You can add more arguments as needed

    args = parser.parse_args()
    return args

def main():
    args = parse_arguments()

    # Set the multiprocessing start method to 'spawn' for CUDA compatibility
    mp.set_start_method('spawn', force=True)

    # Number of GPUs available
    total_gpus = torch.cuda.device_count()
    if args.num_gpus is None:
        num_gpus = total_gpus
    else:
        num_gpus = min(args.num_gpus, total_gpus)
        if num_gpus < args.num_gpus:
            print(f"Requested {args.num_gpus} GPUs, but only {total_gpus} are available. Using {num_gpus} GPUs.")

    # Number of processes per GPU
    num_procs_per_gpu = args.num_procs_per_gpu

    # Total number of processes
    total_procs = num_gpus * num_procs_per_gpu

    # Path to the TorchScript model
    model_path = args.model_path

    # Load or prepare the inputs
    inputs = load_inputs(args.input_path, args.batch_size)

    # Create the input and output queues
    input_queue = mp.Queue()
    output_queue = mp.Queue()

    # Start the worker processes
    processes = []
    for gpu_idx in range(num_gpus):
        for _ in range(num_procs_per_gpu):
            p = mp.Process(target=worker, args=(gpu_idx, input_queue, output_queue, model_path))
            p.start()
            processes.append(p)

    # Enqueue all inputs
    for input_data in inputs:
        input_queue.put(input_data)

    # Send exit signals to the workers
    for _ in processes:
        input_queue.put(None)

    # Collect the outputs with a progress bar
    results = []
    num_inputs = len(inputs)

    print("Processing inputs...")
    with tqdm(total=num_inputs) as pbar:
        for _ in range(num_inputs):
            output_data = output_queue.get()
            results.append(output_data)
            pbar.update(1)

    # Wait for all processes to finish
    for p in processes:
        p.join()

    # Save the results
    save_outputs(results, args.output_path)

    print("Processing complete. Results saved to:", args.output_path)

def load_inputs(input_path, batch_size):
    """
    Load or generate input data from the given path.

    Args:
        input_path (str): Path to the input data file or directory.
        batch_size (int): Number of samples per batch.

    Returns:
        list: A list of input tensors.
    """
    # Example: Load inputs from a file or directory
    # Replace this with actual data loading logic
    inputs = []

    # For demonstration, let's assume input_path is a directory containing input files
    if os.path.isdir(input_path):
        file_list = sorted(os.listdir(input_path))
        for file_name in file_list:
            file_path = os.path.join(input_path, file_name)
            # Load the data (this depends on your data format)
            data = torch.load(file_path)
            inputs.append(data)
    elif os.path.isfile(input_path):
        # Load data from a single file
        data = torch.load(input_path)
        # Split data into batches if necessary
        inputs = [data[i:i+batch_size] for i in range(0, len(data), batch_size)]
    else:
        raise FileNotFoundError(f"Input path '{input_path}' does not exist.")

    return inputs

def save_outputs(outputs, output_path):
    """
    Save the outputs to the specified path.

    Args:
        outputs (list): List of output tensors.
        output_path (str): Path to save the outputs.
    """
    # Example: Save outputs as a single file
    torch.save(outputs, output_path)

if __name__ == '__main__':
    main()
