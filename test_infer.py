import torch
import torch.jit
import threading
from queue import Queue
from tqdm import tqdm
import time

# Define a simple model to test the functionality
class SimpleModel(torch.nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.linear = torch.nn.Linear(10, 10)
        
    def forward(self, x):
        # Simulate computation time by performing additional computations
        for _ in range(10):
            x = self.linear(x)
        return x

# Worker thread class
class Worker(threading.Thread):
    def __init__(self, device_id, worker_id, model_path, input_queue, output_queue):
        threading.Thread.__init__(self)
        self.device_id = device_id
        self.worker_id = worker_id
        self.input_queue = input_queue
        self.output_queue = output_queue
        # Load the model instance on the assigned device
        device = torch.device(f'cuda:{device_id}' if torch.cuda.is_available() else 'cpu')
        self.model = torch.jit.load(model_path).to(device)
        self.model.eval()
        self.device = device
        self.daemon = True  # Ensures threads exit when the main thread exits
        
    def run(self):
        while True:
            input_data = self.input_queue.get()
            if input_data is None:
                # Signal to terminate the thread
                self.input_queue.task_done()
                break
            input_id, input_tensor = input_data
            # Move input to the assigned device
            input_tensor = input_tensor.to(self.device)
            
            # Use torch.jit.fork to process the input asynchronously
            fut = torch.jit.fork(self.model.forward, input_tensor)
            
            # Wait for the result without blocking the thread
            output = torch.jit.wait(fut)
            
            # Move output back to CPU
            output = output.cpu()
            
            # Put the result into the output queue
            self.output_queue.put((input_id, output))
            
            self.input_queue.task_done()

# Processing Manager class
class ProcessingManager:
    def __init__(self, model_path, num_devices, threads_per_device):
        self.model_path = model_path
        self.num_devices = num_devices if num_devices > 0 else 1
        self.threads_per_device = threads_per_device
        self.input_queue = Queue()
        self.output_queue = Queue()
        self.workers = []
        self.results = {}
        self.num_inputs = 0

    def start_workers(self):
        for device_id in range(self.num_devices):
            for n in range(self.threads_per_device):
                worker_id = device_id * self.threads_per_device + n
                worker = Worker(device_id, worker_id, self.model_path, self.input_queue, self.output_queue)
                worker.start()
                self.workers.append(worker)

    def add_inputs(self, inputs_list):
        self.num_inputs = len(inputs_list)
        # Add inputs to the input queue
        for input_data in inputs_list:
            self.input_queue.put(input_data)
        # Add termination signals to the input queue
        for _ in self.workers:
            self.input_queue.put(None)

    def process_inputs(self):
        num_processed = 0
        with tqdm(total=self.num_inputs) as pbar:
            while num_processed < self.num_inputs:
                output_data = self.output_queue.get()
                input_id, output = output_data
                self.results[input_id] = output
                num_processed += 1
                pbar.update(1)
        # Wait for all tasks to be processed
        self.input_queue.join()
        # Wait for all worker threads to finish
        for worker in self.workers:
            worker.join()

    def get_results(self):
        return self.results

def main():
    # Define model path
    model_path = 'scripted_model.pt'

    # Convert the model to TorchScript and save it
    scripted_model = torch.jit.script(SimpleModel())
    torch.jit.save(scripted_model, model_path)

    # Number of devices (GPUs) and processes per device
    D = torch.cuda.device_count()
    N = 2  # Adjust based on your preference

    print(f'Number of devices: {D}')
    print(f'Number of threads per device: {N}')

    # Prepare the test inputs
    num_inputs = 10000  # Total number of inputs to process

    # Create dummy inputs for testing
    inputs_list = [(i, torch.randn(10)) for i in range(num_inputs)]

    # Initialize the processing manager
    manager = ProcessingManager(model_path, D, N)
    # Start worker threads
    manager.start_workers()
    # Add inputs to the queue
    manager.add_inputs(inputs_list)
    # Process inputs and collect outputs
    manager.process_inputs()
    # Get results
    results = manager.get_results()

    print("Processing complete.")
    # 'results' now contains the outputs, indexed by input_id
    # You can access the outputs as needed
    # For example, print the outputs
    for idx in sorted(results.keys()):
        print(f"Input ID: {idx}, Output: {results[idx]}")

if __name__ == '__main__':
    main()
