# model.py
import tensorrt as trt
import pycuda.driver as cuda
import numpy as np
import threading
import cv2

class YOLOv8TensorRT:
    def __init__(self, engine_path):
        self.engine_path = engine_path
        self.local_data = threading.local()
        self.input_shape = (1, 3, 640, 640)  # Update for your model
        self.output_shape = (1, 84, 8400)     # Update for your model

    def load_model(self):
        """Initialize CUDA context and TensorRT resources for this thread"""
        if not hasattr(self.local_data, 'is_initialized'):
            try:
                # Initialize CUDA
                cuda.init()
                self.local_data.device = cuda.Device(0)
                self.local_data.cuda_context = self.local_data.device.make_context()

                # Load TensorRT engine
                with open(self.engine_path, "rb") as f:
                    runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
                    self.local_data.engine = runtime.deserialize_cuda_engine(f.read())

                # Create execution context
                self.local_data.trt_context = self.local_data.engine.create_execution_context()

                # Allocate memory
                self._allocate_memory()

                # Create CUDA stream
                self.local_data.stream = cuda.Stream()

                self.local_data.is_initialized = True
            except Exception as e:
                print(f"Failed to initialize model: {e}")
                raise

    def _allocate_memory(self):
        """Allocate GPU memory for input/output tensors"""
        input_size = trt.volume(self.input_shape) * np.dtype(np.float32).itemsize
        output_size = trt.volume(self.output_shape) * np.dtype(np.float32).itemsize

        self.local_data.input_mem = cuda.mem_alloc(input_size)
        self.local_data.output_mem = cuda.mem_alloc(output_size)

    def infer(self, frame):
        """Perform inference on a frame"""
        # Ensure model is initialized
        if not hasattr(self.local_data, 'is_initialized'):
            self.load_model()  # Auto-initialize if not already done

        try:
            self.local_data.cuda_context.push()
            # Preprocess and copy to GPU
            self.preprocess(frame, self.local_data.input_mem)
            
            # Run inference
            self.local_data.trt_context.execute_async_v2(
                bindings=[int(self.local_data.input_mem), int(self.local_data.output_mem)],
                stream_handle=self.local_data.stream.handle
            )
            
            # Copy output from GPU
            output = self._get_output()
            return output
        finally:
            self.local_data.cuda_context.pop()

    def preprocess(self, frame, input_mem):
        """Prepare input tensor and copy to GPU"""
        # Example preprocessing
        resized = cv2.resize(frame, (self.input_shape[3], self.input_shape[2]))
        normalized = (resized / 255.0).astype(np.float32)
        chw = normalized.transpose(2, 0, 1)  # HWC to CHW
        host_input = np.ascontiguousarray(chw)
        cuda.memcpy_htod_async(input_mem, host_input, self.local_data.stream)

    def _get_output(self):
        """Retrieve and postprocess output"""
        host_output = np.empty(self.output_shape, dtype=np.float32)
        cuda.memcpy_dtoh_async(host_output, self.local_data.output_mem, self.local_data.stream)
        self.local_data.stream.synchronize()
        return host_output

    def __del__(self):
        """Clean up resources"""
        if hasattr(self.local_data, 'cuda_context'):
            self.local_data.cuda_context.pop()