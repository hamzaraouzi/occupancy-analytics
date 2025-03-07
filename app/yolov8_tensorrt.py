import numpy as np
import cv2
import tensorrt as trt
import threading
import pycuda.driver as cuda
import pycuda.autoinit



class YOLOv8TensorRT:
    def __init__(self, engine_path, input_shape=(1, 3, 640, 640)):
        self.local_data = threading.local()
        self.local_data.logger = trt.Logger(trt.Logger.WARNING)
        self.local_data.runtime = trt.Runtime(self.logger)
        self.local_data.engine = self._load_trt(engine_path=engine_path)
        # Create execution context
        self.local_data.context = self.local_data.engine.create_execution_context()
        # Store input shape
        self.input_shape = input_shape
        # Get binding indices for input and output tensors
        self.input_binding_idx = self.local_data.engine.get_binding_index("images")  # Input tensor
        self.output_binding_idx = self.local_data.engine.get_binding_index("output0")  # Output tensor

        self.local_data.context.push()
        # Allocate GPU memory for input and output
        self.local_data.input_mem = cuda.mem_alloc(trt.volume(self.input_shape) * 4)  # Allocate memory for input tensor
        self.local_data.output_mem = cuda.mem_alloc(trt.volume((1, 84, 8400)) * 4)  # Allocate memory for output tensor (adjust based on model output size)
        self.local_data.stream = cuda.Stream()  # Create a CUDA stream for asynchronous execution

    def _load_trt(self, engine_path):
        with open(engine_path, "rb") as f:
            engine = self.local_data.runtime.deserialize_cuda_engine(f.read())
        return engine

    def preprocess(self, frame):
        img = cv2.resize(frame, self.input_shape)  # Resize the frame to the model's input size
        img = img[:, :, ::-1] / 255.0  # Convert BGR to RGB and normalize pixel values to [0,1]
        img = np.transpose(img, (2, 0, 1)) # Convert image to CHW format (Channels, Height, Width)
        img = np.expand_dims(img, axis=0)  # Add batch dimension
        return np.ascontiguousarray(img).astype(np.float32) 
    
        
    def postprocess(self, output, img_shape, conf_thresh=0.5, iou_thresh=0.5):
        detections = output[0]
        boxes = []
        confidences = []
        class_ids = []

        for det in detections:
            x, y, w, h, conf, cls = det
            if conf > conf_thresh:
                x1 = int((x - w / 2) * img_shape[1])  # Convert center-width to top-left x
                y1 = int((y - h / 2) * img_shape[0])  # Convert center-height to top-left y
                x2 = int((x + w / 2) * img_shape[1])  # Convert width to bottom-right x
                y2 = int((y + h / 2) * img_shape[0])  # Convert height to bottom-right y
                boxes.append([x1, y1, x2, y2])
                confidences.append(conf)
                class_ids.append(int(cls))

        # Apply Non-Maximum Suppression (NMS) to remove overlapping boxes
        indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_thresh, iou_thresh)
        if len(indices) > 0:
            indices = indices.flatten()
            return np.array([[*boxes[i], confidences[i], class_ids[i]] for i in indices])
        return np.array([])

    def infer(self, frame):
        img = self.preprocess(frame)
        # Copy preprocessed image to GPU memory
        cuda.memcpy_htod_async(self.local_data.input_mem, img, self.stream)
        # Execute inference asynchronously
        self.local_data.context.execute_async_v2(bindings=[int(self.local_data.input_mem),
                                                int(self.local_data.output_mem)],
                                      stream_handle=self.local_data.stream.handle)
        
        # Retrieve inference results from GPU memory
        output = np.empty((1, 84, 8400), dtype=np.float32)  # Adjust based on model output size
        cuda.memcpy_dtoh_async(output, self.local_data.output_mem, self.local_data.stream)  # Copy results from GPU to CPU memory
        self.local_data.stream.synchronize()  # Synchronize CUDA stream to ensure completion
        # self.postprocess(output, frame.shape)
        return output