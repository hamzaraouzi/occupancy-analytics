import numpy as np
import cv2
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit


class YOLOv8TensorRT:
    def __init__(self, engine_path, input_shape=(640, 640)):
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        self.engine = self._load_trt(engine_path=engine_path)
        # Create execution context
        self.context = self.engine.create_execution_context()
        # Store input shape
        self.input_shape = input_shape
        # Get binding indices for input and output tensors
        self.input_binding_idx = self.engine.get_binding_index("images")  # Input tensor
        self.output_binding_idx = self.engine.get_binding_index("output0")  # Output tensor

        # Allocate GPU memory for input and output
        self.input_mem = cuda.mem_alloc(trt.volume(self.input_shape) * 2)  # Allocate memory for input tensor
        self.output_mem = cuda.mem_alloc(trt.volume((1, 25200, 6)) * 2)  # Allocate memory for output tensor (adjust based on model output size)
        self.stream = cuda.Stream()  # Create a CUDA stream for asynchronous execution

    def _load_trt(self, engine_path):
        with open(engine_path, "rb") as f:
            engine = self.runtime.deserialize_cuda_engine(f.read())
        return engine

    def preprocess(self, frame):
        img = cv2.resize(frame, self.input_shape)  # Resize the frame to the model's input size
        img = img[:, :, ::-1] / 255.0  # Convert BGR to RGB and normalize pixel values to [0,1]
        img = np.transpose(img, (2, 0, 1)).astype(np.float16)  # Convert image to CHW format (Channels, Height, Width)
        img = np.expand_dims(img, axis=0)  # Add batch dimension
        return img

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
        cuda.memcpy_htod_async(self.input_mem, img, self.stream)
        # Execute inference asynchronously
        self.context.execute_async_v2(bindings=[int(self.input_mem),
                                                int(self.output_mem)], 
                                      stream_handle=self.stream.handle)
        
        # Retrieve inference results from GPU memory
        output = np.empty((1, 25200, 6), dtype=np.float32)  # Adjust based on model output size
        cuda.memcpy_dtoh_async(output, self.output_mem, self.stream)  # Copy results from GPU to CPU memory
        self.stream.synchronize()  # Synchronize CUDA stream to ensure completion
        return self.postprocess(output, frame.shape)