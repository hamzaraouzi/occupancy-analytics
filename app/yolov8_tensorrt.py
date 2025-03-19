# model.py
import tensorrt as trt
import pycuda.driver as cuda
import numpy as np
import threading
import cv2


class YOLOv8TensorRT:
    def __init__(self,
                 engine_path,
                 original_shape=(720, 1280),
                 input_shape=(1, 3, 640, 640),
                 conf_threshold=0.3,
                 iou_threshold=0.9,
                 num_classes=80):
        self.engine_path = engine_path
        self.local_data = threading.local()
        self.input_shape = input_shape
        self.output_shape = (1, num_classes+4, 8400)
        self.original_shape = original_shape
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold

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
                bindings=[int(self.local_data.input_mem),
                          int(self.local_data.output_mem)],
                stream_handle=self.local_data.stream.handle
            )

            # Copy output from GPU
            output = self._get_output()
            return self.decode_output(output)
            
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
        cuda.memcpy_dtoh_async(host_output, self.local_data.output_mem, 
                               self.local_data.stream)
        self.local_data.stream.synchronize()
        return host_output

    def decode_output(self, output_tensor):
        # Reshape the output tensor
        output = output_tensor.squeeze(0)  # Shape: (84, 8400)
        output = output.transpose(1, 0)    # Shape: (8400, 84)

        # Split into bounding boxes and class probabilities
        bbox_data = output[:, 0:4]    # (cx, cy, w, h) for 8400 anchors
        cls_scores = output[:, 4:84]  # Class probabilities (80 values per anchor)

        # Convert bounding boxes to (x1, y1, x2, y2)
        cx, cy, w, h = bbox_data[:, 0], bbox_data[:, 1], bbox_data[:, 2], bbox_data[:, 3]
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2
        boxes = np.stack([x1, y1, x2, y2], axis=1)

        # Get confidence scores and class IDs
        conf_scores = np.max(cls_scores, axis=1)  # Max class probability per anchor
        class_ids = np.argmax(cls_scores, axis=1)

        # Filter out low-confidence predictions
        keep = conf_scores > self.conf_threshold
        boxes = boxes[keep]
        conf_scores = conf_scores[keep]
        class_ids = class_ids[keep]

        # Apply Non-Maximum Suppression (NMS)
        nms_indices = cv2.dnn.NMSBoxes(
            bboxes=boxes.tolist(),
            scores=conf_scores.tolist(),
            score_threshold=self.conf_threshold,
            nms_threshold=self.iou_threshold,
            top_k=1000
        )

        # Get final detections
        final_boxes = boxes[nms_indices]
        final_scores = conf_scores[nms_indices]
        final_class_ids = class_ids[nms_indices]

        return self.scale_boxes(final_boxes), final_scores, final_class_ids

    def scale_boxes(self, boxes):
        # original_shape: (height, width) of the original image
        # model_input_shape: (height, width) used for inference
        height_ratio = self.original_shape[0] / self.input_shape[2]
        width_ratio = self.original_shape[1] / self.input_shape[3]
        
        boxes[:, [0, 2]] *= width_ratio   # Scale x coordinates
        boxes[:, [1, 3]] *= height_ratio  # Scale y coordinates
        return boxes.astype(int)  # Convert to integers

    def __del__(self):
        """Clean up resources"""
        if hasattr(self.local_data, 'cuda_context'):
            self.local_data.cuda_context.pop()
