import tyro
from src.config.argument_config import ArgumentConfig
from src.config.inference_config import InferenceConfig
from src.config.crop_config import CropConfig
from src.live_portrait_pipeline import LivePortraitPipeline

import cv2
import time
import numpy as np

def partial_fields(target_class, kwargs):
    return target_class(**{k: v for k, v in kwargs.items() if hasattr(target_class, k)})

def main():
    # set tyro theme
    tyro.extras.set_accent_color("bright_cyan")
    args = tyro.cli(ArgumentConfig)

    # specify configs for inference
    inference_cfg = partial_fields(InferenceConfig, args.__dict__)
    crop_cfg = partial_fields(CropConfig, args.__dict__)

    live_portrait_pipeline = LivePortraitPipeline(
        inference_cfg=inference_cfg,
        crop_cfg=crop_cfg
    )

    # Initialize webcam 'assets/examples/driving/d6.mp4'
    cap = cv2.VideoCapture(0)

    # Process the first frame to initialize
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture image")
        return

    source_image_path = args.source_image  # Set the source image path here
    x_s, f_s, R_s, x_s_info, lip_delta_before_animation, crop_info, img_rgb = live_portrait_pipeline.execute_frame(frame, source_image_path)

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        if not ret:
            break

        # Process the frame
        
        result = live_portrait_pipeline.generate_frame(x_s, f_s, R_s, x_s_info, lip_delta_before_animation, crop_info, img_rgb, frame)
        cv2.imshow('img_rgb Image', img_rgb)
        cv2.imshow('Source Frame', frame)
        

        # [Key Change] Convert the result from RGB to BGR before displaying
        result_bgr = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)


        # Display the resulting frame
        cv2.imshow('Live Portrait', result_bgr)

        # Press 'q' to exit the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything is done, release the capture
    cap.release()
    cv2.destroyAllWindows()
    
    # live_portrait_pipeline.execute_frame(result_bgr)


if __name__ == '__main__':
    st = time.time()
    main()
    print("Generation time:", (time.time() - st) * 1000)

# 3. Reduced webcam latency 350 to 160

# import cv2
# import time
# import threading
# import numpy as np
# import tyro
# from src.config.argument_config import ArgumentConfig
# from src.config.inference_config import InferenceConfig
# from src.config.crop_config import CropConfig
# from src.live_portrait_pipeline import LivePortraitPipeline

# def partial_fields(target_class, kwargs):
#     return target_class(**{k: v for k, v in kwargs.items() if hasattr(target_class, k)})

# class VideoCaptureThread:
#     def __init__(self, src=0):
#         self.cap = cv2.VideoCapture(src)
#         self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
#         self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
#         self.cap.set(cv2.CAP_PROP_FPS, 60)
        
#         if not self.cap.isOpened():
#             print("Failed to open camera")
#             self.running = False
#         else:
#             self.ret = False
#             self.frame = None
#             self.running = True
#             self.thread = threading.Thread(target=self.update, args=())
#             self.thread.start()
    
#     def update(self):
#         while self.running:
#             self.ret, self.frame = self.cap.read()
#             if not self.ret:
#                 print("Failed to read frame")
#                 break
    
#     def read(self):
#         return self.ret, self.frame
    
#     def release(self):
#         self.running = False
#         self.thread.join()
#         self.cap.release()

# def main():
#     # Set tyro theme
#     tyro.extras.set_accent_color("bright_cyan")
#     args = tyro.cli(ArgumentConfig)

#     # Specify configs for inference
#     inference_cfg = partial_fields(InferenceConfig, args.__dict__)
#     crop_cfg = partial_fields(CropConfig, args.__dict__)

#     live_portrait_pipeline = LivePortraitPipeline(
#         inference_cfg=inference_cfg,
#         crop_cfg=crop_cfg
#     )

#     # Initialize webcam 'assets/examples/driving/d6.mp4'
#     cap_thread = VideoCaptureThread(0)

#     # Wait for the first frame to be captured
#     while not cap_thread.ret and cap_thread.running:
#         time.sleep(0.1)

#     if not cap_thread.ret:
#         print("Failed to capture image")
#         cap_thread.release()
#         return

#     source_image_path = args.source_image  # Set the source image path here
#     ret, frame = cap_thread.read()
#     x_s, f_s, R_s, x_s_info, lip_delta_before_animation, crop_info, img_rgb = live_portrait_pipeline.execute_frame(frame, source_image_path)

#     while cap_thread.running:
#         # Capture frame-by-frame
#         ret, frame = cap_thread.read()
#         if not ret:
#             break

#         # Process the frame
#         result = live_portrait_pipeline.generate_frame(x_s, f_s, R_s, x_s_info, lip_delta_before_animation, crop_info, img_rgb, frame)
#         # cv2.imshow('img_rgb Image', img_rgb)
#         cv2.imshow('Webcam Frame', frame)
        
#         # Convert the result from RGB to BGR before displaying
#         result_bgr = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
        
#         # Display the resulting frame
#         cv2.imshow('Webcam Live Portrait', result_bgr)

#         # Press 'q' to exit the loop
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#     # When everything is done, release the capture
#     cap_thread.release()
#     cv2.destroyAllWindows()

# if __name__ == '__main__':
#     st = time.time()
#     main()
#     print("Generation time:", (time.time() - st) * 1000)
