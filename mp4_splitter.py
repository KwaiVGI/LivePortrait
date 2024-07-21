import cv2

def split_mp4(input_file):
    cap = cv2.VideoCapture(input_file)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    frame_interval = 15
    max_clips = 10
    clip_count = 0
    frame_number = 0

    while clip_count < max_clips and frame_number < frame_count:
        out_file = f'output_clip_{clip_count + 1}.mp4'
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(out_file, fourcc, fps, (int(cap.get(3)), int(cap.get(4))))

        for _ in range(frame_interval):
            ret, frame = cap.read()
            if not ret:
                break
            out.write(frame)
            frame_number += 1

        out.release()
        clip_count += 1

    cap.release()

# 使用例
split_mp4('D:\LivePortrait/assets\examples\source\s20.mp4')
