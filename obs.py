import cv2
import time
import pickle

# 仮想カメラのデバイスID（通常は0または1）
camera_id = 1

# カメラを開く
cap = cv2.VideoCapture(camera_id)

# フレームレートを設定
cap.set(cv2.CAP_PROP_FPS, 30)

if not cap.isOpened():
    print("カメラを開けませんでした")
    exit()

frame_count = 0
capture_interval = 3  # キャプチャ間隔（秒）
last_capture_time = time.time()
frames = []  # フレームを保存するリスト
frame_batch_size = 15  # バッチサイズ

while True:
    ret, frame = cap.read()
    if not ret:
        print("フレームを取得できませんでした")
        break

    # フレームをウィンドウに表示
    cv2.imshow('OBS Virtual Camera', frame)

    frames.append(frame)  # フレームをリストに追加

    if len(frames) >= frame_batch_size:
        frame_count += 1
        filename = f"{frame_count}.pkl"

        # フレームを指定された形式のdict型でpklファイルに保存
        frame_dict = {
            'n_frames': len(frames),
            'output_fps': 30,
            'motion': [{'scale': frame.mean(axis=(0, 1)), 'R_d': frame.mean(axis=(0, 1)), 'exp': frame.mean(axis=(0, 1)), 't': frame.mean(axis=(0, 1))} for frame in frames],
            'c_d_eyes_lst': [frame.mean(axis=(0, 1)) for frame in frames],
            'c_d_lip_lst': [frame.mean(axis=(0, 1)) for frame in frames]
        }
        with open(filename, 'wb') as f:
            pickle.dump(frame_dict, f)

        print(f"キャプチャを保存しました: {filename}")
        frames = []  # フレームリストをリセット

    # 'q'キーが押されたらループを終了
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# リソースを解放
cap.release()
cv2.destroyAllWindows()
