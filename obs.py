import cv2
import time

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

    # ビデオライターの設定
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(f'output_{frame_count}.mp4', fourcc, 30.0, (int(cap.get(3)), int(cap.get(4))))

    ret, frame = cap.read()
    if not ret:
        print("フレームを取得できませんでした")
        break

    # フレームをウィンドウに表示
    cv2.imshow('OBS Virtual Camera', frame)

    frames.append(frame)  # フレームをリストに追加

    if len(frames) >= frame_batch_size:
        frame_count += 1

        # フレームをビデオファイルに書き込む
        for frame in frames:
            out.write(frame)

        print(f"キャプチャを保存しました: output_{frame_count}.mp4")
        frames = []  # フレームリストをリセット

    # 'q'キーが押されたらループを終了
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# リソースを解放
cap.release()
out.release()
cv2.destroyAllWindows()
