import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
import tensorflow_hub as hub
import cv2


# MoveNet 모델 로드
def load_model():
    model = hub.load("https://tfhub.dev/google/movenet/singlepose/lightning/4")
    movenet = model.signatures['serving_default']
    return movenet

# 이미지 불러오기 For MoveNet
def process_image(image_path):
    image = tf.io.read_file(image_path)
    image = tf.compat.v1.image.decode_jpeg(image)
    if image.shape[2]==1:
        image = tf.image.grayscale_to_rgb(image)
    return image

# 키포인트 추출
def run_movenet(movenet, frame):
    # 이미지 전처리
    frame = tf.image.resize_with_pad(tf.expand_dims(frame, axis=0), 192, 192)
    frame = tf.cast(frame, dtype=tf.int32)

    # MoveNet 모델 실행
    results = movenet(frame)

    # 결과 반환 (키포인트 좌표 및 신뢰도)
    keypoints = results['output_0'].numpy()
    return keypoints[0] # keypoints[1]: scale

def extract_tensor(model, video_path):
    # MP4 영상 읽기
    cap = cv2.VideoCapture(video_path)
    tensor = []
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 5 프레임마다 한 번씩만 MoveNet 실행
        if frame_count % 5 == 0:
            keypoints = run_movenet(model, frame)
            tensor.append(keypoints)

        frame_count += 1

    cap.release()


    tensor_array = np.squeeze(np.array(tensor), axis=1)
    return tensor_array

def flatten_tensor(tensor):
    n = tensor.shape[0]
    flattened_tensor = tensor.reshape(n, -1)  # 결과는 (n, 51) 형태의 배열
    return flattened_tensor