import cv2
import numpy as np
import time
from PIL import Image
from pycoral.adapters import common, detect
from pycoral.utils.edgetpu import make_interpreter
import os
import serial
import tkinter as tk

ser = serial.Serial('/dev/ttyACM0', 9600)

model = "mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite" # MobileNet SSD v2
label_path = "coco_labels.txt" # COCO labels
interpreter = make_interpreter(model) 
interpreter.allocate_tensors()

labels = {} # Label dictionary
box_colors = {} # Box color dictionary
last_seen = {} # Last seen dictionary
person_count = 0 # Person count
re_detection_interval = 5.0 # Re-detection interval
save_path = "videos" # 녹화 저장 위치
os.makedirs(save_path, exist_ok=True)

with open(label_path, 'r') as f: # Load labels
    lines = f.readlines()
    for line in lines:
        id, name = line.strip().split(maxsplit=1)
        labels[int(id)] = name # label과 id를 dictionary에 저장
# 인식된 객체는 id 형태로 output 되는데 이를 label로 변환하기 위해 dictionary를 사용

# cap = cv2.VideoCapture("sample_exported.mp4") # 영상을 프레임 단위로 읽어옴 <- 예제 영상
cap = cv2.VideoCapture(-1) # 웹캠을 사용하여 영상을 프레임 단위로 읽어옴
threshold = 0.5 # 0.5 이상의 확률로 인식된 객체만 표시
input_size = common.input_size(interpreter) # 모델의 입력 크기
fourcc = cv2.VideoWriter_fourcc(*'mp4v') # 녹화 포맷
video_writer = None
recording_start_time = None


# 이벤트 리셋 GUI
def reset_event(): # Reset event count
    global person_count
    person_count = 0
    ser.write('G'.encode())  # 아두이노에 "G" 커맨드 전달
    update_label()

def update_label():
    event_label.config(text=f"Event: {person_count}")

root = tk.Tk()
root.title("HOME CCTV")
reset_button = tk.Button(root, text="Reset Event", command=reset_event)
reset_button.pack()
event_label = tk.Label(root, text=f"Event: {person_count}")
event_label.pack()

# Main loop for object detection
while True:
    root.update()
    ret, frame = cap.read()
    if not ret:
        print("cannot read frame.")
        break

    current_time = time.strftime('%Y-%m-%d %H:%M:%S') # 현재 시간
    (text_width, text_height), _ = cv2.getTextSize(current_time, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1) # 현재 시간 텍스트 크기
    cv2.putText(frame, current_time, (frame.shape[1] - text_width - 10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA) # 현재 시간 텍스트 추가
    cv2.putText(frame, f"Event Count: {person_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2) # 이벤트 카운트 텍스트 추가

    # 이미지 전처리
    img = frame[:, :, ::-1].copy() # BGR to RGB
    img_pil = Image.fromarray(img) # PIL Image로 변환
    img_pil = img_pil.resize(input_size, Image.Resampling.LANCZOS) # 입력 크기로 리사이즈
    common.set_input(interpreter, img_pil) # 입력 이미지 설정

    interpreter.invoke() # 추론 실행
    objs = detect.get_objects(interpreter, threshold) # 객체 검출
    filtered_objs = [obj for obj in objs if obj.score >= threshold] # 확률이 threshold 이상인 객체만 필터링

    detected_person = False # 사람이 감지되었는지 여부
    for obj in filtered_objs: # 필터링된 객체에 대해 반복
        if labels[obj.id] == 'person': # 객체가 사람일 경우
            detected_person = True # 사람이 감지되었음을 표시
            current_time = time.time()
            last_detected = last_seen.get(obj.id, 0) # 마지막으로 감지된 시간
            if current_time - last_detected > re_detection_interval: # re_detection_interval 이상 감지되지 않은 경우
                if video_writer is None: # 녹화 중이 아닌 경우
                    video_file_path = f"{save_path}/capture_{int(current_time)}.mp4"
                    video_writer = cv2.VideoWriter(video_file_path, fourcc, 20.0, (frame.shape[1], frame.shape[0]))
                    recording_start_time = current_time
                    person_count += 1 # 이벤트 카운트 증가
                    update_label() # 이벤트 카운트 업데이트
                last_seen[obj.id] = current_time # 마지막으로 감지된 시간 업데이트

            bbox = obj.bbox # 객체의 바운딩 박스 정보
            scale_x, scale_y = frame.shape[1] / input_size[0], frame.shape[0] / input_size[1] # 이미지 크기 비율
            xmin, ymin, xmax, ymax = max(0, int(bbox.xmin * scale_x)), max(0, int(bbox.ymin * scale_y)), min(frame.shape[1], int(bbox.xmax * scale_x)), min(frame.shape[0], int(bbox.ymax * scale_y)) # 바운딩 박스 좌표
            box_color = box_colors.get(obj.id, [int(j) for j in np.random.randint(0, 255, 3)]) # 객체별로 색상 지정
            box_colors[obj.id] = box_color # 색상 저장
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), box_color, 2) # 바운딩 박스 그리기
            label_text = f"{labels[obj.id]}: {int(obj.score * 100)}%" # 레이블와 퍼센트 텍스트
            cv2.putText(frame, label_text, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2) # 레이블 텍스트 추가

    if detected_person:
        ser.write('R'.encode())  # Red LED 녹화중/사람 객체 인식 시
    elif person_count > 0:
        ser.write('O'.encode())  # Orange LED 이전에 이벤트가 있을 시
    else:
        ser.write('G'.encode())  # Green LED 이벤트 없음

    if video_writer and (time.time() - recording_start_time > re_detection_interval): # re_detection_interval 이상 감지되지 않은 경우
        video_writer.release()
        video_writer = None
        print(f"Saved video to {video_file_path}")

    if video_writer:
        video_writer.write(frame)

    cv2.imshow('Home CCTV', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): # 'q' 키를 누르면 종료
        break

if video_writer: # 녹화 중지
    video_writer.release() # 녹화 파일 저장
    

ser.write('X'.encode()) # 종료 시 LED 초기화
cap.release()
cv2.destroyAllWindows()
root.destroy()
