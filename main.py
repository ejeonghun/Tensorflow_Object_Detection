import cv2
import numpy as np
import time
from PIL import Image
from pycoral.adapters import common, detect
from pycoral.utils.edgetpu import make_interpreter
import os
import serial
import tkinter as tk
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders

# 아두이노 시리얼 통신 설정
ser = serial.Serial('/dev/ttyACM0', 9600) 

# 이메일 설정 및 전송 함수 (사용자에게 맞는 설정 필요)
def send_email(file_path):
    receiver_email = "wjdgns4019@gmail.com"  # 수신자 이메일 주소

    sender_email = "##########"  # 네이버 발신 이메일
    sender_password = "###########"  # 네이버 발신 이메일 비밀번호

    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = receiver_email
    msg['Subject'] = "[경고] (Home CCTV)사람이 인식되었습니다."  # 메일 제목

    body = "현재 사람이 인식된 이벤트가 발생하였습니다!"  # 메일 본문
    msg.attach(MIMEText(body, 'plain'))

    # 첨부 파일 설정
    attachment = open(file_path, "rb")
    p = MIMEBase('application', 'octet-stream')
    p.set_payload((attachment).read())
    encoders.encode_base64(p)
    p.add_header('Content-Disposition', "attachment; filename= %s" % file_path)
    msg.attach(p)

    # 동영상 스냅샷 촬영 및 첨부
    video_capture = cv2.VideoCapture(file_path)
    ret, frame = video_capture.read()
    video_capture.release()

    # 첨부할 스냅샷 이미지 경로 설정
    snapshot_path = f"{os.path.splitext(file_path)[0]}.jpg"
    cv2.imwrite(snapshot_path, frame)

    # 스냅샷 파일 첨부
    with open(snapshot_path, 'rb') as f:
        attachment = MIMEBase('application', 'octet-stream')
        attachment.set_payload(f.read())
        encoders.encode_base64(attachment)
        attachment.add_header('Content-Disposition', f'attachment; filename={os.path.basename(snapshot_path)}')
        msg.attach(attachment)

    # SMTP 세션 생성 및 메일 전송
    server = smtplib.SMTP_SSL('smtp.naver.com', 465)  # Naver SMTP 서버 주소 및 포트 (SSL 사용)
    server.login(sender_email, sender_password)
    server.sendmail(sender_email, receiver_email, msg.as_string())
    server.quit()

    print(f"Email sent to {receiver_email} with attachment {file_path}")

    # 스냅샷 파일 삭제
    os.remove(snapshot_path)



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
        send_email(video_file_path)

    if video_writer: # 녹화 중인 경우
        video_writer.write(frame) # 프레임 저장

    cv2.imshow('Home CCTV', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): # 'q' 키를 누르면 종료
        break

if video_writer: # 녹화 중지
    video_writer.release() # 녹화 파일 저장
    

ser.write('X'.encode()) # 종료 시 LED 초기화
cap.release()
cv2.destroyAllWindows()
root.destroy()
