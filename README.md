# Tensorflow_Object_Detection
Tensorflow를 사용한 객체 인식 프로그램

## 프로젝트 개요
Raspberry Pi, Arduino, Google Coral TPU와 웹캠 을 이용하여 실시간으로  “사람(person)” 을 인식하여 사용자에게 알림과 이벤트를 저장하여 주는 인공지능 홈 CCTV 프로젝트

[프로젝트 실행 동영상](https://www.youtube.com/watch?v=WfvgOQHzNRg)

## 사용 기술
- Raspberry Pi
- Google Coral TPU
- Tensorflow
- Arduino

라즈베리파이의 부족한 연산 능력을 저전력의 외장 TPU로 빠른 병렬 연산을 가능케해 실시간 처리에 성능 향상을 위하여 Google에서 개발한 외장형 Coral TPU를 이용하여 실시간으로 데이터를 처리한다.

모델 사용 : MobileNet_SSD_V2(COCO) 객체 추적 모델 사용


![image](https://github.com/ejeonghun/Tensorflow_Object_Detection/assets/41509711/ff4ed329-65f7-418c-ba22-5cc366ace460)

[포스터 원본](https://github.com/ejeonghun/Tensorflow_Object_Detection/files/15291630/2001481_._.-2.pdf)

[pptx 발표본](https://github.com/user-attachments/files/15945019/2001481_._.pptx)
