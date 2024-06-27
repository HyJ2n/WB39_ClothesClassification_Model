from ultralytics import YOLO

def main():
    # YOLOv8 모델 경로를 설정합니다.
    model = YOLO('yolov8x.pt')

    # 모델 학습을 시작합니다.
    model.train(
        data='data.yaml',  # data.yaml 파일 경로를 설정합니다.     
        epochs=50, 
        batch=16, 
        lrf=0.01,
        workers=0,         
        patience=0         
    )

if __name__ == '__main__':
    main()