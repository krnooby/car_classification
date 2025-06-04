import cv2

cap = cv2.VideoCapture(0)  # 기본 웹캠

while True:
    ret, frame = cap.read()
    if not ret:
        break
    flipped = cv2.flip(frame, 1)  # 좌우반전
    cv2.imshow('Mirrored Webcam', flipped)

    if cv2.waitKey(1) & 0xFF == ord('q'):  # q 누르면 종료
        break

cap.release()
cv2.destroyAllWindows()
