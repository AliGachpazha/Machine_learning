import cv2

# آدرس RTSP یا MJPEG را اینجا قرار بده. اگر نمی‌دانی، باید مدل دوربین را بفرستی.
url = 'rtsp://20.20.20.58/stream1'

cap = cv2.VideoCapture(url)

if not cap.isOpened():
    print("❌ دوربین متصل نشد.")
else:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ دریافت تصویر شکست خورد.")
            break

        cv2.imshow('Live Camera', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
