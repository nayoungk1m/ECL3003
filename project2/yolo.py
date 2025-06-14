from ultralytics import YOLO
import cv2
import Camera.jetsonCam as jetCam
import time

model = YOLO("models/250610_n_detection.engine")
cam = jetCam.jetsonCam()
cam.open(sensor_id=0,
          sensor_mode=3,
          flip_method=0,
          display_height=540,
          display_width=960)
cam.start()


while True:
    ret, frame = cam.read()
    if frame is None:
        break
    start_time = time.time()
    results = model(frame)
    print(f"Detection time: {time.time() - start_time:.2f} seconds")
    # Visualize detection results
    annotated_frame = results[0].plot()

    # Display the frame
    cv2.imshow('YOLO Detection', annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cam.stop()
cam.release()
cv2.destroyAllWindows()
