import cv2
import numpy as np
from ultralytics import YOLO
import time
from StereoCameraCalibrate.Stereo_Calib_Camera import getStereoCameraParameters, getStereoSingleCameraParameters
import signal
import sys
import Camera.jetsonCam as jetCam 

FRAME_INTERVAL = 10
running = True
frame_counter = -1

def signal_handler(sig, frame):
    print('\nExiting...')
    cv2.destroyAllWindows()
    print('OpenCV: destroyed all windows')
    print('CUDA: releasing GPU resources')
    # GPU 자원 해제
    gpu_mat_l.release()
    gpu_mat_r.release()
    gpu_disparity.release()
    stream.waitForCompletion()
    cv2.cuda.resetDevice()

    # 카메라 자원 해제
    cam1.stop()
    cam2.stop()
    cam1.release()
    cam2.release()
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

cam1 = jetCam.jetsonCam()
cam2 = jetCam.jetsonCam()

#left camera 
cam1.open(sensor_id=1,
          sensor_mode=3,
          flip_method=0,
          display_height=540,
          display_width=960,
        )
#right camera
cam2.open(sensor_id=0,
          sensor_mode=3,
          flip_method=0,
          display_height=540,
          display_width=960,
        )

cam1.start()
cam2.start()

lod_data = getStereoCameraParameters('stereo_depth_estimation/jetson_stereo_8MP.npz')
lod_datac1 = getStereoSingleCameraParameters('stereo_depth_estimation/jetson_stereo_8MPc1.npz')
lod_datac2 = getStereoSingleCameraParameters('stereo_depth_estimation/jetson_stereo_8MPc2.npz')

print(lod_datac1[0])
print(lod_datac2[0])
print(lod_datac1[1])
print(lod_datac2[1])

camera_matrix_left = lod_data[0]
dist_coeffs_left =  lod_data[1]
camera_matrix_right =  lod_data[2]
dist_coeffs_right =  lod_data[3]
R =  lod_data[4]
T =  lod_data[5]
# print camera matrix
print("RAW camera matrix")
print(camera_matrix_right)
print(camera_matrix_left)

# 카메라 파라미터 설정
f_x = camera_matrix_left[0, 0]  # 왼쪽 카메라 초점 거리
f_y = camera_matrix_left[1, 1]  # 왼쪽 카메라 초점 거리
c_x = camera_matrix_left[0, 2]  # 왼쪽 카메라 주점 x 좌표
c_y = camera_matrix_left[1, 2]  # 왼쪽 카메라 주점 y 좌표
# 두 카메라 간의 기준 거리(B) (임의 값 설정)
B = 0.4  # 기준 거리, 예시로 10cm 설정 (두 카메라 사이의 거리)

def calculate_depth(disparity):
    """
    깊이 계산 함수
    :param disparity: disparity map
    :return: depth map
    """
    # 0으로 나누는 것을 방지하기 위해 작은 값 추가
    # 깊이 계산
    disparity = disparity.astype(np.float32) * 16.0
    depth = (f_x * B) / (disparity + 1e-5)
    return depth

def calculate_bbox_depth(disparity, bbox):
    """
    바운딩 박스 내의 평균 깊이 계산 함수
    :param disparity: disparity map
    :param bbox: 바운딩 박스 (x, y, width, height)
    :return: 평균 깊이
    """
    x, y, w, h = map(int, bbox)
    w, h = w // 2, h // 2  # 바운딩 박스 크기를 절반으로 조정
    box_disparity = disparity[y - h //2 :y + h//2, x - w//2:x + w//2]
    box_disparity = box_disparity[box_disparity > 0] # 0 제거
    average_depth = np.mean(calculate_depth(box_disparity))
    return average_depth

def undistort_bbox(bbox_xywh):
    x, y, w, h = bbox_xywh
    # 네 꼭짓점 배열 (N,1,2) 형태
    pts = np.array([
        [x - w/2, y - h/2],
        [x + w/2, y - h/2],
        [x + w/2, y + h/2],
        [x - w/2, y + h/2]
    ], dtype=np.float32).reshape(-1,1,2)

    # 왜곡 제거 + 정렬 → P1 좌표계
    pts_rect = cv2.undistortPoints(
        pts,
        camera_matrix_left, dist_coeffs_left,
        R=R1, P=P1      # ← stereoRectify 에서 받은 행렬
    ).reshape(-1,2)

    # 새 박스 좌표 계산
    x_r, y_r = pts_rect[:,0].min(), pts_rect[:,1].min()
    w_r = pts_rect[:,0].max() - x_r
    h_r = pts_rect[:,1].max() - y_r
    return [int(x_r + w_r/2), int(y_r + h_r/2), int(w_r), int(h_r)]

ret,img_s = cam1.read()
if ret:
    image_size = (img_s.shape[1],img_s.shape[0])
    print(image_size)
  
else:
    cam1.stop()
    cam2.stop()
    cam1.release()
    cam2.release()
    exit()

#stereo rectify
R1, R2, P1,P2, Q, roi1, roi2= cv2.stereoRectify(camera_matrix_left,dist_coeffs_left, camera_matrix_right, dist_coeffs_right, image_size, R, T)

block_s = 17
num_disp= 64

# Create a StereoBM object
stereo = cv2.cuda.createStereoBM(numDisparities=num_disp, blockSize=block_s)

# Load rectification maps
map1_left, map2_left = cv2.initUndistortRectifyMap( camera_matrix_left, dist_coeffs_left, R1, P1, image_size, cv2.CV_16SC2)
map1_right, map2_right = cv2.initUndistortRectifyMap(camera_matrix_right, dist_coeffs_right, R2, P2, image_size, cv2.CV_16SC2)

gpu_mat_l = cv2.cuda_GpuMat()
gpu_mat_r = cv2.cuda_GpuMat()

gpu_disparity = cv2.cuda_GpuMat()
stream = cv2.cuda_Stream()

def sync_bm_params(bm):
    bm.setPreFilterType(cv2.STEREO_BM_PREFILTER_XSOBEL)  # CPU 기본값과 맞춤
    bm.setTextureThreshold(10)
    bm.setUniquenessRatio(15)
    bm.setPreFilterCap(31)          # CPU·CUDA 동일
    bm.setSpeckleWindowSize(0)      # 필요하면 동일하게 조정
    bm.setSpeckleRange(0)
    return bm

# CPU
cpu_bm = cv2.StereoBM_create(num_disp, block_s)
sync_bm_params(cpu_bm)

# CUDA
gpu_bm = cv2.cuda.createStereoBM(num_disp, block_s)
sync_bm_params(gpu_bm)

model_yolo = YOLO("../models/250521_n_detection.engine", verbose=False)

while running:
# ================ perception ================ #
    frame_counter += 1
    
    # Read stereo images
    ret, image_left = cam1.read()
    ret, image_right = cam2.read()
    if not ret: continue # 카메라랑 동시에 맞도록...

    if frame_counter % FRAME_INTERVAL == 0:
        depth_estimation_start_time = time.time()
        # Remap the images using rectification maps
        rectified_left = cv2.remap(image_left, map1_left, map2_left, cv2.INTER_LINEAR)
        rectified_right = cv2.remap(image_right, map1_right, map2_right, cv2.INTER_LINEAR)

        gpu_mat_l.upload(rectified_left)
        gpu_mat_r.upload(rectified_right)

        # Convert images to grayscale
        gray_left = cv2.cuda.cvtColor(gpu_mat_l, cv2.COLOR_BGR2GRAY)
        gray_right = cv2.cuda.cvtColor(gpu_mat_r, cv2.COLOR_BGR2GRAY)

        gpu_disparity = stereo.compute(gray_left, gray_right, stream=stream)
        # Compute disparity map
        disparity = gpu_disparity.download()

        normalized_disparity_map = cv2.normalize(disparity, None, 0.0, 1.0, cv2.NORM_MINMAX,cv2.CV_32F)

        colormap_image = cv2.applyColorMap(np.uint8(normalized_disparity_map * 255), cv2.COLORMAP_JET)
        print(f"Depth estimation time: {time.time() - depth_estimation_start_time} sec")

    
    if frame_counter % FRAME_INTERVAL == 5:
        # =============================================
        # yolo = detect
        yolo_detect_start_time = time.time()
        result = model_yolo(image_left, verbose=False)
        classes = result[0].boxes.cls.to("cpu").tolist()
        boxes = result[0].boxes.xywh.to("cpu").tolist()
        scores = result[0].boxes.conf.to("cpu").tolist()
        print(f"YOLO Detect time: {time.time() - yolo_detect_start_time} sec")

        for cls, box, score in zip(classes, boxes, scores):
            if cls != 7.0:  # Except vehicle
                # Calculate bbox depth
                box_rect = undistort_bbox(box)
                average_depth = calculate_bbox_depth(disparity, box_rect)
                print(f"Average depth: {average_depth:.2f} meters")
                x, y, w, h = map(int, box)
                # Draw bounding box on the colormap image
                cv2.rectangle(colormap_image, (x - w // 2, y - h // 2), (x + w // 2, y + h // 2), (0, 255, 0), 2)
                cv2.putText(colormap_image, f"Depth: {average_depth:.2f} m", (x - w // 2, y - h // 2 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        detected_image = result[0].plot()
        if not isinstance(detected_image, np.ndarray):
            detected_image = np.array(detected_image)
    
    
        com_img=  cv2.hconcat([detected_image, colormap_image])
        com_img = cv2.resize(com_img, (0, 0), fx=0.6, fy=0.6)
        cv2.imshow('YOLO - depth map', com_img)
        cv2.waitKey(1)
