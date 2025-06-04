import cv2
import numpy as np
from matplotlib import pyplot as plt
from StereoCameraCalibrate.Stereo_Calib_Camera import stereoCalibrateCamera
from StereoCameraCalibrate.Stereo_Calib_Camera import getStereoCameraParameters, getStereoSingleCameraParameters
import signal
import sys
import Camera.jetsonCam as jetCam 

def signal_handler(sig, frame):
  print('\nExiting...')
  cv2.destroyAllWindows()
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
#stereoCalibrateCamera(cam1,cam2,'jetson_stereo_8MP',24)
lod_data = getStereoCameraParameters('jetson_stereo_8MP.npz')
lod_datac1 = getStereoSingleCameraParameters('jetson_stereo_8MPc1.npz')
lod_datac2 = getStereoSingleCameraParameters('jetson_stereo_8MPc2.npz')

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
  depth = (f_x * B) / (disparity + 1e-5)
  return depth

def calculate_bbox_depth(disparity, bbox):
  """
  바운딩 박스 내의 평균 깊이 계산 함수
  :param disparity: disparity map
  :param bbox: 바운딩 박스 (x, y, width, height)
  :return: 평균 깊이
  """
  x, y, w, h = bbox
  w, h = w // 2, h // 2  # 바운딩 박스 크기를 절반으로 조정
  box_disparity = disparity[y-h:y+h, x-w:x+w]
  box_disparity = box_disparity[box_disparity > 0] # -16 제거
  average_depth = np.mean(calculate_depth(box_disparity))
  return average_depth

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
stereo = cv2.StereoBM_create(numDisparities=num_disp, blockSize=block_s)


# Load rectification maps
map1_left, map2_left = cv2.initUndistortRectifyMap( camera_matrix_left, dist_coeffs_left, R1, P1, image_size, cv2.CV_16SC2)
map1_right, map2_right = cv2.initUndistortRectifyMap(camera_matrix_right, dist_coeffs_right, R2, P2, image_size, cv2.CV_16SC2)

while True:
  # Read stereo images
  ret, image_left = cam1.read()
  ret, image_right = cam2.read()

  # Remap the images using rectification maps
  rectified_left = cv2.remap(image_left, map1_left, map2_left, cv2.INTER_LINEAR)
  rectified_right = cv2.remap(image_right, map1_right, map2_right, cv2.INTER_LINEAR)

  # Convert images to grayscale
  gray_left = cv2.cvtColor(rectified_left, cv2.COLOR_BGR2GRAY)
  gray_right = cv2.cvtColor(rectified_right, cv2.COLOR_BGR2GRAY)

  # Compute disparity map
  disparity = stereo.compute(gray_left, gray_right)

  # Calculate depth map
  depth_map = calculate_depth(disparity.astype(np.float32))
  print(f"Depth map shape: {depth_map.shape}")  
  bbox = [480, 270, 10, 10] # Example bounding box (x, y, width, height)
  x, y, w, h = bbox
  average_depth = calculate_bbox_depth(disparity, bbox)
  print(f"Average depth in bounding box {bbox}: {average_depth:.2f} meters")

  # Normalize the disparity map to the range [0, 1]
  normalized_disparity_map = cv2.normalize(disparity, None, 0.0, 1.0, cv2.NORM_MINMAX,cv2.CV_32F)

  colormap_image = cv2.applyColorMap(np.uint8(normalized_disparity_map * 255), cv2.COLORMAP_JET)

  # Display rectangle
  cv2.rectangle(colormap_image,
                (x - w // 2, y - h // 2),
                (x + w // 2, y + h // 2),
                (0, 255, 0), 2)  # Blue color with thickness 2
  
  cv2.imshow('Depth color map', colormap_image)

  k=cv2.waitKey(33)
  if k == ord('x'):
    break
  elif k == ord('q'):
    #increase block size
    block_s += 2
    print("block_size:"+str(block_s))
    stereo.setBlockSize(block_s)

  elif k == ord('a'):
    #decrease block size
    block_s = max(block_s-2,5)
    print("block_size:"+str(block_s))
    stereo.setBlockSize(block_s)

  elif k == ord('w'):
    #increase disparity
    num_disp += 16
    print("disparity:"+str(num_disp))
    stereo.setNumDisparities(num_disp)

  elif k == ord('s'):
    #decrease disparity
    num_disp = max(16, num_disp-16)
    print("disparity:"+str(num_disp))
    stereo.setNumDisparities(num_disp)

cv2.destroyAllWindows()
cam1.stop()
cam2.stop()
cam1.release()
cam2.release()


