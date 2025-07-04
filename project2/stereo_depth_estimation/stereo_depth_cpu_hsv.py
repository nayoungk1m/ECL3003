import cv2
import numpy as np
from matplotlib import pyplot as plt
from StereoCameraCalibrate.Stereo_Calib_Camera import stereoCalibrateCamera
from StereoCameraCalibrate.Stereo_Calib_Camera import getStereoCameraParameters, getStereoSingleCameraParameters

import Camera.jetsonCam as jetCam


def draw_lines(img):
   for i in range(0,img.shape[0],30):
       cv2.line(img,(0,i),(img.shape[1],i),(255,0,0),1)
   return img   

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
#print camera matrix
print("RAW camera matrix")
print(camera_matrix_right)
print(camera_matrix_left)


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
R1,R2,P1,P2,Q,roi1,roi2= cv2.stereoRectify(camera_matrix_left,dist_coeffs_left, camera_matrix_right, dist_coeffs_right, image_size, R, T)

block_s = 17
num_disp= 64

# Create a StereoBM object
stereo = cv2.StereoBM_create(numDisparities=num_disp, blockSize=block_s)


# Load rectification maps
map1_left, map2_left = cv2.initUndistortRectifyMap( camera_matrix_left, dist_coeffs_left, R1, P1, image_size, cv2.CV_16SC2)
map1_right, map2_right = cv2.initUndistortRectifyMap(camera_matrix_right, dist_coeffs_right, R2, P2, image_size, cv2.CV_16SC2)

while True:
   # Read stereo images
   ret,image_left = cam1.read()
   ret, image_right = cam2.read()

   # Remap the images using rectification maps
   rectified_left = cv2.remap(image_left, map1_left, map2_left, cv2.INTER_LINEAR)
   rectified_right = cv2.remap(image_right, map1_right, map2_right, cv2.INTER_LINEAR)

    # Convert images to grayscale
   gray_left = cv2.cvtColor(rectified_left, cv2.COLOR_BGR2GRAY)
   gray_right = cv2.cvtColor(rectified_right, cv2.COLOR_BGR2GRAY)

   # Compute disparity map
   disparity = stereo.compute(gray_left, gray_right)

   # Normalize the disparity map to the range [0, 1]
   normalized_disparity_map = cv2.normalize(disparity, None, 0.0, 1.0, cv2.NORM_MINMAX,cv2.CV_32F)


   colormap_image = cv2.applyColorMap(np.uint8(normalized_disparity_map * 255), cv2.COLORMAP_JET)
   #cv2.imshow('Depth colour map',colormap_image )
   com_img=  cv2.hconcat([rectified_left,rectified_right])
   com_img=draw_lines(com_img)
   cv2.imshow('img left - img right - depth map',cv2.hconcat([cv2.resize(com_img, (0,0), fx=0.6, fy=0.6),cv2.resize(colormap_image, (0,0), fx=0.6, fy=0.6)])) 
   k=cv2.waitKey(33)
   if k == ord('x'):
     break
   elif k == ord('q'):
     #increase block size
     block_s +=2
     print("block_size:"+str(block_s))
     stereo.setBlockSize(block_s)

   elif k == ord('a'):
     #decrease block size
     block_s =max(block_s-2,5)
     print("block_size:"+str(block_s))
     stereo.setBlockSize(block_s)

   elif k == ord('w'):
     #increase disparity
     num_disp +=16
     print("disparity:"+str(num_disp))
     stereo.setNumDisparities(num_disp)

   elif k == ord('s'):
     #decrease disparity
     num_disp =max(16, num_disp-16)
     print("disparity:"+str(num_disp))
     stereo.setNumDisparities(num_disp)

cv2.destroyAllWindows()
cam1.stop()
cam2.stop()
cam1.release()
cam2.release()


