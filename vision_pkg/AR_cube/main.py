import cv2
import numpy as np
import os
from vision_pkg.AR_cube.vision import Vision

vs = Vision()

image_count = 736

# Load camera poses
# Each row i of matrix 'poses' contains the transformations that transforms
# points expressed in the world frame to points expressed in the camera frame.

pose_vectors = np.matrix(np.loadtxt(str(os.getcwd()) + '/data/poses.txt'))

# Define 3D corner position

# [Nx3] matrix containing the corners of the checkerboard as 3D points
# (X,Y,Z), expressed in the world coordinate system

square_size = 0.04
num_corners_x = 9
num_corners_y = 6
num_corners = num_corners_x * num_corners_y

[X, Y] = np.meshgrid(np.arange(num_corners_x), np.arange(num_corners_y))
p_W_corners = square_size * np.matrix(zip(X.flatten(), Y.flatten()))
p_W_corners = np.hstack((p_W_corners, np.zeros((num_corners, 1)))).T

# Load camera intrinsics

K = np.matrix(np.loadtxt(str(os.getcwd()) + '/data/K.txt'))     # calibration matrix [3x3]
D = np.matrix(np.loadtxt(str(os.getcwd()) + '/data/D.txt')).T     # distortion coefficients [2x1]

# for image in range(image_count):
    # print image
    # image_index = image+1

image_index = 1
# Load one image with a given index
img = cv2.imread(str(os.getcwd()) + '/data/images/img_%04d.jpg' % image_index)

# Project the corners on the image

# Compute the 4x4 homogeneous transformation matrix that maps points from the world
# to the camera coordinate frame

T_C_W = vs.poseVectorToTransformationMatrix(pose_vectors[image_index-1, :])

# Transform 3d points from world to current camera pose
p_C_corners = T_C_W * np.matrix(np.vstack((p_W_corners, np.ones((1, num_corners)))))
p_C_corners = p_C_corners[0:3, :]
projected_pts = vs.projectPoints(p_C_corners, K, D)

for point in projected_pts.T:
    cv2.drawMarker(img, (int(point[0, 0]), int(point[0, 1])), (0, 0, 255), markerType=cv2.MARKER_STAR,
                   markerSize=4, thickness=2, line_type=cv2.LINE_AA)

cv2.imshow("plotted image", img)

# Undistort image with bilinear interpolation
img_undistorted = vs.undistortImage(img, K, D, 1)
cv2.imshow("undis_image", img_undistorted)

# Vectorized undistortion without bilinear interpolation
img_undistorted_vectorized = vs.undistortImageVectorized(img, K, D)
cv2.imshow("vectorozed_image", img_undistorted_vectorized)

# Draw a cube on the undistorted image
offset_x = 0.04 * 3
offset_y = 0.04
s = 2 * 0.04
[X, Y, Z] = np.meshgrid([0, 1], [0, 1], [-1, 0])
X = X.flatten()
Y = Y.flatten()
Z = Z.flatten()
p_W_cube = np.matrix(np.vstack((offset_x + X*s, offset_y + Y[:]*s, Z[:]*s, np.ones((1, 8)))))

p_C_cube = T_C_W * p_W_cube
p_C_cube = p_C_cube[0:3, :]

cube_pts = np.matrix(vs.projectPoints(p_C_cube, K, np.zeros((4, 1))).T)
cube_pts = cube_pts.astype(np.int)
# print np.vstack((X,Y,Z))

lw = 5
# base layer of the cube
img_undistorted_vectorized = cv2.line(img_undistorted_vectorized, (cube_pts[1, 0], cube_pts[1, 1]), (cube_pts[3, 0], cube_pts[3, 1]), (0, 0, 255), lw)
img_undistorted_vectorized = cv2.line(img_undistorted_vectorized, (cube_pts[3, 0], cube_pts[3, 1]), (cube_pts[7, 0], cube_pts[7, 1]), (0, 0, 255), lw)
img_undistorted_vectorized = cv2.line(img_undistorted_vectorized, (cube_pts[7, 0], cube_pts[7, 1]), (cube_pts[5, 0], cube_pts[5, 1]), (0, 0, 255), lw)
img_undistorted_vectorized = cv2.line(img_undistorted_vectorized, (cube_pts[5, 0], cube_pts[5, 1]), (cube_pts[1, 0], cube_pts[1, 1]), (0, 0, 255), lw)

# top layer
img_undistorted_vectorized = cv2.line(img_undistorted_vectorized, (cube_pts[0, 0], cube_pts[0, 1]), (cube_pts[2, 0], cube_pts[2, 1]), (0, 255, 0), lw)
img_undistorted_vectorized = cv2.line(img_undistorted_vectorized, (cube_pts[2, 0], cube_pts[2, 1]), (cube_pts[6, 0], cube_pts[6, 1]), (0, 255, 0), lw)
img_undistorted_vectorized = cv2.line(img_undistorted_vectorized, (cube_pts[6, 0], cube_pts[6, 1]), (cube_pts[4, 0], cube_pts[4, 1]), (0, 255, 0), lw)
img_undistorted_vectorized = cv2.line(img_undistorted_vectorized, (cube_pts[4, 0], cube_pts[4, 1]), (cube_pts[0, 0], cube_pts[0, 1]), (0, 255, 0), lw)

# vertical lines
img_undistorted_vectorized = cv2.line(img_undistorted_vectorized, (cube_pts[1, 0], cube_pts[1, 1]), (cube_pts[0, 0], cube_pts[0, 1]), (255, 0, 0), lw)
img_undistorted_vectorized = cv2.line(img_undistorted_vectorized, (cube_pts[3, 0], cube_pts[3, 1]), (cube_pts[2, 0], cube_pts[2, 1]), (255, 0, 0), lw)
img_undistorted_vectorized = cv2.line(img_undistorted_vectorized, (cube_pts[7, 0], cube_pts[7, 1]), (cube_pts[6, 0], cube_pts[6, 1]), (255, 0, 0), lw)
img_undistorted_vectorized = cv2.line(img_undistorted_vectorized, (cube_pts[5, 0], cube_pts[5, 1]), (cube_pts[4, 0], cube_pts[4, 1]), (255, 0, 0), lw)

cv2.imshow("vectorozed_image", img_undistorted_vectorized)
cv2.waitKey(1)

cv2.destroyAllWindows()


