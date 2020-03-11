import numpy as np

class Vision:
    def __init__(self):
        # print("vision initialized")
        pass

    def distortPoints(self, x, D):
        ''' Applies lens distortion D[2x1] to 2D points x[2xN] on the image plane.'''
        # print D
        k1 = D[0, 0]
        k2 = D[1, 0]

        xp = x[0, :]
        yp = x[1, :]

        r2 = np.square(xp) + np.square(yp)

        xpp = np.multiply(xp, 1 + k1*r2 + k2*np.square(r2))
        ypp = np.multiply(yp, 1 + k1*r2 + k2*np.square(r2))

        return np.vstack((xpp, ypp))

    def poseVectorToTransformationMatrix(self, pose_vec):
        '''Converts a 6x1 pose vector into a 4x4 transformation matrix'''

        omega = pose_vec[:, 0:3]
        t = pose_vec[:, 3:6]

        theta = np.linalg.norm(omega)
        k = omega/theta
        kx = k[:, 0]
        ky = k[:, 1]
        kz = k[:, 2]
        K = np.matrix([[0, -kz, ky],
                       [kz, 0, -kx],
                       [-ky, kx, 0]])

        # Rodrigues' rotation formula
        R = np.eye(3) + np.sin(theta)*K + (1-np.cos(theta))*(K**2)

        T = np.eye(4)
        T[0:3, 0:3] = R
        T[0:3, 3] = t
        return T
    
    def projectPoints(self, points_3d, K, D=np.zeros((4,1))):
        ''' Projects 3d points to the image plane [3xN], given the camera matrix [3x3] and
            distortion coefficients [4x1].
        '''

        # get normalized coordinates
        xp = points_3d[0, :] / points_3d[2, :]
        yp = points_3d[1, :] / points_3d[2, :]
    
        # apply distortion

        x_d = self.distortPoints(np.vstack((xp, yp)), D)
        xpp = x_d[0, :]
        ypp = x_d[1, :]

        # convert to pixel coordinates
        projected_points = np.matrix(K) * np.vstack((xpp, ypp, np.ones((1, points_3d.shape[1]))))
        return projected_points[0:2, :]

    def undistortImage(self, img, K, D, bilinear_interpolation = 0):
        """ Corrects an image for lens distortion. """

        [X, Y] = np.meshgrid(np.arange(img.shape[1]), np.arange(img.shape[0]))
        pixels_pos = np.hstack((X.reshape(X.size, 1), Y.reshape(Y.size, 1), np.ones((X.size, 1)))).T
        # convert to normalized coordinates
        normalized_coords = np.linalg.solve(K,pixels_pos)
        # apply distortion
        x_d = self.distortPoints(normalized_coords, D)
        # convert back to pixel coordinates
        distorted_coords = K * np.vstack((x_d, np.ones((1, x_d.shape[1]))))
        distorted_coords[:, :] = distorted_coords[:, :]/distorted_coords[2, :]
        distorted_coords = distorted_coords.astype(np.int)
        undistorted_img = img[distorted_coords[1, :], distorted_coords[0, :], :]
        undistorted_img = undistorted_img.astype(np.uint8)
        return undistorted_img.reshape(img.shape[0], img.shape[1], img.shape[2])

        # for row in range(height):
        #     for column in range(width):
        #         # convert to normalized coordinates
        #         normalized_coords = np.linalg.solve(K,np.matrix([[column], [row], [1]]))
        #
        #         # apply distortion
        #         x_d = self.distortPoints(normalized_coords, D)
        #
        #         # convert back to pixel coordinates
        #         distorted_coords = K * np.vstack((x_d, 1))
        #         u = distorted_coords[0, 0] / distorted_coords[2, 0];
        #         v = distorted_coords[1, 0] / distorted_coords[2, 0];
        #
        #         # bilinear interpolation
        #
        #         u1 = np.floor(distorted_coords[0, 0])
        #         v1 = np.floor(distorted_coords[1, 0])
        #         if bilinear_interpolation > 0:
        #             a = u - u1
        #             b = v - v1
        #             if u1+1 > 0 and u1+1 <= width and v1+1 > 0 and v1+1 <= height:
        #                 undistorted_img[row, column] = (1-b) * ((1-a)*img[v1,u1] + a*img[v1,u1+1])\
        #                                                + b * ((1-a)*img[v1+1,u1] + a*img[v1+1,u1+1])
        #         elif u1 > 0 and u1 <= width and v1 > 0 and v1 <= height:
        #             undistorted_img[row, column] = img[v1,u1]


    def undistortImageVectorized(self, img, K, D):
        """ Corrects an image for lens distortion. """

        [X, Y] = np.meshgrid(np.arange(img.shape[1]), np.arange(img.shape[0]))
        px_locs = np.hstack((X.reshape(X.size, 1), Y.reshape(Y.size, 1), np.ones((X.size, 1)))).T

        normalized_px_locs = np.linalg.inv(K) * px_locs
        normalized_px_locs = normalized_px_locs[0:2, :]
        normalized_dist_px_locs = self.distortPoints(normalized_px_locs, D)
        dist_px_locs = K * np.vstack((normalized_dist_px_locs, np.ones((1, normalized_dist_px_locs.shape[1]))))
        dist_px_locs = dist_px_locs[0:2, :]
        dist_px_locs = dist_px_locs.astype(np.int)
        intensity_vals = img[dist_px_locs[1, :], dist_px_locs[0, :], :]
        intensity_vals = intensity_vals.astype(np.uint8)
        return intensity_vals.reshape(img.shape[0], img.shape[1], img.shape[2])

