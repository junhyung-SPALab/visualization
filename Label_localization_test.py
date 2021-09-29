import cv2
import numpy as np
import matplotlib.pyplot as plt

#IN_PCD_PATH     =   '/TANet/data/JRDB_kitti/training/velodyne/'
#IN_LABEL_PATH   =   '/TANet/data/JRDB_kitti/training/label_2/'
IN_PCD_PATH     = './'
IN_LABEL_PATH     = './'
#IN_INDEX        =   '013751'
#IN_INDEX        =   '000000'
#IN_INDEX        =   '007920'
IN_INDEX        =   'n015-2018-08-01-16-54-05+0800__LIDAR_TOP__1533113750697793.pcd'

#Visualization factor
scale_factor    =   30
blue            = (255, 0, 0)
green           = (0, 255, 0)
red             = (0, 0, 255)
thickness       = [1, 2]
radius_robot    = 10

#############################################

#raw_data_pcd = np.load(IN_PCD_PATH + IN_INDEX + '.npy')
raw_data_pcd = np.fromfile(IN_PCD_PATH + IN_INDEX + '.bin', dtype=np.float32).reshape([-1, 5])

vis_p = raw_data_pcd[:,:2]

vis_p *= scale_factor

pos_x_min = vis_p[:,0].min()
pos_y_min = vis_p[:,1].min()

vis_p[:,0] -= pos_x_min
vis_p[:,1] -= pos_y_min

x_idx = np.round(vis_p[:,0]).astype(np.int64)
y_idx = np.round(vis_p[:,1]).astype(np.int64)

x_max_idx = np.max(x_idx)
y_max_idx = np.max(y_idx)

bev = np.zeros((x_max_idx+1,y_max_idx+1, 3))

bev[x_idx, y_idx, :] = 255

cv2.imwrite('BEV' + IN_INDEX +'.png', bev)
'''
#############################################
# Load label data & Visualization of BEV bbox

label_data = open(IN_LABEL_PATH + IN_INDEX + '.txt',"r")
label_string = label_data.readlines()

gt_3D_dim_scaled = []
gt_3D_loc_scaled = []

for label_str in label_string:
    label_str_split = label_str.split(" ")
    tmp_3D_dim = map(float, label_str_split[-8:-5])
    tmp_3D_loc = map(float, label_str_split[-5:-2])
    gt_3D_dim_scaled.append([i * scale_factor for i in tmp_3D_dim])
    gt_3D_loc_scaled.append([i * scale_factor for i in tmp_3D_loc])


# Load projected point cloud image
bev_domain = cv2.imread('BEV' + IN_INDEX +'.png')

# Visualize the robot's position & X-Axis(LiDAR coordinate)
pos_robot = (np.round(-pos_x_min).astype(np.int64), np.round(-pos_y_min).astype(np.int64))
cv2.circle(bev_domain, (pos_robot[1], pos_robot[0]), radius_robot, red, thickness[1])
for i in range (radius_robot):
    bev_domain[pos_robot[0]+i, pos_robot[1], 0] = 0
    bev_domain[pos_robot[0]+i, pos_robot[1], 1] = 255
    bev_domain[pos_robot[0]+i, pos_robot[1], 2] = 255


# Visualize projected bbox of ground truth label 
for i in range(len(gt_3D_loc_scaled)):
    h_scaled, w_scaled, l_scaled       = gt_3D_dim_scaled[i]
    x_c, y_c, z_c = gt_3D_loc_scaled[i]
    print('h: %s / w: %s / l: %s /x_c: %s /y_c: %s /z_c: %s'%(h_scaled,w_scaled,l_scaled,x_c,y_c,z_c))
    
    lidar_x_c = z_c
    lidar_y_c = -x_c

    print("lidar_x_c : %d, lidar_y_c : %d"%(lidar_x_c,lidar_y_c))

    # Translation of bbox center point
    lidar_x_c -= pos_x_min
    lidar_y_c -= pos_y_min
    
    lidar_x_c_int = np.round(lidar_x_c).astype(np.int64)
    lidar_y_c_int = np.round(lidar_y_c).astype(np.int64)
    
    # Visualize center point of bbox
    bev_domain[lidar_x_c_int, lidar_y_c_int, 0] = 0
    bev_domain[lidar_x_c_int, lidar_y_c_int, 1] = 0
    bev_domain[lidar_x_c_int, lidar_y_c_int, 2] = 255
    

    # [Left top y, Left top x,  Right bottom y, Right bottom x]
    bev_bboxes = np.round([lidar_y_c-0.5*w_scaled, lidar_x_c-0.5*l_scaled, lidar_y_c+0.5*w_scaled, lidar_x_c+0.5*l_scaled]).astype(np.int64)

    left_top     = (bev_bboxes[0], bev_bboxes[1])
    right_bottom = (bev_bboxes[2], bev_bboxes[3])

    cv2.rectangle(bev_domain, left_top, right_bottom, green, thickness[0])

# Make marked image file
cv2.imwrite('BEV' + IN_INDEX + '_bbox' + '.png', bev_domain)    
'''