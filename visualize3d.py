import argparse
import os
import json
from mayavi.tools.camera import view
import numpy as np
import seaborn as sns
import mayavi.mlab as mlab
import cv2


nuscene_colors = sns.color_palette('bright', 6)

colors = sns.color_palette('Paired', 9 * 2)


names = ['Car', 'Van', 'Truck', 'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram', 'Misc', 'DontCare']

nuscene_class = {
    0 : 'car', 1 : 'truck', 2 : 'trailer', 3 : 'bus', 4 : 'construction_vehicle', 5 : 'bicycle',
    6 : 'motorcycle', 7 : 'pedestrian', 8 : 'traffic_cone', 9 : 'barrier', -1 : 'DontCare'
}
nuscene_class_to_color = {'car' : nuscene_colors[0], 'truck': nuscene_colors[0], 'trailer': nuscene_colors[1], 'bus': nuscene_colors[1], 'construction_vehicle': nuscene_colors[4], 
'bicycle': nuscene_colors[2], 'motorcycle': nuscene_colors[2], 'pedestrian': nuscene_colors[2], 'traffic_cone': nuscene_colors[3], 'barrier': nuscene_colors[4], 'DontCare' : nuscene_colors[5]}

def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')

    parser.add_argument('--dataset', type=str, default=None, help='specify the type of dataset')
    parser.add_argument('--view', type=str, default=None, help='3D view, BEV')
    parser.add_argument('--start_file', type=str, default=None, help='Name for the start file')
    parser.add_argument('--seq_length', type=int, default=None, help='The length of sequence')

    args = parser.parse_args()

    
    return args

def image_to_rect(projected_x,projected_y,cz):

    theta = ((projected_x / 3760) * (np.pi * 2)) - np.pi
    x_rect = cz * np.tan(theta)
    horizontal_theta_rect = np.arctan(x_rect / cz)
    horizontal_theta_rect += (cz < 0) * np.pi
    y_rect = (projected_y - (0.4375 * 480))*((1 / np.cos(horizontal_theta_rect))*cz)/(485.78)
    z_rect = cz
    return x_rect,y_rect,z_rect

def projection(x,y,z):
    '''
    JRDB에서 메일로 받은 코드
    '''
    horizontal_theta = np.arctan(x / z)
    horizontal_theta += (z < 0) * np.pi
    horizontal_percent = horizontal_theta/(2 * np.pi)
    result_x = ((horizontal_percent * 3760) + 1880) % 3760
    result_y = (485.78 * (y / ((1 / np.cos(horizontal_theta)) *
        z))) + (0.4375 * 480)
    return result_x,result_y



def gt(cfgs, idx, lidar_path, image_file,class_label_path, bbox_label_path,tag):
    image = cv2.imread(image_file, cv2.COLOR_BGR2RGB) # X    
    #points = np.fromfile(str(lidar_path), dtype=np.float32, count=-1).reshape([-1, 5])[:,:3]
    
    points = np.load(lidar_path).reshape([-1, 4])[:,:3]
    
    # points = np.fromfile(str(points_dir), dtype=np.float32, count=-1).reshape([-1, 3])
    # assert points.ndim != 2 or points.shape[-1] != 3, "points.shape should be (N x 3)"

    fig = mlab.figure(bgcolor=(0, 0, 0), size=(1280, 720)) #default = size=(1280, 720)
    view_type = cfgs['view_type']
    
    if view_type == '3D':
        mlab.points3d(points[:, 0], points[:, 1], points[:, 2], mode="sphere", figure=fig, color = (1,1,1),scale_factor = 0.02)
    elif view_type == 'BEV':
        mlab.points3d(points[:, 0], points[:, 1], np.zeros(len(points[:,2])), mode="sphere", figure=fig, color = (1,1,1),scale_factor = 0.04, scale_mode='none')
    else:
        pass
    
    corners_set = np.load(bbox_label_path)
    class_label_set = np.load(class_label_path).astype(int)
    total_label_cnt = corners_set.shape[0]

    for i in range(total_label_cnt):
        corners_3d = corners_set[i,:,:]
        class_label = class_label_set[i]
        # plot_nuscenes_BEV(fig,corners_3d)
        plot_nuscenes_BEV_v2(fig,corners_3d, class_label)
    
    '''
    with open(label_dir, 'r') as f:
        labels = f.readlines()
    
    for line in labels:
        line = line.split()
        lab, _, _, _, xmin, ymin, xmax, ymax, h, w, l, x, y, z, rot, _ ,_,_= line
        rot = str(np.pi/2-float(rot))

        xmin,ymin,xmax,ymax,h, w, l, x, y, z, rot = map(float, [xmin,ymin,xmax,ymax,h, w, l, x, y, z, rot])

        # if z > 0.1 or z < -0.1 and ymax - ymin < 480:
        #     y,h = convert_label(xmin,ymin,xmax,ymax,h, w, l, x, y, z, rot, image,points)
        
        
        if view_type == '3D':
            plot_3d(lab,points,fig,h, w, l, x, y, z, rot)
        elif view_type == 'BEV':
            plot_BEV(lab,points,fig,h, w, l, x, y, z, rot)
        elif view_type == 'LiDAR_to_Image':
            plot_2d_bbox(xmin,ymin,xmax,ymax,h, w, l, x, y, z, rot,image)   #2d box
            plot_3d_bbox(image,xmin,ymin,xmax,ymax,h, w, l, x, y, z, rot)   #3d -> 2d projection

        else: 
            pass
    
    ''' 
    
    '''
    inds = np.sqrt(points[:,0]**2+points[:,1]**2) > 10
    mlab.points3d(points[inds,0],points[inds,1],np.zeros(len(points[inds,2])),mode="sphere",figure=fig,color=(0,1,0),scale_factor = 0.05)
    '''
    
    #mlab.view(azimuth=170,elevation=0,distance='10',roll=None,focalpoint=(0,0,0),reset_roll=True)
    #mlab.view(azimuth=170,elevation=0,distance='auto',roll=None,focalpoint=(0,0,20),reset_roll=True)
    mlab.view(azimuth=170,elevation=0,distance=130,roll=None,focalpoint=(5,0,10),reset_roll=True)
    
    
    save_path = cfgs['save_path']
    save_dir_name = cfgs['start_file']
    if not os.path.isdir(save_path + '/'+ save_dir_name):
       os.mkdir(save_path + '/'+ save_dir_name)
    
    if tag == 'ori':
        mlab.savefig(filename=f'{save_path}/{save_dir_name}/{idx}_ori.jpg', size=(4000,3500), magnification=5)
    elif tag == 'aug':
        mlab.savefig(filename=f'{save_path}/{save_dir_name}/{idx}_aug.jpg',size=(4000,3500), magnification=5)
    else:
        pass
    mlab.show()
    #mlab.close()

def plot_BEV(lab,points,fig,h, w, l, x, y, z, rot):
    if lab != 'DontCare':
        x_corners = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]
        y_corners = [0, 0, 0, 0, -h, -h, -h, -h]
        z_corners = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]
        corners_3d = np.vstack([x_corners, y_corners, z_corners])  # (3, 8)

        # transform the 3d bbox from object coordiante to camera_0 coordinate
        R = np.array([[np.cos(rot), 0, np.sin(rot)],
                    [0, 1, 0],
                    [-np.sin(rot), 0, np.cos(rot)]])
        corners_3d = np.dot(R, corners_3d).T + np.array([x, y, z])

        # transform the 3d bbox from camera_0 coordinate to velodyne coordinate
        corners_3d = corners_3d[:, [2, 0, 1]] * np.array([[1, -1, -1]])

        
        #inds = np.sqrt(points[:,0]**2+points[:,1]**2) > 10
        #mlab.points3d(points[inds,0],points[inds,1],np.zeros(len(points[inds,2])),mode="sphere",figure=fig,color=(0,1,0),scale_factor = 0.05)


        def draw(p1, p2, front=1):
            mlab.plot3d([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]],
                        color=colors[names.index(lab) * 2 + front], tube_radius=None, line_width=4, figure=fig)

        # draw the lower 4 horizontal lines
        draw(corners_3d[0], corners_3d[1], 0)  # front = 0 for the front lines
        draw(corners_3d[1], corners_3d[2])
        draw(corners_3d[2], corners_3d[3])
        draw(corners_3d[3], corners_3d[0])

def plot_nuscenes_BEV(fig,corners_3d):
        
        def draw(p1, p2, front=1):
            mlab.plot3d([p1[0], p2[0]], [p1[1], p2[1]], [0, 0],
                        color=nuscene_colors[0], tube_radius=None, line_width=2, figure=fig)

        # draw the lower 4 horizontal lines
        draw(corners_3d[0], corners_3d[3])  # front = 0 for the front lines
        draw(corners_3d[0], corners_3d[4])
        draw(corners_3d[3], corners_3d[7])
        draw(corners_3d[4], corners_3d[7])

def plot_nuscenes_BEV_v2(fig,corners_3d,class_label):
        

        def draw(p1, p2, class_label):
            mlab.plot3d([p1[0], p2[0]], [p1[1], p2[1]], [0, 0],
                        color=nuscene_class_to_color[nuscene_class[class_label]], tube_radius=None, line_width=4, figure=fig)

        # draw the lower 4 horizontal lines
        draw(corners_3d[0], corners_3d[3], class_label=class_label)  # front = 0 for the front lines
        draw(corners_3d[0], corners_3d[4], class_label=class_label)
        draw(corners_3d[3], corners_3d[7], class_label=class_label)
        draw(corners_3d[4], corners_3d[7], class_label=class_label)

def plot_3d(lab,points,fig,h, w, l, x, y, z, rot):
    if lab != 'DontCare':
        x_corners = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]
        y_corners = [0, 0, 0, 0, -h, -h, -h, -h]
        z_corners = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]
        corners_3d = np.vstack([x_corners, y_corners, z_corners])  # (3, 8)

        # transform the 3d bbox from object coordiante to camera_0 coordinate
        R = np.array([[np.cos(rot), 0, np.sin(rot)],
                    [0, 1, 0],
                    [-np.sin(rot), 0, np.cos(rot)]])
        corners_3d = np.dot(R, corners_3d).T + np.array([x, y, z])

        # transform the 3d bbox from camera_0 coordinate to velodyne coordinate
        corners_3d = corners_3d[:, [2, 0, 1]] * np.array([[1, -1, -1]])

        inds = (points[:,0]>min(corners_3d[:,0]))*(points[:,0]<max(corners_3d[:,0]))*(points[:,1]>min(corners_3d[:,1]))*(points[:,1]<max(corners_3d[:,1]))*(points[:,2]<min(corners_3d[:,2]))
        mlab.points3d(points[inds,0],points[inds,1],points[inds,2],mode="sphere",figure=fig,color=(0,1,0),scale_factor = 0.05)

        inds_ = (points[:,0]>min(corners_3d[:,0]))*(points[:,0]<max(corners_3d[:,0]))*(points[:,1]>min(corners_3d[:,1]))*(points[:,1]<max(corners_3d[:,1]))*(points[:,2]>max(corners_3d[:,2]))
        mlab.points3d(points[inds_,0],points[inds_,1],points[inds_,2],mode="sphere",figure=fig,color=(0,1,0),scale_factor = 0.05)

        def draw(p1, p2, front=1):
            mlab.plot3d([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]],
                        color=colors[names.index(lab) * 2 + front], tube_radius=None, line_width=2, figure=fig)


        # draw the upper 4 horizontal lines
        draw(corners_3d[0], corners_3d[1], 0)  # front = 0 for the front lines
        draw(corners_3d[1], corners_3d[2])
        draw(corners_3d[2], corners_3d[3])
        draw(corners_3d[3], corners_3d[0])

        # draw the lower 4 horizontal lines
        draw(corners_3d[4], corners_3d[5], 0)
        draw(corners_3d[5], corners_3d[6])
        draw(corners_3d[6], corners_3d[7])
        draw(corners_3d[7], corners_3d[4])

        # draw the 4 vertical lines
        draw(corners_3d[4], corners_3d[0], 0)
        draw(corners_3d[5], corners_3d[1], 0)
        draw(corners_3d[6], corners_3d[2])
        draw(corners_3d[7], corners_3d[3])
'''
def plot_3d_bbox(img,xmin,ymin,xmax,ymax,h, w, l, x, y, z, rot):
    box_3d = []
    center = np.array([x,y,z])
    dims = np.array([h,w,l])
    rot_y = rot
    
    for i in [1, -1]:
        for j in [1, -1]:
            for k in [0, 1]:
                point = np.copy(center)
                

                h,w,l = dims
                point[0] = center[0] + (j * i) * l / 2 * np.cos(rot_y) + i * w / 2 * np.sin(rot_y) 
                point[2] = center[2] - (j * i) * l / 2 * np.sin(rot_y) + i * w / 2 * np.cos(rot_y) 
                point[1] = center[1] - k * h
                x = point[0]
                y = point[1]
                z = point[2]
                
                
                point[0],point[1]=projection(x,y,z)
                point = point.astype(np.int16)
                box_3d.append(point[:2])
    
    best = []
    for i in range(8):
        point_1_ = box_3d[i]
        point_2_ = box_3d[(i + 2) % 8]
        best.append(abs(point_1_[0]-point_2_[0]))
    if max(best) > 1000:
        return img
    
    for i in range(8):
        point_1_ = box_3d[i]
        point_2_ = box_3d[(i + 2) % 8]
        cv2.line(img, (point_1_[0], point_1_[1]), (point_2_[0], point_2_[1]), colors1['pink'], 3)
    cv2.line(img,(box_3d[1][0],box_3d[1][1]),(box_3d[7][0],box_3d[7][1]),colors1['nuscene'],3)
    cv2.line(img,(box_3d[0][0],box_3d[0][1]),(box_3d[6][0],box_3d[6][1]),colors1['nuscene'],3)

    for i in range(4):
        point_1_ = box_3d[2 * i]
        point_2_ = box_3d[2 * i + 1]
        cv2.line(img, (point_1_[0], point_1_[1]), (point_2_[0], point_2_[1]), colors1['pink'], 3)
    
    cv2.line(img,(box_3d[0][0],box_3d[0][1]),(box_3d[1][0],box_3d[1][1]),colors1['nuscene'],3)
    cv2.line(img,(box_3d[7][0],box_3d[7][1]),(box_3d[6][0],box_3d[6][1]),colors1['nuscene'],3)

    cv2.imwrite(f"{file_id}.jpg", img)
'''
def plot_2d_bbox(xmin,ymin,xmax,ymax,h, w, l, x, y, z, rot,image):

    img = cv2.rectangle(image,(int(xmin),int(ymin)),(int(xmax),int(ymax)),(255,0,0),2)
    # cv2.imshow(f"{file_id}.jpg", img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # cv2.imwrite(f"{file_id}.jpg", img)
    cv2.imwrite("2d.jpg", img)

def convert_label(*args):
    
    center = args[7:10]
    dims = args[4:7]
    rot_y = args[-3]
    box_3d = []
    box_2d = args[:4]
    points = args[-1]
    for i in [1, -1]:
        for j in [1, -1]:
            for k in [0, 1]:

                '''
                center로 부터 corner를 만드는 과정 (총 8개)
                '''
                point = np.copy(center)
                point[0] = center[0] + (j * i) * dims[2] / 2 * np.cos(rot_y) + i * dims[1] / 2 * np.sin(rot_y) 
                point[2] = center[2] - (j * i) * dims[2] / 2 * np.sin(rot_y) + i * dims[1] / 2 * np.cos(rot_y) 
                point[1] = center[1] - k * dims[0]
                x = point[0]
                y = point[1]
                z = point[2]


                point[0],point[1] = projection(x,y,z)

                point = point.astype(np.int16)
                box_3d.append(point[:2])

    x_p = []
    
    for i in range(len(box_3d)):
        x_p.append(box_3d[i][0])
        
    target_x = sum(sorted(x_p)[::2]) / len(sorted(x_p)[::2])

    if max(sorted(x_p))-min(sorted(x_p)) > 1000:
        return center[1] , dims[0]     

    _, result_y_up, _ = image_to_rect(target_x,box_2d[1],center[2])
    _, result_y, _ = image_to_rect(target_x,box_2d[3],center[2])
    

    h = abs(result_y_up-result_y)
    w = dims[1]
    l = dims[2]
    x_corners = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]
    y_corners = [0, 0, 0, 0, -h, -h, -h, -h]
    z_corners = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]
    corners_3d = np.vstack([x_corners, y_corners, z_corners])  # (3, 8)

    # transform the 3d bbox from object coordiante to camera_0 coordinate
    R = np.array([[np.cos(rot_y), 0, np.sin(rot_y)],
                [0, 1, 0],
                [-np.sin(rot_y), 0, np.cos(rot_y)]])
    corners_3d = np.dot(R, corners_3d).T + np.array([center[0], result_y, center[2]])

    # transform the 3d bbox from camera_0 coordinate to velodyne coordinate
    corners_3d = corners_3d[:, [2, 0, 1]] * np.array([[1, -1, -1]])
    inds_ = (points[:,0]>min(corners_3d[:,0]))*(points[:,0]<max(corners_3d[:,0]))*(points[:,1]>min(corners_3d[:,1]))*(points[:,1]<max(corners_3d[:,1]))*(points[:,2]>max(corners_3d[:,2]))
    if sum(inds_) != 0:
        if abs(min(points[inds_,2])-max(corners_3d[:,2])) < 0.05:
                return result_y, abs(result_y_up-result_y) + 0.1            

    return result_y, abs(result_y_up-result_y)
    

def load_file_list(cfgs):
    start_token = cfgs['start_file']
    file_list_path = os.getcwd() + '/' + cfgs['dataset_type']
    file_list = []

    with open(file_list_path + '/sample.json', 'r') as f:
        sample_json = json.load(f)

    for i in range(len(sample_json)):
        if start_token == sample_json[i]['token']:
            start_idx = i
            break
    
    idx = start_idx
    
    for i in range(cfgs['seq_length']):
        file_list.append(sample_json[idx]['token'])
        if sample_json[idx]['next'] == '':
            print("File no more exist for the same scene\n")
            break
        idx += 1

    return file_list    

    pass

if __name__ == '__main__':
    type = 'train'
    
    args = parse_config()
    
    cfgs = {'dataset_type' : args.dataset,
            'view_type'    : args.view,
            'start_file'    : args.start_file,
            'seq_length'    : args.seq_length
             }

    file_list = []
    file_list = load_file_list(cfgs)
    
    
    if cfgs['dataset_type'] == 'nuscenes':
        sample_path = 'nuscenes/samples/',
        bbox_label_path = 'nuscenes/corners/',
        class_label_path = 'nuscenes/class_label/'
        file_suffix = '.npy'
        cfgs['save_path'] = './nuscenes/results'
    

    for idx, file_name in enumerate(file_list):
        
        a = str(0).zfill(6)
        file_id = a
        img_file = f'./{file_id}.jpg'
        #label_dir = f'./{file_id}.txt'
        
        lidar_path = f'./nuscenes/samples/{file_name}_ori{file_suffix}'
        corner_path = f'./nuscenes/corners/{file_name}_ori{file_suffix}'
        class_label_path = f'./nuscenes/class_label/{file_name}_ori{file_suffix}'
        gt(cfgs = cfgs, idx = idx, lidar_path=lidar_path, image_file=img_file, class_label_path=class_label_path, bbox_label_path=corner_path, tag = 'ori')

        lidar_path = f'./nuscenes/samples/{file_name}_aug{file_suffix}'
        corner_path = f'./nuscenes/corners/{file_name}_aug{file_suffix}'
        class_label_path = f'./nuscenes/class_label/{file_name}_aug{file_suffix}'
        gt(cfgs = cfgs, idx = idx, lidar_path=lidar_path, image_file=img_file, class_label_path=class_label_path, bbox_label_path=corner_path, tag = 'aug')
    