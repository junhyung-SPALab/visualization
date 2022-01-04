import json
import os

with open(os.getcwd() + '/nuscenes/sample.json', 'r') as f:
    sample_json = json.load(f)

pathClasslabel = os.getcwd() + '/nuscenes/val_set/class_label'
pathCorners = os.getcwd() + '/nuscenes/val_set/corners'
pathSamples = os.getcwd() + '/nuscenes/val_set/samples'

file_list_classlabel    = os.listdir(pathClasslabel)
file_list_corners       = os.listdir(pathCorners)
file_list_samples       = os.listdir(pathSamples)

print(len(file_list_classlabel))
print(len(file_list_corners))
print(len(file_list_samples))

scene_token_list = []

for i in range(len(file_list_corners)):
	tmp_file_name_split = file_list_corners[i].split('_')
	if tmp_file_name_split[0] not in scene_token_list:
		scene_token_list.append(tmp_file_name_split[0])

idx = 0
scene_index_dict = {}
latest_scene_token = None

for i in range(len(sample_json)):
	cur_scene_token = sample_json[i]['scene_token']
	if (cur_scene_token in scene_token_list) & (cur_scene_token != latest_scene_token):
		latest_scene_token = cur_scene_token
		scene_index_dict[idx] = cur_scene_token
		idx += 1
	else:
		pass

for key, value in scene_index_dict.items():
	print(key, value)
	scene_idx = key
	scene_token = value
	cnt = 0
	for i in range(len(sample_json)):
		if sample_json[i]['scene_token'] == scene_token:
			cur_token = sample_json[i]['token']
			old_name = f'{pathCorners}/{scene_token}_{cur_token}_ori.npy'
			new_name = f'{pathCorners}/val_scene_{scene_idx}_{cnt}_ori.npy'
			cnt += 1
			os.rename(old_name, new_name)
		



                


