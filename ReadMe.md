### Command
    argument
     --dataset : 'nuscenes'
     --view : '3D', 'BEV'
     --start_file : 'file name'
     --seq_length : 출력하고자 하는 시퀀스 길이

     example
     python visualize3d.py --dataset {dataset} --view {view} --start_file {Token name} --seq_length {int}
     python visualize3d.py --dataset nuscenes --view 'BEV' --start_file 'aa977e6409f249a9a8a8dc6d24b390f7' --seq_length 1

### File system
    'nuscenes'
    /class_label
        encoded label for object.npy

    /corners
        corners for bbox.npy (LiDAR coordinate)
    /results
        /Token_name     (Token_name : --start_file)
            0_aug.jpg
            0_aug.jpg
            1_aug.jpg
            1_aug.jpg
                *
                *
                *
            n-1_aug.jpg (n : --seq_length)
            n-1_aug.jpg

            
    /samples
        LiDAR data.npy
    sample_annotation.json
    sample_data.json
    sample.json