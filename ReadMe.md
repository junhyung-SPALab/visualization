### Install
    - Git repository clone
        git clone https://github.com/junhyung-SPALab/visualization.git

    - Change 'environment.yml'
        name: {Your conda env name}
        prefix: /home/{Your pc host name}/anaconda3 (설치된 conda 경로를 접두사로 지정할 것)
    
    - Create conda env for utilizing mayavi
        conda env create -f environment.yml
    

### Command
    argument
     --dataset : 'nuscenes'
     --view : '3D', 'BEV'
     --start_file : 'file name'
     --seq_length : 확인하고자 하는 하는 시퀀스 길이

     example
     python visualize3d.py --dataset {dataset} --view {view} --start_file {Token name} --seq_length {int}
     python visualize3d.py --dataset nuscenes --view 'BEV' --start_file 'aa977e6409f249a9a8a8dc6d24b390f7' --seq_length 1

### File system (You need to prepare)
    nuScenes dataset
    /nuscenes
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