# configs/_base_/datasets/ranpak_flaps.py
dataset_info = dict(
    dataset_name='ranpak_flaps',
    paper_info=dict(author='Ranpak', title='Box Flap Metrology', year='2026'),
    keypoint_info={
        0: dict(name='NORTH_1', id=0, color=[51, 153, 255], type='upper', swap='NORTH_2'),
        1: dict(name='NORTH_2', id=1, color=[51, 153, 255], type='upper', swap='NORTH_1'),
        2: dict(name='EAST_1', id=2, color=[0, 255, 0], type='upper', swap='WEST_2'),
        3: dict(name='EAST_2', id=3, color=[0, 255, 0], type='upper', swap='WEST_1'),
        4: dict(name='SOUTH_1', id=4, color=[255, 128, 0], type='lower', swap='SOUTH_2'),
        5: dict(name='SOUTH_2', id=5, color=[255, 128, 0], type='lower', swap='SOUTH_1'),
        6: dict(name='WEST_1', id=6, color=[255, 51, 255], type='upper', swap='EAST_2'),
        7: dict(name='WEST_2', id=7, color=[255, 51, 255], type='upper', swap='EAST_1'),
    },
    skeleton_info={
        0: dict(link=('NORTH_1', 'NORTH_2'), id=0, color=[255, 255, 255]),
        1: dict(link=('EAST_1', 'EAST_2'), id=1, color=[255, 255, 255]),
        2: dict(link=('SOUTH_1', 'SOUTH_2'), id=2, color=[255, 255, 255]),
        3: dict(link=('WEST_1', 'WEST_2'), id=3, color=[255, 255, 255]),
    },
    joint_weights=[1.] * 8,
    sigmas=[0.1] * 8)