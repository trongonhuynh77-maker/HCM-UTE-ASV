import os
from glob import glob
from setuptools import setup, find_packages

package_name = 'mophong'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        
        # --- THÊM 2 DÒNG NÀY VÀO ĐỂ ROS 2 NHẬN DIỆN THƯ MỤC CONFIG VÀ LAUNCH ---
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='anh',
    maintainer_email='anh@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            # Khai báo các file chạy Python của anh ở đây
            'nav2_mission_commander = mophong.nav2_mission_commander:main',
            'sac_local_controller_onnx = mophong.sac_local_controller_onnx:main',
        ],
    },
)