import os.path
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    package_path = get_package_share_directory('point_lio')
    config_file = os.path.join(package_path, 'config','siasun.yaml')
    rviz_file = os.path.join(package_path, 'config', 'rviz.rviz')
    # use_sim_time = False
    point_lio_node = Node(
        package='point_lio',
        executable='pointlio_mapping',
        parameters=[config_file],  # , {'use_sim_time': use_sim_time}
        # prefix=['gnome-terminal -- gdb -ex run --args'],
        output='screen'
    )
    
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        arguments=['-d', rviz_file]
    )

    ld = LaunchDescription()
    ld.add_action(point_lio_node)
    ld.add_action(rviz_node)

    return ld
