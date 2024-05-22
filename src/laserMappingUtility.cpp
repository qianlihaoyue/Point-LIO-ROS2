#include "laserMapping.h"
#include <pcl/io/pcd_io.h>
#include <geometry_msgs/msg/vector3.hpp>

void LaserMappingNode::readParameters() {
    declare_and_get_parameter<bool>("prop_at_freq_of_imu", prop_at_freq_of_imu, true);
    // declare_and_get_parameter<bool>("use_imu_as_input", use_imu_as_input, true);
    declare_and_get_parameter<bool>("check_satu", check_satu, true);
    declare_and_get_parameter<bool>("space_down_sample", space_down_sample, true);
    // declare_and_get_parameter<bool>("common.con_frame", con_frame, false);
    // declare_and_get_parameter<bool>("common.cut_frame", cut_frame, false);
    declare_and_get_parameter<bool>("mapping.imu_en", imu_en, true);
    // declare_and_get_parameter<bool>("mapping.extrinsic_est_en", extrinsic_est_en, true);
    declare_and_get_parameter<bool>("publish.path_en", path_en, true);
    declare_and_get_parameter<bool>("publish.scan_publish_en", scan_pub_en, true);
    declare_and_get_parameter<bool>("publish.scan_bodyframe_pub_en", scan_body_pub_en, true);
    declare_and_get_parameter<bool>("runtime_pos_log_enable", runtime_pos_log, false);
    declare_and_get_parameter<bool>("pcd_save.pcd_save_en", pcd_save_en, false);
    declare_and_get_parameter<bool>("mapping.gravity_align", gravity_align, true);

    declare_and_get_parameter<int>("init_map_size", init_map_size, 100);
    // declare_and_get_parameter<int>("common.con_frame_num", con_frame_num, 1);
    declare_and_get_parameter<int>("preprocess.lidar_type", p_pre->lidar_type, 1);
    declare_and_get_parameter<int>("point_filter_num", p_pre->point_filter_num, 2);
    declare_and_get_parameter<int>("preprocess.scan_line", p_pre->N_SCANS, 16);
    declare_and_get_parameter<int>("preprocess.scan_rate", p_pre->SCAN_RATE, 10);
    declare_and_get_parameter<int>("preprocess.timestamp_unit", p_pre->time_unit, 1);
    declare_and_get_parameter<int>("pcd_save.interval", pcd_save_interval, -1);

    declare_and_get_parameter<double>("mapping.satu_acc", satu_acc, 3.0);
    declare_and_get_parameter<double>("mapping.satu_gyro", satu_gyro, 35.0);
    declare_and_get_parameter<double>("mapping.acc_norm", acc_norm, 1.0);
    declare_and_get_parameter<float>("mapping.plane_thr", plane_thr, 0.05f);
    // declare_and_get_parameter<double>("common.cut_frame_time_interval", cut_frame_time_interval, 0.1);
    declare_and_get_parameter<double>("common.time_diff_lidar_to_imu", time_diff_lidar_to_imu, 0.0);
    declare_and_get_parameter<double>("filter_size_surf", filter_size_surf_min, 0.5);
    declare_and_get_parameter<double>("filter_size_map", filter_size_map_min, 0.5);
    declare_and_get_parameter<double>("cube_side_length", cube_len, 200);
    declare_and_get_parameter<float>("mapping.det_range", DET_RANGE, 300.f);
    // declare_and_get_parameter<double>("mapping.fov_degree", fov_deg, 180);
    declare_and_get_parameter<double>("mapping.imu_time_inte", imu_time_inte, 0.005);
    declare_and_get_parameter<double>("mapping.lidar_meas_cov", laser_point_cov, 0.1);
    declare_and_get_parameter<double>("mapping.acc_cov_input", acc_cov_input, 0.1);
    declare_and_get_parameter<double>("mapping.vel_cov", vel_cov, 20);
    declare_and_get_parameter<double>("mapping.gyr_cov_input", gyr_cov_input, 0.1);
    declare_and_get_parameter<double>("mapping.gyr_cov_output", gyr_cov_output, 0.1);
    declare_and_get_parameter<double>("mapping.acc_cov_output", acc_cov_output, 0.1);
    declare_and_get_parameter<double>("mapping.b_gyr_cov", b_gyr_cov, 0.0001);
    declare_and_get_parameter<double>("mapping.b_acc_cov", b_acc_cov, 0.0001);
    declare_and_get_parameter<double>("mapping.imu_meas_acc_cov", imu_meas_acc_cov, 0.1);
    declare_and_get_parameter<double>("mapping.imu_meas_omg_cov", imu_meas_omg_cov, 0.1);
    declare_and_get_parameter<double>("mapping.match_s", match_s, 81.0);
    declare_and_get_parameter<double>("preprocess.blind", p_pre->blind, 1.0);

    // 字符串型参数
    declare_and_get_parameter<std::string>("common.lid_topic", lid_topic, "/livox/lidar");
    declare_and_get_parameter<std::string>("common.imu_topic", imu_topic, "/livox/imu");

    // 向量型参数
    declare_and_get_parameter<std::vector<double>>("mapping.gravity", gravity, std::vector<double>());
    declare_and_get_parameter<std::vector<double>>("mapping.gravity_init", gravity_init, std::vector<double>());
    declare_and_get_parameter<std::vector<double>>("mapping.extrinsic_T", extrinT, std::vector<double>());
    declare_and_get_parameter<std::vector<double>>("mapping.extrinsic_R", extrinR, std::vector<double>());
}

template <typename T>
void LaserMappingNode::set_posestamp(T& out) {
    out.position.x = kf_output.x_.pos(0);
    out.position.y = kf_output.x_.pos(1);
    out.position.z = kf_output.x_.pos(2);
    out.orientation.x = kf_output.x_.rot.coeffs()[0];
    out.orientation.y = kf_output.x_.rot.coeffs()[1];
    out.orientation.z = kf_output.x_.rot.coeffs()[2];
    out.orientation.w = kf_output.x_.rot.coeffs()[3];
}

void LaserMappingNode::pointBodyLidarToIMU(PointType const* const pi, PointType* const po) {
    V3D p_body_lidar(pi->x, pi->y, pi->z);
    V3D p_body_imu = Lidar_R_wrt_IMU * p_body_lidar + Lidar_T_wrt_IMU;

    po->x = p_body_imu(0);
    po->y = p_body_imu(1);
    po->z = p_body_imu(2);
    po->intensity = pi->intensity;
}

void LaserMappingNode::publish_frame_world(rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubLaserCloudFullRes) {
    sensor_msgs::msg::PointCloud2 laserCloudmsg;
    pcl::toROSMsg(*feats_down_world, laserCloudmsg);

    laserCloudmsg.header.stamp = get_ros_time(lidar_end_time);
    laserCloudmsg.header.frame_id = "camera_init";
    pubLaserCloudFullRes->publish(laserCloudmsg);

    /**************** save map ****************/
    if (pcd_save_en) {
        *pcl_wait_save += *feats_down_world;

        static int scan_wait_num = 0;
        scan_wait_num++;
        if (pcl_wait_save->size() > 0 && pcd_save_interval > 0 && scan_wait_num >= pcd_save_interval) {
            pcd_index++;
            string all_points_dir(string(string(ROOT_DIR) + "PCD/scans_") + to_string(pcd_index) + string(".pcd"));
            pcl::PCDWriter pcd_writer;
            cout << "current scan saved to /PCD/" << all_points_dir << endl;
            pcd_writer.writeBinary(all_points_dir, *pcl_wait_save);
            pcl_wait_save->clear();
            scan_wait_num = 0;
        }
    }
}

void LaserMappingNode::publish_frame_body(rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubLaserCloudFull_body) {
    int size = feats_undistort->points.size();
    CloudType::Ptr laserCloudIMUBody(new CloudType(size, 1));

    for (int i = 0; i < size; i++) {
        pointBodyLidarToIMU(&feats_undistort->points[i], &laserCloudIMUBody->points[i]);
    }

    sensor_msgs::msg::PointCloud2 laserCloudmsg;
    pcl::toROSMsg(*laserCloudIMUBody, laserCloudmsg);
    laserCloudmsg.header.stamp = get_ros_time(lidar_end_time);
    laserCloudmsg.header.frame_id = "body";
    pubLaserCloudFull_body->publish(laserCloudmsg);
}

void LaserMappingNode::publish_odometry(const rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pubOdomAftMapped,
                                        std::unique_ptr<tf2_ros::TransformBroadcaster>& tf_br) {
    odomAftMapped.header.frame_id = "camera_init";
    odomAftMapped.child_frame_id = "body";
    if (publish_odometry_without_downsample) {
        odomAftMapped.header.stamp = get_ros_time(time_current);
    } else {
        odomAftMapped.header.stamp = get_ros_time(lidar_end_time);
    }
    set_posestamp(odomAftMapped.pose.pose);
    pubOdomAftMapped->publish(odomAftMapped);

    auto P = kf_output.get_P();
    for (int i = 0; i < 6; i++) {
        int k = i < 3 ? i + 3 : i - 3;
        odomAftMapped.pose.covariance[i * 6 + 0] = P(k, 3);
        odomAftMapped.pose.covariance[i * 6 + 1] = P(k, 4);
        odomAftMapped.pose.covariance[i * 6 + 2] = P(k, 5);
        odomAftMapped.pose.covariance[i * 6 + 3] = P(k, 0);
        odomAftMapped.pose.covariance[i * 6 + 4] = P(k, 1);
        odomAftMapped.pose.covariance[i * 6 + 5] = P(k, 2);
    }

    geometry_msgs::msg::TransformStamped trans;
    trans.header.frame_id = "camera_init";
    trans.child_frame_id = "body";
    trans.transform.translation.x = odomAftMapped.pose.pose.position.x;
    trans.transform.translation.y = odomAftMapped.pose.pose.position.y;
    trans.transform.translation.z = odomAftMapped.pose.pose.position.z;
    trans.transform.rotation.w = odomAftMapped.pose.pose.orientation.w;
    trans.transform.rotation.x = odomAftMapped.pose.pose.orientation.x;
    trans.transform.rotation.y = odomAftMapped.pose.pose.orientation.y;
    trans.transform.rotation.z = odomAftMapped.pose.pose.orientation.z;
    tf_br->sendTransform(trans);
}

void LaserMappingNode::publish_path(rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr pubPath) {
    set_posestamp(msg_body_pose.pose);
    // msg_body_pose.header.stamp = ros::Time::now();
    msg_body_pose.header.stamp = get_ros_time(lidar_end_time);
    msg_body_pose.header.frame_id = "camera_init";

    // if path is too large, the rvis will crash
    static int jjj = 0;
    jjj++;
    if (jjj % 5 == 0) {
        path.poses.emplace_back(msg_body_pose);
        pubPath->publish(path);
    }
}

void LaserMappingNode::dump_lio_state_to_log(FILE* fp) {
    V3D rot_ang = SO3ToEuler(kf_output.x_.rot);

    fprintf(fp, "%lf ", Measures.lidar_beg_time - first_lidar_time);
    fprintf(fp, "%lf %lf %lf ", rot_ang(0), rot_ang(1), rot_ang(2));  // Angle

    fprintf(fp, "%lf %lf %lf ", kf_output.x_.pos(0), kf_output.x_.pos(1), kf_output.x_.pos(2));              // Pos
    fprintf(fp, "%lf %lf %lf ", 0.0, 0.0, 0.0);                                                              // omega
    fprintf(fp, "%lf %lf %lf ", kf_output.x_.vel(0), kf_output.x_.vel(1), kf_output.x_.vel(2));              // Vel
    fprintf(fp, "%lf %lf %lf ", 0.0, 0.0, 0.0);                                                              // Acc
    fprintf(fp, "%lf %lf %lf ", kf_output.x_.bg(0), kf_output.x_.bg(1), kf_output.x_.bg(2));                 // Bias_g
    fprintf(fp, "%lf %lf %lf ", kf_output.x_.ba(0), kf_output.x_.ba(1), kf_output.x_.ba(2));                 // Bias_a
    fprintf(fp, "%lf %lf %lf ", kf_output.x_.gravity(0), kf_output.x_.gravity(1), kf_output.x_.gravity(2));  // Bias_a

    fprintf(fp, "\r\n");
    fflush(fp);
}
