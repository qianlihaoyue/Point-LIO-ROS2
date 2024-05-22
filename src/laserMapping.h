#pragma once

#include "preprocess.h"
#include "Estimator.h"
#include "IMU_Processing.hpp"

#include <mutex>
#include <fstream>
#include <pcl/filters/voxel_grid.h>
#include <tf2_ros/transform_broadcaster.h>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <geometry_msgs/msg/transform.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <nav_msgs/msg/path.hpp>

class LaserMappingNode : public rclcpp::Node {
public:
    ////// process
    std::shared_ptr<ImuProcess> p_imu{new ImuProcess()};
    std::shared_ptr<Preprocess> p_pre{new Preprocess()};

    ////// ros
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_pcl_pc;
    rclcpp::Subscription<livox_ros_driver2::msg::CustomMsg>::SharedPtr sub_pcl_livox;
    rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr sub_imu;

    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubLaserCloudFullRes, pubLaserCloudFullRes_body, pubLaserCloudMap;
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pubOdomAftMapped;
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr pubPath;
    // rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr plane_pub;

    rclcpp::TimerBase::SharedPtr timer;
    std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster;

    nav_msgs::msg::Path path;
    nav_msgs::msg::Odometry odomAftMapped;
    geometry_msgs::msg::PoseStamped msg_body_pose;

    ////// variable
    int feats_down_size = 0;
    CloudType::Ptr feats_undistort{new CloudType()};
    CloudType::Ptr init_feats_world{new CloudType()};

    pcl::VoxelGrid<PointType> downSizeFilterSurf;
    pcl::VoxelGrid<PointType> downSizeFilterMap;

    MeasureGroup Measures;

    vector<BoxPointType> cub_needrm;
    BoxPointType LocalMap_Points;

    Eigen::Matrix<double, 24, 24> Q_input;
    Eigen::Matrix<double, 30, 30> Q_output;

    // once flg
    bool is_first_frame = true, flg_reset = false, flg_first_scan = true, init_map = false;
    // time
    double time_update_last = 0.0, time_current = 0.0, time_predict_last_const = 0.0;
    double lidar_end_time = 0.0, first_lidar_time = 0.0;
    double last_timestamp_lidar = -1.0, last_timestamp_imu = -1.0;

    ////// load parameters
    std::string lid_topic, imu_topic;
    bool space_down_sample, publish_odometry_without_downsample, prop_at_freq_of_imu;
    int init_map_size;
    double filter_size_surf_min, filter_size_map_min;
    double cube_len;
    float DET_RANGE;
    std::vector<double> gravity_init, gravity;
    std::vector<double> extrinT, extrinR;

    /////// data input
    deque<CloudType::Ptr> lidar_buffer;
    deque<double> time_buffer;
    deque<std::shared_ptr<const sensor_msgs::msg::Imu>> imu_deque;
    sensor_msgs::msg::Imu imu_last, imu_next;
    std::shared_ptr<const sensor_msgs::msg::Imu> imu_last_ptr;
    mutex mtx_buffer;

    ////// debug & timer & log
    bool runtime_pos_log, pcd_save_en, path_en, scan_pub_en, scan_body_pub_en;
    int time_log_counter = 0, scan_count = 0, frame_num = 0;
    int pcd_index = 0, pcd_save_interval;
    double aver_time_consu = 0, aver_time_icp = 0, aver_time_match = 0, aver_time_incre = 0, aver_time_solve = 0, aver_time_propag = 0;
    double T1[720000], s_plot[720000], s_plot2[720000], s_plot3[720000], s_plot11[720000];
    double match_time = 0, update_time = 0;
    FILE* fp;
    ofstream fout_pre, fout_out, fout_dbg, fout_imu_pbp;

    ////// imu
    bool imu_en, gravity_align;
    double imu_time_inte;
    double time_diff_lidar_to_imu = 0.0;

    CloudType::Ptr pcl_wait_pub{new CloudType(500000, 1)};
    CloudType::Ptr pcl_wait_save{new CloudType()};

    LaserMappingNode(const rclcpp::NodeOptions& options = rclcpp::NodeOptions()) : Node("laser_mapping", options) {
        readParameters();

        path.header.stamp = rclcpp::Time(lidar_end_time);  // this->get_clock()->now();
        path.header.frame_id = "camera_init";

        memset(point_selected_surf, true, sizeof(point_selected_surf));
        downSizeFilterSurf.setLeafSize(filter_size_surf_min, filter_size_surf_min, filter_size_surf_min);
        downSizeFilterMap.setLeafSize(filter_size_map_min, filter_size_map_min, filter_size_map_min);
        Lidar_T_wrt_IMU << VEC_FROM_ARRAY(extrinT);
        Lidar_R_wrt_IMU << MAT_FROM_ARRAY(extrinR);

        p_imu->imu_en = imu_en;

        kf_output.init_dyn_share_modified_2h(get_f_output, df_dx_output, h_model_output, h_model_IMU_output);
        Eigen::Matrix<double, 30, 30> P_init_output = MD(30, 30)::Identity() * 0.01;
        P_init_output.block<3, 3>(21, 21) = MD(3, 3)::Identity() * 0.0001;
        P_init_output.block<6, 6>(6, 6) = MD(6, 6)::Identity() * 0.0001;
        P_init_output.block<6, 6>(24, 24) = MD(6, 6)::Identity() * 0.001;
        kf_output.change_P(P_init_output);

        Q_input = process_noise_cov_input();
        Q_output = process_noise_cov_output();

        /*** debug record ***/
        fp = fopen(DEBUG_FILE_DIR("pos_log.txt").c_str(), "w");
        fout_out.open(DEBUG_FILE_DIR("mat_out.txt"), ios::out);
        fout_imu_pbp.open(DEBUG_FILE_DIR("imu_pbp.txt"), ios::out);
        if (fout_out && fout_imu_pbp)
            cout << "~~~~" << ROOT_DIR << " file opened" << endl;
        else
            cout << "~~~~" << ROOT_DIR << " doesn't exist" << endl;

        /*** ROS subscribe initialization ***/
        if (p_pre->lidar_type == AVIA)
            sub_pcl_livox = this->create_subscription<livox_ros_driver2::msg::CustomMsg>(
                lid_topic, 20, std::bind(&LaserMappingNode::livox_pcl_cbk, this, std::placeholders::_1));
        else
            sub_pcl_pc = this->create_subscription<sensor_msgs::msg::PointCloud2>(lid_topic, rclcpp::SensorDataQoS(),
                                                                                  std::bind(&LaserMappingNode::standard_pcl_cbk, this, std::placeholders::_1));

        sub_imu = this->create_subscription<sensor_msgs::msg::Imu>(imu_topic, 10, std::bind(&LaserMappingNode::imu_cbk, this, std::placeholders::_1));
        pubLaserCloudFullRes = this->create_publisher<sensor_msgs::msg::PointCloud2>("/cloud_registered", 20);
        pubLaserCloudFullRes_body = this->create_publisher<sensor_msgs::msg::PointCloud2>("/cloud_registered_body", 20);
        pubLaserCloudMap = this->create_publisher<sensor_msgs::msg::PointCloud2>("/Laser_map", 20);
        pubOdomAftMapped = this->create_publisher<nav_msgs::msg::Odometry>("/aft_mapped_to_init", 20);
        pubPath = this->create_publisher<nav_msgs::msg::Path>("/path", 20);
        // plane_pub = this->create_publisher<visualization_msgs::msg::Marker>("/planner_normal", 20);

        tf_broadcaster = std::make_unique<tf2_ros::TransformBroadcaster>(*this);
        timer = rclcpp::create_timer(this, this->get_clock(), std::chrono::milliseconds(10), std::bind(&LaserMappingNode::timer_callback, this));
    }

    ~LaserMappingNode() {
        fout_out.close();
        fout_imu_pbp.close();
        fclose(fp);
    }

private:
    template <typename MsgType>
    void lidar_pcl_cbk(const MsgType& msg);
    void standard_pcl_cbk(const sensor_msgs::msg::PointCloud2::UniquePtr msg) { lidar_pcl_cbk<sensor_msgs::msg::PointCloud2::UniquePtr>(msg); }
    void livox_pcl_cbk(const livox_ros_driver2::msg::CustomMsg::UniquePtr msg) { lidar_pcl_cbk<livox_ros_driver2::msg::CustomMsg::UniquePtr>(msg); }
    void imu_cbk(const sensor_msgs::msg::Imu::UniquePtr msg_in);
    bool sync_packages(MeasureGroup& meas);

    void timer_callback();
    void map_incremental();
    void lasermap_fov_segment();
    void publish_init_kdtree(rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubLaserCloudFullRes);

    ////// laserMappingUtility.cpp
    void readParameters();
    void pointBodyLidarToIMU(PointType const* const pi, PointType* const po);
    void publish_frame_world(rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubLaserCloudFullRes);
    void publish_frame_body(rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubLaserCloudFull_body);
    void publish_odometry(const rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pubOdomAftMapped, std::unique_ptr<tf2_ros::TransformBroadcaster>& tf_br);
    void publish_path(rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr pubPath);
    void dump_lio_state_to_log(FILE* fp);

    template <typename T>
    void set_posestamp(T& out);

    template <typename T>
    void declare_and_get_parameter(const std::string& name, T& variable, const T& default_value) {
        this->declare_parameter<T>(name, default_value);
        this->get_parameter(name, variable);
    }
};
