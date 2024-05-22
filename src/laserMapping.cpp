#include "laserMapping.h"
#include <csignal>
#include <rclcpp/logging.hpp>
#include <unistd.h>

void LaserMappingNode::timer_callback() {
    if (sync_packages(Measures)) {
        if (flg_first_scan) {
            first_lidar_time = Measures.lidar_beg_time;
            flg_first_scan = false;
            cout << "first lidar time" << first_lidar_time << endl;
            return;
        }

        if (flg_reset) {
            printf("reset when rosbag play back");
            p_imu->Reset();
            flg_reset = false;
            return;
        }

        double t0, t1, t2, t3, t4, t5, match_start, solve_start;
        match_time = update_time = 0;
        t0 = omp_get_wtime();
        // IMU初始化
        p_imu->Process(Measures, feats_undistort);

        if (feats_undistort->empty() || feats_undistort == NULL) return;

        if (imu_en) {
            if (p_imu->imu_need_init_) return;

            // 默认gravity_align_为false,但通过Process转true, 所以下面代码只执行一次估计
            if (!p_imu->gravity_align_) {
                cout << "g_align ";
                // 遍历, 直到找到一个 时间戳大于激光雷达测量开始时间lidar_beg_time
                while (Measures.lidar_beg_time > get_time_sec(imu_next.header.stamp)) {
                    // imu_last 和 imu_next 分别保存了上一帧和下一帧 IMU 数据
                    imu_last = imu_next;
                    imu_next = *(imu_deque.front());
                    imu_deque.pop_front();
                }
                // 计算重力矢量估计值: mean_acc乘以重力加速度常数 G_m_s2，除以加速度范数acc_norm, 将结果分别赋值，注意方向取反
                state_out.gravity = -1 * p_imu->mean_acc * G_m_s2 / acc_norm;
                state_out.acc = p_imu->mean_acc * G_m_s2 / acc_norm;
                // 如果启用了重力对齐, 一般启用
                if (gravity_align) {
                    Eigen::Matrix3d rot_init;
                    p_imu->gravity_ << VEC_FROM_ARRAY(gravity);  // 格式转换
                    // 基于输入的重力矢量，计算初始旋转矩阵 rot_init
                    p_imu->Set_init(state_out.gravity, rot_init);
                    state_out.rot.normalize();
                    // 计算并更新 state_out.acc，即将初始旋转矩阵的转置乘以重力矢量，再取反
                    state_out.acc = -rot_init.transpose() * state_out.gravity;
                }
                // 使用更新后的状态值 state_out 更新卡尔曼滤波器的状态
                kf_output.change_x(state_out);
            }
        } else {
            if (!p_imu->gravity_align_) {
                state_out.gravity << VEC_FROM_ARRAY(gravity_init);
                state_out.acc << VEC_FROM_ARRAY(gravity_init);
                state_out.acc *= -1;
            }
        }

        /*** Segment the map in lidar FOV ***/
        lasermap_fov_segment();
        /*** downsample the feature points in a scan ***/
        t1 = omp_get_wtime();
        // 降采样,默认为true
        if (space_down_sample) {
            // 输入点云为 feats_undistort，滤波结果为 feats_down_body
            downSizeFilterSurf.setInputCloud(feats_undistort);
            downSizeFilterSurf.filter(*feats_down_body);
            // 根据时间对点云进行排序
            sort(feats_down_body->points.begin(), feats_down_body->points.end(), time_list);
        } else {
            feats_down_body = Measures.lidar;
            sort(feats_down_body->points.begin(), feats_down_body->points.end(), time_list);
        }
        // 进行时间压缩处理，并将结果赋值给 time_seq
        time_seq = time_compressing<int>(feats_down_body);
        feats_down_size = feats_down_body->points.size();

        /*** initialize the map kdtree ***/
        if (!init_map) {
            if (ikdtree.Root_Node == nullptr) {
                ikdtree.set_downsample_param(filter_size_map_min);
            }
            feats_down_world->resize(feats_down_size);
            for (int i = 0; i < feats_down_size; i++) {
                pointBodyToWorld(&(feats_down_body->points[i]), &(feats_down_world->points[i]));
            }
            for (size_t i = 0; i < feats_down_world->size(); i++) {
                init_feats_world->points.emplace_back(feats_down_world->points[i]);
            }
            if (init_feats_world->size() < init_map_size) return;
            ikdtree.Build(init_feats_world->points);
            init_map = true;
            publish_init_kdtree(pubLaserCloudMap);  //(pubLaserCloudFullRes);
            return;
        }

        /*** ICP and Kalman filter update ***/
        t2 = omp_get_wtime();
        normvec->resize(feats_down_size);
        feats_down_world->resize(feats_down_size);
        Nearest_Points.resize(feats_down_size);
        /*** iterated state estimation ***/
        crossmat_list.reserve(feats_down_size);
        pbody_list.reserve(feats_down_size);
        // 格式转换 与 计算反对称矩阵
        for (size_t i = 0; i < feats_down_body->size(); i++) {
            // 点云格式转换,并存到pbody_list
            V3D point_this(feats_down_body->points[i].x, feats_down_body->points[i].y, feats_down_body->points[i].z);
            pbody_list[i] = point_this;
            // 用固定外参对点云进行旋转平移 (来自配置文件)
            point_this = Lidar_R_wrt_IMU * point_this + Lidar_T_wrt_IMU;
            // 计算point_this的反对称矩阵（也称为叉乘矩阵）,添加到crossmat_list中
            M3D point_crossmat;
            point_crossmat << SKEW_SYM_MATRX(point_this);
            crossmat_list[i] = point_crossmat;
        }

        // 逐点更新,通过与IMU数据进行融合，对当前点云进行运动补偿
        bool imu_upda_cov = false;
        effct_feat_num = 0;
        /**** point by point update ****/
        // 初始化时间
        double pcl_beg_time = Measures.lidar_beg_time;
        idx = -1;  // 修正
        // 遍历时间分组
        for (k = 0; k < time_seq.size(); k++) {
            // 是具体一个点, 每组的第一个点
            PointType& point_body = feats_down_body->points[idx + time_seq[k]];
            // 表示当前时间。用于判断是否有新的 IMU 数据到达
            time_current = point_body.curvature / 1000.0 + pcl_beg_time;

            if (is_first_frame) {
                if (imu_en) {
                    // 如果是第一帧且启用IMU，将IMU的线性加速度和角速度赋值给acc_avr和angvel_avr
                    while (time_current > get_time_sec(imu_next.header.stamp)) {
                        imu_last = imu_next;
                        imu_next = *(imu_deque.front());
                        imu_deque.pop_front();
                    }

                    angvel_avr << imu_last.angular_velocity.x, imu_last.angular_velocity.y, imu_last.angular_velocity.z;
                    acc_avr << imu_last.linear_acceleration.x, imu_last.linear_acceleration.y, imu_last.linear_acceleration.z;
                }
                is_first_frame = false;
                imu_upda_cov = true;
                time_update_last = time_current;
                time_predict_last_const = time_current;
            }
            if (imu_en) {
                // 检查是否有新IMU数据到来,即新的 IMU 数据的时间戳是否大于当前时间
                bool imu_comes = time_current > get_time_sec(imu_next.header.stamp);
                while (imu_comes) {
                    // 如果有新数据，则需要进行协方差更新
                    imu_upda_cov = true;
                    // 提取新的IMU线性加速度和角速度
                    angvel_avr << imu_next.angular_velocity.x, imu_next.angular_velocity.y, imu_next.angular_velocity.z;
                    acc_avr << imu_next.linear_acceleration.x, imu_next.linear_acceleration.y, imu_next.linear_acceleration.z;

                    /*** covariance update ***/
                    imu_last = imu_next;
                    imu_next = *(imu_deque.front());
                    imu_deque.pop_front();
                    // 计算当前点的时间与上一次预测时间的差值dt，并根据时间差进行状态预测,(time_predict_last_const表示上一次执行状态预测的时间)
                    double dt = get_time_sec(imu_last.header.stamp) - time_predict_last_const;
                    // 新 IMU 到达时，根据当前的状态估计、输入信号和时间差 dt 更新状态估计。只更新状态（执行状态预测），不更新协方差矩阵。
                    kf_output.predict(dt, Q_output, input_in, true, false);
                    time_predict_last_const = get_time_sec(imu_last.header.stamp);  // big problem
                    imu_comes = time_current > get_time_sec(imu_next.header.stamp);
                    // if (!imu_comes)
                    {
                        // 计算协方差更新时间差dt_cov
                        double dt_cov = get_time_sec(imu_last.header.stamp) - time_update_last;
                        // 如果dt_cov大于0, 表示需要执行协方差更新
                        if (dt_cov > 0.0) {
                            // time_update_last：表示上一次执行协方差矩阵更新的时间
                            time_update_last = get_time_sec(imu_last.header.stamp);

                            // 新 IMU 到达后，根据时间差 dt_cov 更新协方差矩阵，不更新状态预测
                            kf_output.predict(dt_cov, Q_output, input_in, false, true);

                            kf_output.update_iterated_dyn_share_IMU();
                        }
                    }
                }
            }

            double dt = time_current - time_predict_last_const;
            // 默认不执行
            if (!prop_at_freq_of_imu) {
                double dt_cov = time_current - time_update_last;
                if (dt_cov > 0.0) {
                    kf_output.predict(dt_cov, Q_output, input_in, false, true);
                    time_update_last = time_current;
                }
            }
            // 在循环结束后，对 EKF 进行一次额外的状态预测，以便在下一次 IMU 数据到达前保持状态估计的实时性。只更新状态，不更新协方差矩阵。
            kf_output.predict(dt, Q_output, input_in, true, false);
            time_predict_last_const = time_current;

            double t_update_start = omp_get_wtime();
            // 例外情况处理, 点太少
            if (feats_down_size < 1) {
                printf("No point, skip this scan!\n");
                idx += time_seq[k];
                continue;
            }
            // 状态更新
            if (!kf_output.update_iterated_dyn_share_modified()) {
                idx = idx + time_seq[k];
                continue;
            }

            // 默认true
            if (prop_at_freq_of_imu) {
                double dt_cov = time_current - time_update_last;
                if (!imu_en && (dt_cov >= imu_time_inte)) {
                    kf_output.predict(dt_cov, Q_output, input_in, false, true);
                    imu_upda_cov = false;
                    time_update_last = time_current;
                }
            }
            // 一般false, pub全部里程计
            if (publish_odometry_without_downsample) {
                publish_odometry(pubOdomAftMapped, tf_broadcaster);
                if (runtime_pos_log) {
                    state_out = kf_output.x_;
                    V3D euler_cur = SO3ToEuler(state_out.rot);
                    fout_out << setw(20) << Measures.lidar_beg_time - first_lidar_time << " " << euler_cur.transpose() << " " << state_out.pos.transpose()
                             << " " << state_out.vel.transpose() << " " << state_out.omg.transpose() << " " << state_out.acc.transpose() << " "
                             << state_out.gravity.transpose() << " " << state_out.bg.transpose() << " " << state_out.ba.transpose() << " "
                             << feats_undistort->points.size() << endl;
                }
            }
            // 遍历该组的点, 逐个转换
            for (int j = 0; j < time_seq[k]; j++) {
                PointType& point_body_j = feats_down_body->points[idx + j + 1];
                PointType& point_world_j = feats_down_world->points[idx + j + 1];
                pointBodyToWorld(&point_body_j, &point_world_j);
            }

            update_time += omp_get_wtime() - t_update_start;
            idx += time_seq[k];  // 下一组
        }
        t3 = omp_get_wtime();

        if (!publish_odometry_without_downsample) publish_odometry(pubOdomAftMapped, tf_broadcaster);

        /*** add the feature points to map kdtree ***/
        if (feats_down_size > 4) map_incremental();

        t5 = omp_get_wtime();
        /******* Publish points *******/
        publish_path(pubPath);
        if (scan_pub_en || pcd_save_en) publish_frame_world(pubLaserCloudFullRes);
        if (scan_pub_en && scan_body_pub_en) publish_frame_body(pubLaserCloudFullRes_body);

        /*** Debug variables Logging ***/
        if (runtime_pos_log) {
            frame_num++;
            aver_time_consu = aver_time_consu * (frame_num - 1) / frame_num + (t5 - t0) / frame_num;
            { aver_time_icp = aver_time_icp * (frame_num - 1) / frame_num + update_time / frame_num; }
            aver_time_match = aver_time_match * (frame_num - 1) / frame_num + (match_time) / frame_num;
            T1[time_log_counter] = Measures.lidar_beg_time;
            s_plot[time_log_counter] = t5 - t0;
            s_plot2[time_log_counter] = feats_undistort->points.size();
            s_plot3[time_log_counter] = aver_time_consu;
            time_log_counter++;

            RCLCPP_INFO(this->get_logger(), "[tim]: IMU+Map+match: %0.3f ICP: %0.3f ave icp: %0.3f ave total: %0.3f", (t1 - t0 + aver_time_match) * 1000.0,
                        (t3 - t1) * 1000.0, aver_time_icp * 1000.0, aver_time_consu * 1000.0);
            if (!publish_odometry_without_downsample) {
                state_out = kf_output.x_;
                V3D euler_cur = SO3ToEuler(state_out.rot);
                fout_out << setw(20) << Measures.lidar_beg_time - first_lidar_time << " " << euler_cur.transpose() << " " << state_out.pos.transpose() << " "
                         << state_out.vel.transpose() << " " << state_out.omg.transpose() << " " << state_out.acc.transpose() << " "
                         << state_out.gravity.transpose() << " " << state_out.bg.transpose() << " " << state_out.ba.transpose() << " "
                         << feats_undistort->points.size() << endl;
            }
            dump_lio_state_to_log(fp);
        }
    }
}

condition_variable sig_buffer;
bool flg_exit = false;
void SigHandle(int sig) {
    flg_exit = true;
    printf("catch sig %d", sig);
    sig_buffer.notify_all();
    rclcpp::shutdown();
}

template <typename MsgType>
void LaserMappingNode::lidar_pcl_cbk(const MsgType& msg) {
    mtx_buffer.lock();
    scan_count++;
    double preprocess_start_time = omp_get_wtime();
    if (get_time_sec(msg->header.stamp) < last_timestamp_lidar) {
        std::cerr << "lidar loop back, clear buffer" << std::endl;
        // lidar_buffer.shrink_to_fit();
        mtx_buffer.unlock();
        sig_buffer.notify_all();
        return;
    }

    last_timestamp_lidar = get_time_sec(msg->header.stamp);

    CloudType::Ptr ptr(new CloudType());
    p_pre->process(msg, ptr);

    // if (cut_frame)  if (con_frame)
    // 一般不需要 https://github.com/hku-mars/Point-LIO

    lidar_buffer.emplace_back(ptr);
    time_buffer.emplace_back(last_timestamp_lidar);

    s_plot11[scan_count] = omp_get_wtime() - preprocess_start_time;
    mtx_buffer.unlock();
    sig_buffer.notify_all();
}

void LaserMappingNode::imu_cbk(const sensor_msgs::msg::Imu::UniquePtr msg_in) {
    std::shared_ptr<sensor_msgs::msg::Imu> msg(new sensor_msgs::msg::Imu(*msg_in));

    msg->header.stamp = get_ros_time(get_time_sec(msg_in->header.stamp) - time_diff_lidar_to_imu);
    double timestamp = get_time_sec(msg->header.stamp);
    mtx_buffer.lock();

    if (timestamp < last_timestamp_imu) {
        std::cerr << "imu loop back, clear deque" << std::endl;
        // imu_deque.shrink_to_fit();
        mtx_buffer.unlock();
        sig_buffer.notify_all();
        return;
    }

    imu_deque.emplace_back(msg);
    last_timestamp_imu = timestamp;
    mtx_buffer.unlock();
    sig_buffer.notify_all();
}

bool LaserMappingNode::sync_packages(MeasureGroup& meas) {
    static bool lidar_pushed = false;
    // only lidar
    if (imu_en == false) {
        if (!lidar_buffer.empty()) {
            meas.lidar = lidar_buffer.front();
            meas.lidar_beg_time = time_buffer.front();
            time_buffer.pop_front();
            lidar_buffer.pop_front();
            if (meas.lidar->points.size() < 1) {
                cout << "lose lidar" << std::endl;
                return false;
            }
            double end_time = meas.lidar->points.back().curvature;
            for (auto pt : meas.lidar->points) {
                if (pt.curvature > end_time) end_time = pt.curvature;
            }
            lidar_end_time = meas.lidar_beg_time + end_time / double(1000);
            meas.lidar_last_time = lidar_end_time;
            return true;
        }
        return false;
    }

    // lidar & imu
    if (lidar_buffer.empty() || imu_deque.empty()) return false;

    /*** push a lidar scan ***/
    if (!lidar_pushed) {
        // 将 buffer 第一帧取出 meas.lidar
        meas.lidar = lidar_buffer.front();
        // 异常帧
        if (meas.lidar->points.size() < 1) {
            cout << "lose lidar" << endl;
            lidar_buffer.pop_front();
            time_buffer.pop_front();
            return false;
        }
        // 取第一帧时间
        meas.lidar_beg_time = time_buffer.front();
        double end_time = meas.lidar->points.back().curvature;
        for (auto pt : meas.lidar->points) {
            if (pt.curvature > end_time) end_time = pt.curvature;
        }
        lidar_end_time = meas.lidar_beg_time + end_time / double(1000);
        meas.lidar_last_time = lidar_end_time;
        lidar_pushed = true;
    }

    if (last_timestamp_imu < lidar_end_time) return false;

    /*** push imu data, and pop from imu buffer ***/
    if (p_imu->imu_need_init_) {
        double imu_time = get_time_sec(imu_deque.front()->header.stamp);
        meas.imu.shrink_to_fit();
        while ((!imu_deque.empty()) && (imu_time < lidar_end_time)) {
            imu_time = get_time_sec(imu_deque.front()->header.stamp);
            if (imu_time > lidar_end_time) break;
            meas.imu.emplace_back(imu_deque.front());
            imu_last = imu_next;
            imu_last_ptr = imu_deque.front();
            imu_next = *(imu_deque.front());
            imu_deque.pop_front();
        }
    } else if (!init_map) {
        double imu_time = get_time_sec(imu_deque.front()->header.stamp);
        meas.imu.shrink_to_fit();
        meas.imu.emplace_back(imu_last_ptr);

        while ((!imu_deque.empty()) && (imu_time < lidar_end_time)) {
            imu_time = get_time_sec(imu_deque.front()->header.stamp);
            if (imu_time > lidar_end_time) break;
            meas.imu.emplace_back(imu_deque.front());
            imu_last = imu_next;
            imu_last_ptr = imu_deque.front();
            imu_next = *(imu_deque.front());
            imu_deque.pop_front();
        }
    }

    lidar_buffer.pop_front();
    time_buffer.pop_front();
    lidar_pushed = false;
    return true;
}

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);

    signal(SIGINT, SigHandle);

    rclcpp::spin(std::make_shared<LaserMappingNode>());

    if (rclcpp::ok()) rclcpp::shutdown();
    //--------------------------save map-----------------------------------
    // if (pcl_wait_save->size() > 0 && pcd_save_en) {
    //     string file_name = string("scans.pcd");
    //     string all_points_dir(string(string(ROOT_DIR) + "PCD/") + file_name);
    //     pcl::PCDWriter pcd_writer;
    //     pcd_writer.writeBinary(all_points_dir, *pcl_wait_save);
    // }
    return 0;
}
