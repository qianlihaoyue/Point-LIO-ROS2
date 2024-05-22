#pragma once

#include "common_lib.h"

#define MAX_INI_COUNT (100)

/// *************IMU Process and undistortion
class ImuProcess {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    ImuProcess() : b_first_frame_(true), imu_need_init_(true), gravity_align_(false) {
        imu_en = true;
        init_iter_num = 1;
        mean_acc = V3D(0, 0, -1.0);
        mean_gyr = V3D(0, 0, 0);
    }

    // 处理包含 IMU 数据和激光雷达数据的 MeasureGroup
    void Process(const MeasureGroup &meas, PointCloudXYZI::Ptr cur_pcl_un_) {
        if (imu_en) {
            // 检查IMU和Lidar数据
            if (meas.imu.empty()) return;
            assert(meas.lidar != nullptr);
            // 初始化
            if (imu_need_init_) {
                /// The very first lidar frame
                IMU_init(meas, init_iter_num);
                // 有点多余
                imu_need_init_ = true;
                // 需要初始化多次
                if (init_iter_num > MAX_INI_COUNT) {
                    printf("IMU Initializing: %.1f %%", 100.0);
                    imu_need_init_ = false;
                    *cur_pcl_un_ = *(meas.lidar);
                }
                return;
            }
            // 初始化完成, 设置重力对齐,在初始化过程中,主程序在进行重力估计
            if (!gravity_align_) gravity_align_ = true;
            // 将雷达数据从meas转移出去
            *cur_pcl_un_ = *(meas.lidar);
            return;
        }
        // only lidar 不做处理
        else {
            if (!b_first_frame_) {
                if (!gravity_align_) gravity_align_ = true;
            } else {
                b_first_frame_ = false;
                return;
            }
            *cur_pcl_un_ = *(meas.lidar);
            return;
        }
    }

    void Set_init(Eigen::Vector3d &tmp_gravity, Eigen::Matrix3d &rot) {
        /** 1. initializing the gravity, gyro bias, acc and gyro covariance
         ** 2. normalize the acceleration measurenments to unit gravity **/
        // V3D tmp_gravity = - mean_acc / mean_acc.norm() * G_m_s2; // state_gravity;
        M3D hat_grav;
        hat_grav << 0.0, gravity_(2), -gravity_(1), -gravity_(2), 0.0, gravity_(0), gravity_(1), -gravity_(0), 0.0;
        double align_norm = (hat_grav * tmp_gravity).norm() / gravity_.norm() / gravity_.norm();
        double align_cos = gravity_.transpose() * tmp_gravity;
        align_cos = align_cos / gravity_.norm() / gravity_.norm();
        if (align_norm < 1e-6) {
            if (align_cos > 1e-6) {
                rot = Eye3d;
            } else {
                rot = -Eye3d;
            }
        } else {
            V3D align_angle = hat_grav * tmp_gravity / (hat_grav * tmp_gravity).norm() * acos(align_cos);
            rot = Exp(align_angle(0), align_angle(1), align_angle(2));
        }
    }

    void Reset() {
        printf("Reset ImuProcess");
        mean_acc = V3D(0, 0, -1.0);
        mean_gyr = V3D(0, 0, 0);
        imu_need_init_ = true;
        init_iter_num = 1;
    }

    bool imu_en;
    V3D mean_acc, gravity_;
    bool imu_need_init_ = true;
    bool b_first_frame_ = true;
    bool gravity_align_ = false;

private:
    V3D mean_gyr;
    int init_iter_num = 1;

    // 初始化重力、陀螺仪偏置、加速度和陀螺仪协方差； 将加速度测量值归一化为单位重力。
    void IMU_init(const MeasureGroup &meas, int &N) {
        /** 1. initializing the gravity, gyro bias, acc and gyro covariance
         ** 2. normalize the acceleration measurenments to unit gravity **/
        printf("IMU Initializing: %.1f %%", double(N) / MAX_INI_COUNT * 100);
        // 声明两个三维向量，用于保存当前加速度和陀螺仪测量值
        V3D cur_acc, cur_gyr;

        if (b_first_frame_) {
            // 复位参数
            Reset();
            N = 1;
            b_first_frame_ = false;
            // 从means中取值
            const auto &imu_acc = meas.imu.front()->linear_acceleration;
            const auto &gyr_acc = meas.imu.front()->angular_velocity;
            mean_acc << imu_acc.x, imu_acc.y, imu_acc.z;
            mean_gyr << gyr_acc.x, gyr_acc.y, gyr_acc.z;
        }
        // 累计存取100帧数据
        for (const auto &imu : meas.imu) {
            const auto &imu_acc = imu->linear_acceleration;
            const auto &gyr_acc = imu->angular_velocity;
            cur_acc << imu_acc.x, imu_acc.y, imu_acc.z;
            cur_gyr << gyr_acc.x, gyr_acc.y, gyr_acc.z;
            // 计算当前加速度和陀螺仪测量值的均值
            mean_acc += (cur_acc - mean_acc) / N;
            mean_gyr += (cur_gyr - mean_gyr) / N;

            N++;
        }
    }
};
