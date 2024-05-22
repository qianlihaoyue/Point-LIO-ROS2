// #include <../include/IKFoM/IKFoM_toolkit/esekfom/esekfom.hpp>
#include "Estimator.h"

double laser_point_cov, acc_norm;
double vel_cov, acc_cov_input, gyr_cov_input;
double gyr_cov_output, acc_cov_output, b_gyr_cov, b_acc_cov;
double imu_meas_acc_cov, imu_meas_omg_cov;
double match_s, satu_acc, satu_gyro;
bool check_satu;
float plane_thr;

PointCloudXYZI::Ptr normvec(new PointCloudXYZI(100000, 1));
std::vector<int> time_seq;
PointCloudXYZI::Ptr feats_down_body(new PointCloudXYZI());
PointCloudXYZI::Ptr feats_down_world(new PointCloudXYZI());
std::vector<V3D> pbody_list;
std::vector<PointVector> Nearest_Points;
KD_TREE<PointType> ikdtree;
std::vector<float> pointSearchSqDis(NUM_MATCH_POINTS);
bool point_selected_surf[100000] = {0};
std::vector<M3D> crossmat_list;
int effct_feat_num = 0;
int k;
int idx;
esekfom::esekf<state_output, 30, input_ikfom> kf_output;
state_output state_out;
input_ikfom input_in;
V3D angvel_avr, acc_avr;

V3D Lidar_T_wrt_IMU(Zero3d);
M3D Lidar_R_wrt_IMU(Eye3d);

typedef MTK::vect<3, double> vect3;
typedef MTK::SO3<double> SO3;
typedef MTK::S2<double, 98090, 10000, 1> S2;
typedef MTK::vect<1, double> vect1;
typedef MTK::vect<2, double> vect2;

Eigen::Matrix<double, 24, 24> process_noise_cov_input() {
    Eigen::Matrix<double, 24, 24> cov;
    cov.setZero();
    cov.block<3, 3>(3, 3).diagonal() << gyr_cov_input, gyr_cov_input, gyr_cov_input;
    cov.block<3, 3>(12, 12).diagonal() << acc_cov_input, acc_cov_input, acc_cov_input;
    cov.block<3, 3>(15, 15).diagonal() << b_gyr_cov, b_gyr_cov, b_gyr_cov;
    cov.block<3, 3>(18, 18).diagonal() << b_acc_cov, b_acc_cov, b_acc_cov;
    return cov;
}

Eigen::Matrix<double, 30, 30> process_noise_cov_output() {
    Eigen::Matrix<double, 30, 30> cov;
    cov.setZero();
    cov.block<3, 3>(12, 12).diagonal() << vel_cov, vel_cov, vel_cov;
    cov.block<3, 3>(15, 15).diagonal() << gyr_cov_output, gyr_cov_output, gyr_cov_output;
    cov.block<3, 3>(18, 18).diagonal() << acc_cov_output, acc_cov_output, acc_cov_output;
    cov.block<3, 3>(24, 24).diagonal() << b_gyr_cov, b_gyr_cov, b_gyr_cov;
    cov.block<3, 3>(27, 27).diagonal() << b_acc_cov, b_acc_cov, b_acc_cov;
    return cov;
}

Eigen::Matrix<double, 30, 1> get_f_output(state_output &s, const input_ikfom &in) {
    Eigen::Matrix<double, 30, 1> res = Eigen::Matrix<double, 30, 1>::Zero();
    vect3 a_inertial = s.rot * s.acc;
    for (int i = 0; i < 3; i++) {
        res(i) = s.vel[i];
        res(i + 3) = s.omg[i];
        res(i + 12) = a_inertial[i] + s.gravity[i];
    }
    return res;
}

Eigen::Matrix<double, 30, 30> df_dx_output(state_output &s, const input_ikfom &in) {
    Eigen::Matrix<double, 30, 30> cov = Eigen::Matrix<double, 30, 30>::Zero();
    cov.template block<3, 3>(0, 12) = Eigen::Matrix3d::Identity();
    cov.template block<3, 3>(12, 3) = -s.rot.normalized().toRotationMatrix() * MTK::hat(s.acc);
    cov.template block<3, 3>(12, 18) = s.rot.normalized().toRotationMatrix();
    cov.template block<3, 3>(12, 21) = Eigen::Matrix3d::Identity();  // grav_matrix;
    cov.template block<3, 3>(3, 15) = Eigen::Matrix3d::Identity();
    return cov;
}

// 将 SO3 旋转（四元数表示）转换为欧拉角
vect3 SO3ToEuler(const SO3 &orient) {
    Eigen::Matrix<double, 3, 1> _ang;
    // 将输入的 orient 参数转换为四元数表示
    Eigen::Vector4d q_data = orient.coeffs().transpose();
    // scalar w=orient.coeffs[3], x=orient.coeffs[0], y=orient.coeffs[1], z=orient.coeffs[2];
    double sqw = q_data[3] * q_data[3];
    double sqx = q_data[0] * q_data[0];
    double sqy = q_data[1] * q_data[1];
    double sqz = q_data[2] * q_data[2];
    double unit = sqx + sqy + sqz + sqw;  // if normalized is one, otherwise is correction factor
    // 计算一个测试值 test，用于检查输入旋转是否靠近极点（北极或南极）
    double test = q_data[3] * q_data[1] - q_data[2] * q_data[0];

    if (test > 0.49999 * unit) {  // singularity at north pole
        _ang << 2 * std::atan2(q_data[0], q_data[3]), M_PI / 2, 0;
        double temp[3] = {_ang[0] * 57.3, _ang[1] * 57.3, _ang[2] * 57.3};
        vect3 euler_ang(temp, 3);
        return euler_ang;
    }
    if (test < -0.49999 * unit) {  // singularity at south pole
        _ang << -2 * std::atan2(q_data[0], q_data[3]), -M_PI / 2, 0;
        double temp[3] = {_ang[0] * 57.3, _ang[1] * 57.3, _ang[2] * 57.3};
        vect3 euler_ang(temp, 3);
        return euler_ang;
    }
    // 如果旋转没有靠近极点，计算欧拉角并返回
    _ang << std::atan2(2 * q_data[0] * q_data[3] + 2 * q_data[1] * q_data[2], -sqx - sqy + sqz + sqw), std::asin(2 * test / unit),
        std::atan2(2 * q_data[2] * q_data[3] + 2 * q_data[1] * q_data[0], sqx - sqy - sqz + sqw);
    double temp[3] = {_ang[0] * 57.3, _ang[1] * 57.3, _ang[2] * 57.3};
    vect3 euler_ang(temp, 3);
    return euler_ang;
}

// 根据特征点和对应的平面参数求解 观测矩阵H
void h_model_output(state_output &s, esekfom::dyn_share_modified<double> &ekfom_data) {
    bool match_in_map = false;
    VF(4) pabcd;
    pabcd.setZero();

    normvec->resize(time_seq[k]);
    int effect_num_k = 0;  // 记录有效特征点数量
    for (int j = 0; j < time_seq[k]; j++) {
        // 获取点云数据中的点，将其从机器人体坐标系转换到世界坐标系。??
        PointType &point_body_j = feats_down_body->points[idx + j + 1];
        PointType &point_world_j = feats_down_world->points[idx + j + 1];
        pointBodyToWorld(&point_body_j, &point_world_j);

        V3D p_body = pbody_list[idx + j + 1];
        V3D p_world;
        p_world << point_world_j.x, point_world_j.y, point_world_j.z;
        {
            // 搜索最近邻,并拟合平面
            auto &points_near = Nearest_Points[idx + j + 1];

            ikdtree.Nearest_Search(point_world_j, NUM_MATCH_POINTS, points_near, pointSearchSqDis);

            if ((points_near.size() < NUM_MATCH_POINTS) || pointSearchSqDis[NUM_MATCH_POINTS - 1] > 5) {
                point_selected_surf[idx + j + 1] = false;
            } else {
                point_selected_surf[idx + j + 1] = false;
                if (esti_plane(pabcd, points_near, plane_thr))  //(planeValid)
                {
                    float pd2 = pabcd(0) * point_world_j.x + pabcd(1) * point_world_j.y + pabcd(2) * point_world_j.z + pabcd(3);

                    if (p_body.norm() > match_s * pd2 * pd2) {
                        point_selected_surf[idx + j + 1] = true;
                        normvec->points[j].x = pabcd(0);
                        normvec->points[j].y = pabcd(1);
                        normvec->points[j].z = pabcd(2);
                        normvec->points[j].intensity = pabcd(3);
                        effect_num_k++;
                    }
                }
            }
        }
    }
    // 如果有效特征点数量为零，则将 ekfom_data.valid 设置为 false
    if (effect_num_k == 0) {
        ekfom_data.valid = false;
        return;
    }
    // 外置参数0.1
    ekfom_data.M_Noise = laser_point_cov;
    ekfom_data.h_x = Eigen::MatrixXd::Zero(effect_num_k, 12);
    ekfom_data.z.resize(effect_num_k);
    int m = 0;
    for (int j = 0; j < time_seq[k]; j++) {
        // 遍历有效特征点
        if (point_selected_surf[idx + j + 1]) {
            // 计算法向量 norm_vec，并将其存储在 ekfom_data.h_x
            V3D norm_vec(normvec->points[j].x, normvec->points[j].y, normvec->points[j].z);

            M3D point_crossmat = crossmat_list[idx + j + 1];
            V3D C(s.rot.conjugate() * norm_vec);
            V3D A(point_crossmat * C);
            // V3D A(point_crossmat * state.rot_end.transpose() * norm_vec);
            ekfom_data.h_x.block<1, 12>(m, 0) << norm_vec(0), norm_vec(1), norm_vec(2), VEC_FROM_ARRAY(A), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
            // 计算测量值 z，并将其存储在 ekfom_data.z
            ekfom_data.z(m) = -norm_vec(0) * feats_down_world->points[idx + j + 1].x - norm_vec(1) * feats_down_world->points[idx + j + 1].y -
                              norm_vec(2) * feats_down_world->points[idx + j + 1].z - normvec->points[j].intensity;
            m++;
        }
    }
    effct_feat_num += effect_num_k;
}

// 处理 IMU 输出数据，计算 IMU 测量误差，并检查 IMU 饱和状态
void h_model_IMU_output(state_output &s, esekfom::dyn_share_modified<double> &ekfom_data) {
    // 将 satu_check 数组的所有元素初始化为 false
    std::memset(ekfom_data.satu_check, false, 6);
    // 计算陀螺仪和加速度计的测量误差，分别存储在 z_IMU 的前三个元素和后三个元素中
    ekfom_data.z_IMU.block<3, 1>(0, 0) = angvel_avr - s.omg - s.bg;
    ekfom_data.z_IMU.block<3, 1>(3, 0) = acc_avr * G_m_s2 / acc_norm - s.acc - s.ba;
    // 设置 R_IMU 为陀螺仪和加速度计的测量协方差矩阵
    ekfom_data.R_IMU << imu_meas_omg_cov, imu_meas_omg_cov, imu_meas_omg_cov, imu_meas_acc_cov, imu_meas_acc_cov, imu_meas_acc_cov;
    // 饱和检查,默认配置启用
    if (check_satu) {
        // 分别检查三轴陀螺仪是否接近饱和值（satu_gyro * 0.99），如果接近，则将对应的 satu_check 设为 z_IMU 设为 0
        if (fabs(angvel_avr(0)) >= 0.99 * satu_gyro) {
            ekfom_data.satu_check[0] = true;
            ekfom_data.z_IMU(0) = 0.0;
        }

        if (fabs(angvel_avr(1)) >= 0.99 * satu_gyro) {
            ekfom_data.satu_check[1] = true;
            ekfom_data.z_IMU(1) = 0.0;
        }

        if (fabs(angvel_avr(2)) >= 0.99 * satu_gyro) {
            ekfom_data.satu_check[2] = true;
            ekfom_data.z_IMU(2) = 0.0;
        }

        // 三轴加速度计同理
        if (fabs(acc_avr(0)) >= 0.99 * satu_acc) {
            ekfom_data.satu_check[3] = true;
            ekfom_data.z_IMU(3) = 0.0;
        }

        if (fabs(acc_avr(1)) >= 0.99 * satu_acc) {
            ekfom_data.satu_check[4] = true;
            ekfom_data.z_IMU(4) = 0.0;
        }

        if (fabs(acc_avr(2)) >= 0.99 * satu_acc) {
            ekfom_data.satu_check[5] = true;
            ekfom_data.z_IMU(5) = 0.0;
        }
    }
}

void pointBodyToWorld(PointType const *const pi, PointType *const po) {
    V3D p_body(pi->x, pi->y, pi->z);
    V3D p_global = kf_output.x_.rot * (Lidar_R_wrt_IMU * p_body + Lidar_T_wrt_IMU) + kf_output.x_.pos;

    po->x = p_global(0);
    po->y = p_global(1);
    po->z = p_global(2);
    po->intensity = pi->intensity;
}

const bool time_list(PointType &x, PointType &y) { return (x.curvature < y.curvature); };