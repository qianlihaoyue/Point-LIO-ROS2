#ifndef Estimator_H
#define Estimator_H

#include <../include/IKFoM/IKFoM_toolkit/esekfom/esekfom.hpp>
#include "common_lib.h"
#include <pcl_conversions/pcl_conversions.h>
#include <ikd-Tree/ikd_Tree.h>

extern double laser_point_cov, acc_norm;
extern double vel_cov, acc_cov_input, gyr_cov_input;
extern double gyr_cov_output, acc_cov_output, b_gyr_cov, b_acc_cov;
extern double imu_meas_acc_cov, imu_meas_omg_cov;
extern double match_s, satu_acc, satu_gyro;
extern bool prop_at_freq_of_imu, check_satu;
extern float plane_thr;

extern PointCloudXYZI::Ptr normvec;  //(new PointCloudXYZI(100000, 1));
extern std::vector<int> time_seq;
extern PointCloudXYZI::Ptr feats_down_body;   //(new PointCloudXYZI());
extern PointCloudXYZI::Ptr feats_down_world;  //(new PointCloudXYZI());
extern std::vector<V3D> pbody_list;
extern std::vector<PointVector> Nearest_Points;
extern KD_TREE<PointType> ikdtree;
extern std::vector<float> pointSearchSqDis;
extern bool point_selected_surf[100000];  // = {0};
extern std::vector<M3D> crossmat_list;
extern int effct_feat_num;
extern int k;
extern int idx;
extern V3D angvel_avr, acc_avr;

extern V3D Lidar_T_wrt_IMU;  //(Zero3d);
extern M3D Lidar_R_wrt_IMU;  //(Eye3d);

typedef MTK::vect<3, double> vect3;
typedef MTK::SO3<double> SO3;
typedef MTK::S2<double, 98090, 10000, 1> S2;
typedef MTK::vect<1, double> vect1;
typedef MTK::vect<2, double> vect2;

MTK_BUILD_MANIFOLD(state_output, ((vect3, pos))((SO3, rot))((SO3, offset_R_L_I))((vect3, offset_T_L_I))((vect3, vel))((vect3, omg))((vect3, acc))(
                                     (vect3, gravity))((vect3, bg))((vect3, ba)));

MTK_BUILD_MANIFOLD(input_ikfom, ((vect3, acc))((vect3, gyro)));

MTK_BUILD_MANIFOLD(process_noise_input, ((vect3, ng))((vect3, na))((vect3, nbg))((vect3, nba)));

MTK_BUILD_MANIFOLD(process_noise_output, ((vect3, vel))((vect3, ng))((vect3, na))((vect3, nbg))((vect3, nba)));

extern esekfom::esekf<state_output, 30, input_ikfom> kf_output;
extern state_output state_out;
extern input_ikfom input_in;

Eigen::Matrix<double, 24, 24> process_noise_cov_input();

Eigen::Matrix<double, 30, 30> process_noise_cov_output();

Eigen::Matrix<double, 30, 1> get_f_output(state_output &s, const input_ikfom &in);

Eigen::Matrix<double, 30, 30> df_dx_output(state_output &s, const input_ikfom &in);

vect3 SO3ToEuler(const SO3 &orient);

void h_model_output(state_output &s, esekfom::dyn_share_modified<double> &ekfom_data);

void h_model_IMU_output(state_output &s, esekfom::dyn_share_modified<double> &ekfom_data);

void pointBodyToWorld(PointType const *const pi, PointType *const po);

const bool time_list(PointType &x, PointType &y);  // {return (x.curvature < y.curvature);};

#endif