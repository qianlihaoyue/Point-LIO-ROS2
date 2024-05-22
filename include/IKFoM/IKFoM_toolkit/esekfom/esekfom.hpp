/*
 *  Copyright (c) 2019--2023, The University of Hong Kong
 *  All rights reserved.
 *
 *  Author: Dongjiao HE <hdj65822@connect.hku.hk>
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *   * Neither the name of the Universitaet Bremen nor the names of its
 *     contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef ESEKFOM_EKF_HPP
#define ESEKFOM_EKF_HPP

#include <vector>
#include <cstdlib>

#include <boost/bind.hpp>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/Dense>
#include <Eigen/Eigen>
#include <Eigen/Sparse>

#include "../mtk/types/vect.hpp"
#include "../mtk/types/SOn.hpp"
#include "../mtk/types/S2.hpp"
#include "../mtk/types/SEn.hpp"
#include "../mtk/startIdx.hpp"
#include "../mtk/build_manifold.hpp"
#include "util.hpp"

namespace esekfom {

using namespace Eigen;

template <typename T>
struct dyn_share_modified {
    bool valid;
    bool converge;
    T M_Noise;
    Eigen::Matrix<T, Eigen::Dynamic, 1> z;
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> h_x;
    Eigen::Matrix<T, 6, 1> z_IMU;
    Eigen::Matrix<T, 6, 1> R_IMU;
    bool satu_check[6];
};

template <typename state, int process_noise_dof, typename input = state, typename measurement = state, int measurement_noise_dof = 0>
class esekf {
    typedef esekf self;
    enum { n = state::DOF, m = state::DIM, l = measurement::DOF };

public:
    typedef typename state::scalar scalar_type;
    typedef Matrix<scalar_type, n, n> cov;
    typedef Matrix<scalar_type, m, n> cov_;
    typedef SparseMatrix<scalar_type> spMt;
    typedef Matrix<scalar_type, n, 1> vectorized_state;
    typedef Matrix<scalar_type, m, 1> flatted_state;
    typedef flatted_state processModel(state &, const input &);
    typedef Eigen::Matrix<scalar_type, m, n> processMatrix1(state &, const input &);
    typedef Eigen::Matrix<scalar_type, m, process_noise_dof> processMatrix2(state &, const input &);
    typedef Eigen::Matrix<scalar_type, process_noise_dof, process_noise_dof> processnoisecovariance;

    typedef void measurementModel_dyn_share_modified(state &, dyn_share_modified<scalar_type> &);
    typedef Eigen::Matrix<scalar_type, l, n> measurementMatrix1(state &);
    typedef Eigen::Matrix<scalar_type, Eigen::Dynamic, n> measurementMatrix1_dyn(state &);
    typedef Eigen::Matrix<scalar_type, l, measurement_noise_dof> measurementMatrix2(state &);
    typedef Eigen::Matrix<scalar_type, Eigen::Dynamic, Eigen::Dynamic> measurementMatrix2_dyn(state &);
    typedef Eigen::Matrix<scalar_type, measurement_noise_dof, measurement_noise_dof> measurementnoisecovariance;
    typedef Eigen::Matrix<scalar_type, Eigen::Dynamic, Eigen::Dynamic> measurementnoisecovariance_dyn;

    esekf(const state &x = state(), const cov &P = cov::Identity()) : x_(x), P_(P){};

    void init_dyn_share_modified_2h(processModel f_in, processMatrix1 f_x_in, measurementModel_dyn_share_modified h_dyn_share_in1,
                                    measurementModel_dyn_share_modified h_dyn_share_in2) {
        f = f_in;
        f_x = f_x_in;
        // f_w = f_w_in;
        h_dyn_share_modified_1 = h_dyn_share_in1;
        h_dyn_share_modified_2 = h_dyn_share_in2;
        maximum_iter = 1;
        x_.build_S2_state();
        x_.build_SO3_state();
        x_.build_vect_state();
        x_.build_SEN_state();
    }

    // IESEKF 的预测过程。使用当前状态估计和输入信号来估计下一个时刻的状态，并更新状态协方差矩阵
    // dt: 两次测量的时间差  Q: 过程噪声矩阵  i_in: 输入向量
    // predict_state: 是否需要预测状态, prop_cov: 是否需要传播协方差矩阵
    // 共进行三次预测:
    void predict(double &dt, processnoisecovariance &Q, const input &i_in, bool predict_state, bool prop_cov) {
        // 如果predict_state为真，则使用f(x_, i_in)计算状态f_，并使用x_（当前状态）和f_更新状态。
        if (predict_state) {
            flatted_state f_ = f(x_, i_in);
            x_.oplus(f_, dt);
        }
        // 协方差矩阵的传播
        if (prop_cov) {
            // 计算状态f_和雅可比矩阵f_x_
            flatted_state f_ = f(x_, i_in);
            // state x_before = x_;

            cov_ f_x_ = f_x(x_, i_in);
            cov f_x_final;
            // 创建一个单位矩阵F_x1，并遍历vect_state向量，将f_x_的部分元素赋值给f_x_final
            F_x1 = cov::Identity();
            for (std::vector<std::pair<std::pair<int, int>, int> >::iterator it = x_.vect_state.begin(); it != x_.vect_state.end(); it++) {
                int idx = (*it).first.first;
                int dim = (*it).first.second;
                int dof = (*it).second;
                for (int i = 0; i < n; i++) {
                    for (int j = 0; j < dof; j++) {
                        f_x_final(idx + j, i) = f_x_(dim + j, i);
                    }
                }
            }
            // 遍历SO3_state向量，并计算SO(3)李群的局部雅可比矩阵res_temp_SO3。将这个矩阵和f_x_final一起用于更新F_x1
            Matrix<scalar_type, 3, 3> res_temp_SO3;
            MTK::vect<3, scalar_type> seg_SO3;
            for (std::vector<std::pair<int, int> >::iterator it = x_.SO3_state.begin(); it != x_.SO3_state.end(); it++) {
                int idx = (*it).first;
                int dim = (*it).second;
                for (int i = 0; i < 3; i++) {
                    seg_SO3(i) = -1 * f_(dim + i) * dt;
                }
                MTK::SO3<scalar_type> res;
                res.w() = MTK::exp<scalar_type, 3>(res.vec(), seg_SO3, scalar_type(1 / 2));
                F_x1.template block<3, 3>(idx, idx) = res.normalized().toRotationMatrix();
                res_temp_SO3 = MTK::A_matrix(seg_SO3);
                for (int i = 0; i < n; i++) {
                    f_x_final.template block<3, 1>(idx, i) = res_temp_SO3 * (f_x_.template block<3, 1>(dim, i));
                }
            }
            // 使用F_x1和Q更新状态协方差矩阵P_
            F_x1 += f_x_final * dt;
            P_ = F_x1 * P_ * (F_x1).transpose() + Q * (dt * dt);
        }
    }

    // 针对Lidar输入的 EKF 更新
    bool update_iterated_dyn_share_modified() {
        dyn_share_modified<scalar_type> dyn_share;
        // 将当前状态 x_ 赋值给 x_propagated
        state x_propagated = x_;
        int dof_Measurement;  // 测量自由度
        double m_noise;       // 测量噪声
        for (int i = 0; i < maximum_iter; i++) {
            dyn_share.valid = true;
            // 就是逐点拟合平面求 观测矩阵H , 只要不是特征数量为0,就valid
            h_dyn_share_modified_1(x_, dyn_share);
            if (!dyn_share.valid) return false;

            // 计算观测矩阵 z 和雅可比矩阵 h_x, h_x来自于平面法向量, z貌似是点面误差
            Matrix<scalar_type, Eigen::Dynamic, Eigen::Dynamic> z = dyn_share.z;
            Matrix<scalar_type, Eigen::Dynamic, Eigen::Dynamic> h_x = dyn_share.h_x;

            dof_Measurement = h_x.rows();
            m_noise = dyn_share.M_Noise;

            // 计算卡尔曼增益 K_
            Matrix<scalar_type, n, Eigen::Dynamic> PHT;
            Matrix<scalar_type, Eigen::Dynamic, Eigen::Dynamic> HPHT;
            Matrix<scalar_type, n, Eigen::Dynamic> K_;
            {
                PHT = P_.template block<n, 12>(0, 0) * h_x.transpose();
                HPHT = h_x * PHT.topRows(12);
                for (int m = 0; m < dof_Measurement; m++) HPHT(m, m) += m_noise;
                K_ = PHT * HPHT.inverse();
            }
            // 更新状态 x_
            Matrix<scalar_type, n, 1> dx_ = K_ * z;
            x_.boxplus(dx_);
            dyn_share.converge = true;
            // 更新协方差矩阵 P_
            P_ = P_ - K_ * h_x * P_.template block<12, n>(0, 0);
        }
        return true;
    }

    // 针对 IMU 输入的 更新
    void update_iterated_dyn_share_IMU() {
        dyn_share_modified<scalar_type> dyn_share;
        for (int i = 0; i < maximum_iter; i++) {
            dyn_share.valid = true;
            // 调用 h_model_IMU_output 来更新 dyn_share 对象的 IMU 数据
            h_dyn_share_modified_2(x_, dyn_share);
            // 定义 6x1 的矩阵 z，并将其设置为 z_IMU
            Matrix<scalar_type, 6, 1> z = dyn_share.z_IMU;
            // 矩阵赋值
            Matrix<double, 30, 6> PHT;
            Matrix<double, 6, 30> HP;
            Matrix<double, 6, 6> HPHT;
            PHT.setZero();
            HP.setZero();
            HPHT.setZero();
            for (int l_ = 0; l_ < 6; l_++) {
                // 饱和检测
                if (!dyn_share.satu_check[l_]) {
                    PHT.col(l_) = P_.col(15 + l_) + P_.col(24 + l_);
                    HP.row(l_) = P_.row(15 + l_) + P_.row(24 + l_);
                }
            }
            for (int l_ = 0; l_ < 6; l_++) {
                if (!dyn_share.satu_check[l_]) {
                    HPHT.col(l_) = HP.col(15 + l_) + HP.col(24 + l_);
                }
                HPHT(l_, l_) += dyn_share.R_IMU(l_);  //, l);
            }
            // 计算卡尔曼增益矩阵 K
            Eigen::Matrix<double, 30, 6> K = PHT * HPHT.inverse();
            // 计算状态更新量 dx_
            Matrix<scalar_type, n, 1> dx_ = K * z;
            // 更新协方差矩阵 P_
            P_ -= K * HP;
            // 更新状态向量 x_，加上状态更新量 dx_
            x_.boxplus(dx_);
        }
        return;
    }

    void change_x(state &input_state) {
        x_ = input_state;

        if ((!x_.vect_state.size()) && (!x_.SO3_state.size()) && (!x_.S2_state.size()) && (!x_.SEN_state.size())) {
            x_.build_S2_state();
            x_.build_SO3_state();
            x_.build_vect_state();
            x_.build_SEN_state();
        }
    }

    void change_P(cov &input_cov) { P_ = input_cov; }

    const state &get_x() const { return x_; }
    const cov &get_P() const { return P_; }
    state x_;

private:
    measurement m_;
    cov P_;
    spMt l_;
    spMt f_x_1;
    spMt f_x_2;
    cov F_x1 = cov::Identity();
    cov F_x2 = cov::Identity();
    cov L_ = cov::Identity();

    processModel *f;
    processMatrix1 *f_x;
    processMatrix2 *f_w;

    measurementMatrix1 *h_x;
    measurementMatrix2 *h_v;

    measurementMatrix1_dyn *h_x_dyn;
    measurementMatrix2_dyn *h_v_dyn;

    measurementModel_dyn_share_modified *h_dyn_share_modified_1;
    measurementModel_dyn_share_modified *h_dyn_share_modified_2;

    int maximum_iter = 0;
    scalar_type limit[n];

public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

}  // namespace esekfom

#endif  //  ESEKFOM_EKF_HPP
