// line_endpoint_comparison.cpp
// 对比两种线特征表示方法：
// 方案A: Plücker + 线投影 (EdgeSE3Line3DProjection)
// 方案B: 端点 + 点投影 (EdgeProjectXYZ2UV)

#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>

#include "g2o/core/base_multi_edge.h"
#include "g2o/core/block_solver.h"
#include "g2o/core/optimization_algorithm_factory.h"
#include "g2o/core/robust_kernel_impl.h"
#include "g2o/core/sparse_optimizer.h"
#include "g2o/stuff/command_args.h"
#include "g2o/stuff/sampler.h"
#include "g2o/types/sba/types_six_dof_expmap.h"
#include "g2o/types/slam3d/types_slam3d.h"
#include "g2o/types/slam3d_addons/edge_se3_line3d_projection.h"
#include "g2o/types/slam3d_addons/types_slam3d_addons.h"

using namespace g2o;
using namespace std;
using namespace Eigen;

G2O_USE_OPTIMIZATION_LIBRARY(eigen)

// ============================================================================
// 工具函数
// ============================================================================
Eigen::Isometry3d sample_noise_from_se3(const Vector6& cov) {
  double nx = Sampler::gaussRand(0., cov(0));
  double ny = Sampler::gaussRand(0., cov(1));
  double nz = Sampler::gaussRand(0., cov(2));
  double nroll = Sampler::gaussRand(0., cov(3));
  double npitch = Sampler::gaussRand(0., cov(4));
  double nyaw = Sampler::gaussRand(0., cov(5));

  AngleAxisd aa(AngleAxisd(nyaw, Vector3d::UnitZ()) *
                AngleAxisd(nroll, Vector3d::UnitX()) *
                AngleAxisd(npitch, Vector3d::UnitY()));

  Eigen::Isometry3d retval = Isometry3d::Identity();
  retval.matrix().block<3, 3>(0, 0) = aa.toRotationMatrix();
  retval.translation() = Vector3d(nx, ny, nz);
  return retval;
}

Vector2d sample_noise_from_line2d(const Vector2d& cov) {
  return Vector2d(Sampler::gaussRand(0., cov(0)),
                  Sampler::gaussRand(0., cov(1)));
}

// ============================================================================
// 点到线距离边 (参考 PL-VINS)
// ============================================================================
// 误差：观测端点到预测线的距离（2维）
// 顶点：[0] 端点1, [1] 端点2, [2] 位姿
// 测量：观测到的两个端点像素坐标 (u1, v1, u2, v2)
class EdgeEndpointToLine2D : public BaseMultiEdge<2, Vector4d> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

  EdgeEndpointToLine2D(double fx_, double fy_, double cx_, double cy_)
      : fx(fx_), fy(fy_), cx(cx_), cy(cy_) {
    resize(3);  // 3个顶点：端点1, 端点2, 位姿
  }

  void computeError() override {
    const VertexPointXYZ* vp1 =
        static_cast<const VertexPointXYZ*>(_vertices[0]);
    const VertexPointXYZ* vp2 =
        static_cast<const VertexPointXYZ*>(_vertices[1]);
    const VertexSE3Expmap* vpose =
        static_cast<const VertexSE3Expmap*>(_vertices[2]);

    // 3D端点（世界坐标系）
    Vector3d P1_w = vp1->estimate();
    Vector3d P2_w = vp2->estimate();

    // 变换到相机坐标系 (SE3Quat 存储的是 T_cw)
    Vector3d P1_c = vpose->estimate().map(P1_w);
    Vector3d P2_c = vpose->estimate().map(P2_w);

    // 投影到归一化平面
    Vector2d p1_norm(P1_c.x() / P1_c.z(), P1_c.y() / P1_c.z());
    Vector2d p2_norm(P2_c.x() / P2_c.z(), P2_c.y() / P2_c.z());

    // 从两个预测端点拟合归一化平面上的2D直线
    // 直线方程: n·p + d = 0，其中 n = [a, b], d = c
    // 法向量 n 垂直于方向向量
    Vector2d dir = p2_norm - p1_norm;
    double len = dir.norm();

    if (len < 1e-6) {
      // 两点重合，误差设为大值
      _error(0) = 1000.0;
      _error(1) = 1000.0;
      return;
    }

    // 直线法向量（归一化）：垂直于方向向量
    Vector2d normal(-dir.y() / len, dir.x() / len);

    // 直线方程: normal·(p - p1_norm) = 0
    // => normal·p - normal·p1_norm = 0
    // => a*x + b*y + c = 0, 其中 c = -normal·p1_norm
    double c = -normal.dot(p1_norm);

    // 将观测像素坐标转换为归一化坐标
    Vector2d obs1_norm((_measurement(0) - cx) / fx,
                       (_measurement(1) - cy) / fy);
    Vector2d obs2_norm((_measurement(2) - cx) / fx,
                       (_measurement(3) - cy) / fy);

    // 点到线的距离（已归一化，所以分母是1）
    _error(0) = normal.dot(obs1_norm) + c;
    _error(1) = normal.dot(obs2_norm) + c;
  }

  virtual bool read(std::istream& is) override {
    Vector4d meas;
    is >> meas(0) >> meas(1) >> meas(2) >> meas(3);
    setMeasurement(meas);
    return readInformationMatrix(is);
  }

  virtual bool write(std::ostream& os) const override {
    os << _measurement(0) << " " << _measurement(1) << " " << _measurement(2)
       << " " << _measurement(3) << " ";
    return writeInformationMatrix(os);
  }

  // 解析雅可比矩阵（参考 PL-VINS）
  // 暂时注释掉，使用 g2o 默认数值求导来验证 computeError
  void linearizeOplus_DISABLED() {
    const VertexPointXYZ* vp1 =
        static_cast<const VertexPointXYZ*>(_vertices[0]);
    const VertexPointXYZ* vp2 =
        static_cast<const VertexPointXYZ*>(_vertices[1]);
    const VertexSE3Expmap* vpose =
        static_cast<const VertexSE3Expmap*>(_vertices[2]);

    // 3D端点（世界坐标系）
    Vector3d P1_w = vp1->estimate();
    Vector3d P2_w = vp2->estimate();

    // 变换到相机坐标系
    SE3Quat T_cw = vpose->estimate();
    Vector3d P1_c = T_cw.map(P1_w);
    Vector3d P2_c = T_cw.map(P2_w);

    double X1 = P1_c.x(), Y1 = P1_c.y(), Z1 = P1_c.z();
    double X2 = P2_c.x(), Y2 = P2_c.y(), Z2 = P2_c.z();

    // 归一化坐标
    double u0 = X1 / Z1, v0 = Y1 / Z1;
    double u1 = X2 / Z2, v1 = Y2 / Z2;

    // 2D直线参数: l = (n1, n2, rho)，直线方程 n1*x + n2*y + rho = 0
    double n1 = v0 - v1;
    double n2 = u1 - u0;
    double rho = u0 * v1 - u1 * v0;

    double l_norm = n1 * n1 + n2 * n2;
    double l_sqrtnorm = sqrt(l_norm);
    double l_trinorm = l_norm * l_sqrtnorm;

    if (l_sqrtnorm < 1e-6) {
      _jacobianOplus[0].setZero();
      _jacobianOplus[1].setZero();
      _jacobianOplus[2].setZero();
      return;
    }

    // 观测点的归一化坐标
    double obs_x1 = (_measurement(0) - cx) / fx;
    double obs_y1 = (_measurement(1) - cy) / fy;
    double obs_x2 = (_measurement(2) - cx) / fx;
    double obs_y2 = (_measurement(3) - cy) / fy;

    // 未归一化的误差
    double e1_raw = n1 * obs_x1 + n2 * obs_y1 + rho;
    double e2_raw = n1 * obs_x2 + n2 * obs_y2 + rho;

    // 误差对2D线参数 (n1, n2, rho) 的雅可比 (参考 PL-VINS)
    Eigen::Matrix<double, 2, 3> jaco_e_l;
    jaco_e_l << (obs_x1 / l_sqrtnorm - n1 * e1_raw / l_trinorm),
        (obs_y1 / l_sqrtnorm - n2 * e1_raw / l_trinorm), (1.0 / l_sqrtnorm),
        (obs_x2 / l_sqrtnorm - n1 * e2_raw / l_trinorm),
        (obs_y2 / l_sqrtnorm - n2 * e2_raw / l_trinorm), (1.0 / l_sqrtnorm);

    // 2D线参数对归一化坐标 (u0, v0, u1, v1) 的雅可比
    // n1 = v0 - v1, n2 = u1 - u0, rho = u0*v1 - u1*v0
    Eigen::Matrix<double, 3, 4> jaco_l_uv;
    jaco_l_uv << 0, 1, 0, -1,      // dn1/d(u0,v0,u1,v1)
        -1, 0, 1, 0,               // dn2/d(u0,v0,u1,v1)
        v1, -u1, -v0, u0;          // drho/d(u0,v0,u1,v1)

    // 归一化坐标对相机坐标的雅可比
    // u = X/Z, v = Y/Z
    double invZ1 = 1.0 / Z1, invZ1_2 = invZ1 * invZ1;
    double invZ2 = 1.0 / Z2, invZ2_2 = invZ2 * invZ2;

    Eigen::Matrix<double, 2, 3> jaco_uv1_Pc1;
    jaco_uv1_Pc1 << invZ1, 0, -X1 * invZ1_2, 0, invZ1, -Y1 * invZ1_2;

    Eigen::Matrix<double, 2, 3> jaco_uv2_Pc2;
    jaco_uv2_Pc2 << invZ2, 0, -X2 * invZ2_2, 0, invZ2, -Y2 * invZ2_2;

    // 组合: (u0,v0,u1,v1) 对 (Pc1, Pc2) 的雅可比
    Eigen::Matrix<double, 4, 6> jaco_uv_Pc;
    jaco_uv_Pc.setZero();
    jaco_uv_Pc.block<2, 3>(0, 0) = jaco_uv1_Pc1;  // (u0,v0) 对 Pc1
    jaco_uv_Pc.block<2, 3>(2, 3) = jaco_uv2_Pc2;  // (u1,v1) 对 Pc2

    // 误差对相机坐标的雅可比
    Eigen::Matrix<double, 2, 6> jaco_e_Pc = jaco_e_l * jaco_l_uv * jaco_uv_Pc;

    // 相机坐标对世界坐标的雅可比: Pc = R * Pw + t => dPc/dPw = R
    Matrix3d R = T_cw.rotation().toRotationMatrix();

    // 雅可比对端点1 (Pw1)
    _jacobianOplus[0] = jaco_e_Pc.block<2, 3>(0, 0) * R;

    // 雅可比对端点2 (Pw2)
    _jacobianOplus[1] = jaco_e_Pc.block<2, 3>(0, 3) * R;

    // 雅可比对位姿 (SE3)
    // g2o SE3Expmap 使用左乘: T_new = exp(delta) * T_old
    // delta = (omega[0:3], upsilon[3:6])，omega是旋转，upsilon是平移
    // Pc_new = exp(delta) * Pc = (I + [omega]_x) * Pc + upsilon
    //        = Pc + omega × Pc + upsilon = Pc - Pc × omega + upsilon
    // dPc/d(omega) = -skew(Pc), dPc/d(upsilon) = I
    Eigen::Matrix<double, 3, 6> jaco_Pc1_pose, jaco_Pc2_pose;
    jaco_Pc1_pose.block<3, 3>(0, 0) = -skew(P1_c);           // d/d(omega)
    jaco_Pc1_pose.block<3, 3>(0, 3) = Matrix3d::Identity();  // d/d(upsilon)
    jaco_Pc2_pose.block<3, 3>(0, 0) = -skew(P2_c);
    jaco_Pc2_pose.block<3, 3>(0, 3) = Matrix3d::Identity();

    // 组合
    Eigen::Matrix<double, 6, 6> jaco_Pc_pose;
    jaco_Pc_pose.block<3, 6>(0, 0) = jaco_Pc1_pose;
    jaco_Pc_pose.block<3, 6>(3, 0) = jaco_Pc2_pose;

    _jacobianOplus[2] = jaco_e_Pc * jaco_Pc_pose;
  }

  // 反对称矩阵
  static Matrix3d skew(const Vector3d& v) {
    Matrix3d m;
    m << 0, -v.z(), v.y(), v.z(), 0, -v.x(), -v.y(), v.x(), 0;
    return m;
  }

  double fx, fy, cx, cy;  // 相机内参
};

// ============================================================================
// 实验结果结构
// ============================================================================
struct ExperimentResult {
  string method_name;
  double chi2_before;
  double chi2_after;
  double avg_trans_error;
  double avg_rot_error;
  double avg_line_error;
  int num_features;
  int num_observations;
  double computation_time_ms;
  bool converged;
};

// ============================================================================
// 场景数据生成
// ============================================================================
struct SceneData {
  // 线特征（端点表示）
  vector<pair<Vector3d, Vector3d>> lines;

  // 相机轨迹 (Ground Truth)
  vector<Isometry3d> poses;

  // 预生成的位姿噪声（保证两种方法使用相同噪声）
  vector<Isometry3d> pose_noises;
  vector<Isometry3d> odom_noises;

  // 预生成的线端点噪声（保证两种方法使用相同噪声）
  vector<pair<Vector3d, Vector3d>> lines_noisy;

  // 观测数据
  struct Observation {
    int pose_id;
    int line_id;
    Line2D line2d;
    Vector2d pixel1, pixel2;
  };
  vector<Observation> observations;

  // 相机参数
  double fx = 500, fy = 500, cx = 320, cy = 240;
  int img_width = 640, img_height = 480;

  // 噪声参数
  Vector6 odom_noise_sigma = Vector6(0.02, 0.01, 0.01, 0.002, 0.002, 0.005);

  void generateScene(bool debug = false) {
    // 世界坐标系: X朝前, Y向左, Z向上

    // 斜线：位于相机正前方，始终可见
    lines.push_back({Vector3d(3.0, -0.3, 0.2), Vector3d(3.5, 0.3, 0.8)});

    generateTrajectory();
    generateNoises();
    generateObservations(debug);
  }

  void generateNoises() {
    // 预生成位姿初始化噪声
    for (size_t i = 0; i < poses.size(); ++i) {
      pose_noises.push_back(sample_noise_from_se3(odom_noise_sigma));
    }
    // 预生成里程计测量噪声
    for (size_t i = 1; i < poses.size(); ++i) {
      odom_noises.push_back(sample_noise_from_se3(odom_noise_sigma));
    }
    // 预生成线端点初始化噪声
    double init_noise = 0.05;  // 5cm 噪声
    for (size_t i = 0; i < lines.size(); ++i) {
      Vector3d p1_noise(Sampler::gaussRand(0, init_noise),
                        Sampler::gaussRand(0, init_noise),
                        Sampler::gaussRand(0, init_noise));
      Vector3d p2_noise(Sampler::gaussRand(0, init_noise),
                        Sampler::gaussRand(0, init_noise),
                        Sampler::gaussRand(0, init_noise));
      Vector3d p1_noisy = lines[i].first + p1_noise;
      Vector3d p2_noisy = lines[i].second + p2_noise;
      lines_noisy.push_back({p1_noisy, p2_noisy});
    }
  }

 private:
  void generateTrajectory() {
    // 相机初始朝向：光轴指向世界+X方向
    // 线在正前方 X=3~3.5m 处
    Matrix3d R_wc;
    R_wc.col(0) = Vector3d(0, -1, 0);  // 相机X -> 世界-Y
    R_wc.col(1) = Vector3d(0, 0, -1);  // 相机Y -> 世界-Z
    R_wc.col(2) = Vector3d(1, 0, 0);   // 相机Z -> 世界+X

    Isometry3d pose = Isometry3d::Identity();
    pose.linear() = R_wc;
    pose.translation() = Vector3d(0, 0, 0.5);
    poses.push_back(pose);

    // 阶段1：大幅左右平移 - 20帧（总共左右移动约1m）
    for (int i = 0; i < 20; ++i) {
      Isometry3d delta = Isometry3d::Identity();
      double lateral = 0.1 * sin(i * M_PI / 10);  // 左右大幅摆动
      delta.translation() = Vector3d(0, lateral, 0);
      pose = pose * delta;
      poses.push_back(pose);
    }

    // 阶段2：大幅上下平移 - 20帧
    for (int i = 0; i < 20; ++i) {
      Isometry3d delta = Isometry3d::Identity();
      double vertical = 0.08 * sin(i * M_PI / 10);  // 上下大幅摆动
      delta.translation() = Vector3d(0, 0, vertical);
      pose = pose * delta;
      poses.push_back(pose);
    }

    // 阶段3：前进靠近线 - 20帧（靠近约1m）
    for (int i = 0; i < 20; ++i) {
      Isometry3d delta = Isometry3d::Identity();
      delta.translation() = Vector3d(0.05, 0, 0);  // 持续前进
      pose = pose * delta;
      poses.push_back(pose);
    }

    // 阶段4：在靠近位置左右平移 - 20帧
    for (int i = 0; i < 20; ++i) {
      Isometry3d delta = Isometry3d::Identity();
      double lateral = 0.08 * sin(i * M_PI / 10);
      delta.translation() = Vector3d(0, lateral, 0);
      pose = pose * delta;
      poses.push_back(pose);
    }

    // 阶段5：后退远离线 - 20帧
    for (int i = 0; i < 20; ++i) {
      Isometry3d delta = Isometry3d::Identity();
      delta.translation() = Vector3d(-0.05, 0, 0);  // 持续后退
      pose = pose * delta;
      poses.push_back(pose);
    }

    // 阶段6：斜向运动（同时平移和小幅旋转）- 20帧
    for (int i = 0; i < 20; ++i) {
      Isometry3d delta = Isometry3d::Identity();
      delta.translation() = Vector3d(0.03, 0.05 * sin(i * M_PI / 10),
                                     0.03 * cos(i * M_PI / 10));
      pose = pose * delta;
      poses.push_back(pose);
    }
  }

  void generateObservations(bool debug = false) {
    // 观测噪声
    Vector2d line_noise(0.02, 0.02);  // θ和ρ的标准差
    double pixel_noise = 10.0;        // 像素噪声
    double rho_threshold = 10.0;

    int reject_depth = 0, reject_fov = 0, reject_rho = 0;

    for (size_t pose_id = 0; pose_id < poses.size(); ++pose_id) {
      // poses[i] 存储的是相机在世界中的位姿 T_wc
      // T_wc 将相机系坐标变换到世界系: p_w = T_wc * p_c
      // 要把世界点变换到相机系，需要 T_cw = T_wc^(-1)
      Isometry3d T_wc = poses[pose_id];
      Isometry3d T_cw = T_wc.inverse();

      for (size_t line_id = 0; line_id < lines.size(); ++line_id) {
        Vector3d p1_w = lines[line_id].first;
        Vector3d p2_w = lines[line_id].second;

        // 变换到相机系
        Vector3d p1_c = T_cw * p1_w;
        Vector3d p2_c = T_cw * p2_w;

        if (debug && pose_id == 0) {
          cout << "  [调试] pose0, line" << line_id
               << ": p1_c=" << p1_c.transpose() << ", p2_c=" << p2_c.transpose()
               << endl;
        }

        // 检查深度 (点必须在相机前方)
        if (p1_c.z() < 0.1 || p2_c.z() < 0.1) {
          reject_depth++;
          continue;
        }

        // 投影到像素
        Vector2d pixel1(fx * p1_c.x() / p1_c.z() + cx,
                        fy * p1_c.y() / p1_c.z() + cy);
        Vector2d pixel2(fx * p2_c.x() / p2_c.z() + cx,
                        fy * p2_c.y() / p2_c.z() + cy);

        if (debug && pose_id == 0) {
          cout << "  [调试] pose0, line" << line_id
               << ": pixel1=" << pixel1.transpose()
               << ", pixel2=" << pixel2.transpose() << endl;
        }

        // 检查是否在图像内
        if (pixel1.x() < 0 || pixel1.x() >= img_width || pixel1.y() < 0 ||
            pixel1.y() >= img_height || pixel2.x() < 0 ||
            pixel2.x() >= img_width || pixel2.y() < 0 ||
            pixel2.y() >= img_height) {
          reject_fov++;
          continue;
        }

        // 从端点计算2D线参数（使用两点投影法，与边的方法一致）
        // 投影到归一化平面
        double u0 = p1_c.x() / p1_c.z();
        double v0 = p1_c.y() / p1_c.z();
        double u1 = p2_c.x() / p2_c.z();
        double v1 = p2_c.y() / p2_c.z();

        // 齐次叉积得到 2D 线参数
        double n1_raw = v0 - v1;
        double n2_raw = u1 - u0;
        double rho_raw = u0 * v1 - u1 * v0;

        double norm = sqrt(n1_raw * n1_raw + n2_raw * n2_raw);
        if (norm < 1e-10) continue;

        double n1 = n1_raw / norm;
        double n2 = n2_raw / norm;
        double d = rho_raw / norm;
        double theta_true = atan2(n2, n1);
        double rho_true = d;

        if (abs(rho_true) > rho_threshold) {
          reject_rho++;
          continue;
        }

        // 创建观测
        Observation obs;
        obs.pose_id = pose_id;
        obs.line_id = line_id;

        // Plücker观测 (添加噪声)
        Vector2d noise_line = sample_noise_from_line2d(line_noise);
        obs.line2d =
            Line2D(theta_true + noise_line(0), rho_true + noise_line(1));

        // 端点观测 (添加噪声)
        obs.pixel1 = pixel1 + Vector2d(Sampler::gaussRand(0, pixel_noise),
                                       Sampler::gaussRand(0, pixel_noise));
        obs.pixel2 = pixel2 + Vector2d(Sampler::gaussRand(0, pixel_noise),
                                       Sampler::gaussRand(0, pixel_noise));

        observations.push_back(obs);
      }
    }

    if (debug) {
      cout << "  [调试] 拒绝统计: 深度=" << reject_depth
           << ", FOV=" << reject_fov << ", rho=" << reject_rho
           << ", 有效观测=" << observations.size() << endl;
    }
  }
};

// ============================================================================
// 计算两条3D线之间的距离
// ============================================================================
pair<double, double> computeLineDistance(const Line3D& L1, const Line3D& L2) {
  Vector3d d1 = L1.d().normalized();
  Vector3d d2 = L2.d().normalized();
  Vector3d w1 = L1.w();
  Vector3d w2 = L2.w();

  // 方向误差: 两个方向向量的夹角 (考虑180度歧义)
  double dot = abs(d1.dot(d2));
  double dir_error = acos(min(1.0, dot)) * 180.0 / M_PI;

  // 位置误差: 两条线之间的最短距离
  Vector3d p1 = d1.cross(w1);  // L1 上距原点最近的点
  Vector3d p2 = d2.cross(w2);  // L2 上距原点最近的点

  double pos_error;
  if (dot > 0.999) {
    // 几乎平行，使用点到线距离
    Vector3d diff = p1 - p2;
    Vector3d perp = diff - d1 * (diff.dot(d1));
    pos_error = perp.norm();
  } else {
    // 非平行，使用最短距离公式
    Vector3d n = d1.cross(d2);
    double n_norm = n.norm();
    pos_error = abs((p2 - p1).dot(n)) / n_norm;
  }

  return {dir_error, pos_error};
}

// ============================================================================
// 辅助函数：Line3D 转端点（用于可视化）
// 使用真值线的中点作为参考，找到估计线上对应位置的端点
// ============================================================================
pair<Vector3d, Vector3d> line3dToEndpoints(const Line3D& L,
                                           double half_length,
                                           const Vector3d& ref_mid) {
  Vector3d d = L.d().normalized();
  Vector3d w = L.w();

  // 线上距原点最近的点
  Vector3d p0 = d.cross(w);

  // 找到线上距离 ref_mid 最近的点
  // 投影: t = (ref_mid - p0) · d
  double t = (ref_mid - p0).dot(d);
  Vector3d closest_to_ref = p0 + t * d;

  // 以 closest_to_ref 为中心，生成端点
  Vector3d p1 = closest_to_ref - half_length * d;
  Vector3d p2 = closest_to_ref + half_length * d;
  return {p1, p2};
}

// ============================================================================
// 保存线收敛历史到文件
// ============================================================================
void saveLineConvergenceHistory(
    const string& filename, const vector<pair<Vector3d, Vector3d>>& gt_lines,
    const vector<vector<pair<Vector3d, Vector3d>>>& line_history,
    const vector<double>& chi2_history) {
  ofstream ofs(filename);
  if (!ofs.is_open()) {
    cerr << "无法打开文件: " << filename << endl;
    return;
  }

  ofs << fixed << setprecision(6);

  // 写入真值线
  ofs << "# Ground Truth Lines (line_id p1x p1y p1z p2x p2y p2z)" << endl;
  ofs << "GT " << gt_lines.size() << endl;
  for (size_t i = 0; i < gt_lines.size(); ++i) {
    ofs << i << " " << gt_lines[i].first.x() << " " << gt_lines[i].first.y()
        << " " << gt_lines[i].first.z() << " " << gt_lines[i].second.x() << " "
        << gt_lines[i].second.y() << " " << gt_lines[i].second.z() << endl;
  }

  // 写入每次迭代的线估计
  ofs << "# Iteration History (iter chi2 line_id p1x p1y p1z p2x p2y p2z)"
      << endl;
  ofs << "ITERATIONS " << line_history.size() << endl;
  for (size_t iter = 0; iter < line_history.size(); ++iter) {
    ofs << "ITER " << iter << " " << chi2_history[iter] << endl;
    for (size_t lid = 0; lid < line_history[iter].size(); ++lid) {
      const auto& endpoints = line_history[iter][lid];
      ofs << lid << " " << endpoints.first.x() << " " << endpoints.first.y()
          << " " << endpoints.first.z() << " " << endpoints.second.x() << " "
          << endpoints.second.y() << " " << endpoints.second.z() << endl;
    }
  }

  ofs.close();
  cout << "  线收敛历史已保存到: " << filename << endl;
}

// ============================================================================
// 方案A: Plücker + 线投影
// ============================================================================
ExperimentResult runPluckerMethod(SceneData& data, int maxIter, bool verbose,
                                  bool saveHistory) {
  auto start_time = chrono::high_resolution_clock::now();

  SparseOptimizer optimizer;

  OptimizationAlgorithmFactory* solverFactory =
      OptimizationAlgorithmFactory::instance();
  OptimizationAlgorithmProperty solverProperty;
  OptimizationAlgorithm* solver =
      solverFactory->construct("lm_var", solverProperty);
  optimizer.setAlgorithm(solver);

  ParameterSE3Offset* offset = new ParameterSE3Offset();
  offset->setId(0);
  optimizer.addParameter(offset);

  // 添加位姿顶点 (VertexSE3)
  map<int, VertexSE3*> pose_vertices;

  for (size_t i = 0; i < data.poses.size(); ++i) {
    VertexSE3* v = new VertexSE3();
    v->setId(1000 + i);

    // 使用真值位姿（固定所有位姿，只优化线）
    v->setEstimate(data.poses[i]);
    v->setFixed(true);

    optimizer.addVertex(v);
    pose_vertices[i] = v;
  }

  // 添加里程计边
  for (size_t i = 1; i < data.poses.size(); ++i) {
    Isometry3d delta = data.poses[i - 1].inverse() * data.poses[i];

    EdgeSE3* e = new EdgeSE3();
    e->vertices()[0] = pose_vertices[i - 1];
    e->vertices()[1] = pose_vertices[i];
    e->setMeasurement(delta * data.odom_noises[i - 1]);

    Matrix6 info = Matrix6::Identity();
    for (int j = 0; j < 6; ++j) {
      info(j, j) = 1.0 / (data.odom_noise_sigma(j) * data.odom_noise_sigma(j));
    }
    e->setInformation(info);
    optimizer.addEdge(e);
  }

  // 添加线顶点（使用切空间参数化，与端点方法使用相同的初始化噪声）
  map<int, VertexLine3DTangent*> line_vertices;

  for (size_t i = 0; i < data.lines.size(); ++i) {
    VertexLine3DTangent* v = new VertexLine3DTangent();
    v->setId(i);

    // 使用预生成的带噪声端点（与端点方法使用完全相同的噪声）
    Vector3d p1_noisy = data.lines_noisy[i].first;
    Vector3d p2_noisy = data.lines_noisy[i].second;

    Vector3d dir = (p2_noisy - p1_noisy).normalized();
    // 使用线的中点而不是端点来初始化，避免 fromCartesian 的位置偏移问题
    Vector3d mid_point = (p1_noisy + p2_noisy) / 2.0;

    Vector6 cartesian;
    cartesian << mid_point, dir;
    // Vector6 cartesian;
    // cartesian << 0.0, 0.0, 5.0, 0.0, 1.0, 0.0;
    v->setEstimate(Line3D::fromCartesian(cartesian));

    optimizer.addVertex(v);
    line_vertices[i] = v;
  }

  // 统计每条线的观测数量
  vector<int> obs_count(data.lines.size(), 0);

  // 添加线观测边
  for (auto& obs : data.observations) {
    EdgeSE3Line3DProjection* e = new EdgeSE3Line3DProjection();
    e->vertices()[0] = pose_vertices[obs.pose_id];
    e->vertices()[1] = line_vertices[obs.line_id];
    e->setMeasurement(obs.line2d);

    // 信息矩阵: 1 / sigma^2, sigma = 0.02
    Matrix2d info = Matrix2d::Zero();
    info(0, 0) = 1.0 / (0.02 * 0.02);  // theta: 2500
    info(1, 1) = 1.0 / (0.02 * 0.02);  // rho: 2500
    e->setInformation(info);

    // 使用 Huber 核函数
    RobustKernelHuber* rk = new RobustKernelHuber;
    rk->setDelta(1.0);
    e->setRobustKernel(rk);

    optimizer.addEdge(e);
    obs_count[obs.line_id]++;
  }

  // 打印每条线的观测数量
  cout << "\n每条线的观测数量:" << endl;
  for (size_t i = 0; i < data.lines.size(); ++i) {
    cout << "  线 " << i << ": " << obs_count[i] << " 次观测" << endl;
  }

  // 计算真值线的 Plücker 表示
  vector<Line3D> gt_lines_plucker;
  for (size_t i = 0; i < data.lines.size(); ++i) {
    Vector3d gt_mid = (data.lines[i].first + data.lines[i].second) / 2.0;
    Vector3d gt_dir = (data.lines[i].second - data.lines[i].first).normalized();
    Vector6 cart;
    cart << gt_mid, gt_dir;
    gt_lines_plucker.push_back(Line3D::fromCartesian(cart));
  }

  // 打印线的初始状态
  cout << "\n线的初始状态 vs 真值（使用正确的线距离度量）:" << endl;
  for (size_t i = 0; i < data.lines.size(); ++i) {
    Line3D L = line_vertices[i]->estimate();
    auto [dir_err, pos_err] = computeLineDistance(gt_lines_plucker[i], L);
    cout << "  线 " << i << ": 方向误差=" << dir_err << "度, 位置误差=" << pos_err << "m" << endl;
  }

  // 优化
  optimizer.initializeOptimization();
  optimizer.computeActiveErrors();
  double chi2_before = optimizer.chi2();

  // 收敛历史记录
  vector<vector<pair<Vector3d, Vector3d>>> line_history;
  vector<double> chi2_history;

  // 记录初始状态时使用带噪声的端点，保持与端点方法一致
  bool first_record = true;
  auto recordCurrentLines = [&]() {
    vector<pair<Vector3d, Vector3d>> current_lines;
    for (size_t i = 0; i < data.lines.size(); ++i) {
      if (first_record) {
        // ITER 0: 直接使用带噪声的初始端点，保持与端点方法一致
        current_lines.push_back(data.lines_noisy[i]);
      } else {
        // 后续迭代: 从 Plücker 线转换
        Line3D L = line_vertices[i]->estimate();
        double gt_length = (data.lines[i].second - data.lines[i].first).norm();
        Vector3d noisy_mid =
            (data.lines_noisy[i].first + data.lines_noisy[i].second) / 2.0;
        current_lines.push_back(
            line3dToEndpoints(L, gt_length / 2.0, noisy_mid));
      }
    }
    line_history.push_back(current_lines);
    chi2_history.push_back(optimizer.chi2());
    first_record = false;
  };

  optimizer.setVerbose(verbose);

  int iterations = 0;
  if (saveHistory) {
    // 单步迭代模式：记录每次迭代
    recordCurrentLines();  // 记录初始状态
    for (int iter = 0; iter < maxIter; ++iter) {
      int ret = optimizer.optimize(1);
      if (ret <= 0) break;
      iterations++;
      optimizer.computeActiveErrors();
      recordCurrentLines();
    }
  } else {
    // 正常模式：一次性优化
    iterations = optimizer.optimize(maxIter);
  }

  optimizer.computeActiveErrors();
  double chi2_after = optimizer.chi2();

  // 打印线的最终状态
  cout << "\n线的最终状态 vs 真值（使用正确的线距离度量）:" << endl;
  double total_line_error = 0;
  for (size_t i = 0; i < data.lines.size(); ++i) {
    Line3D L = line_vertices[i]->estimate();
    auto [dir_err, pos_err] = computeLineDistance(gt_lines_plucker[i], L);
    total_line_error += pos_err;
    cout << "  线 " << i << ": 方向误差=" << dir_err << "度, 位置误差=" << pos_err << "m";
    if (pos_err < 0.1 && dir_err < 5.0) {
      cout << " [收敛]";
    } else {
      cout << " [未收敛]";
    }
    cout << endl;
  }

  // 保存收敛历史
  if (saveHistory) {
    saveLineConvergenceHistory("line_convergence_plucker.txt", data.lines,
                               line_history, chi2_history);
  }

  auto end_time = chrono::high_resolution_clock::now();
  double time_ms =
      chrono::duration<double, milli>(end_time - start_time).count();

  // 计算误差 (位姿都固定了，这里只是占位)
  double total_trans = 0, total_rot = 0;
  int pose_count = 1;  // 避免除零

  ExperimentResult result;
  result.method_name = "Plücker+线投影";
  result.chi2_before = chi2_before;
  result.chi2_after = chi2_after;
  result.avg_trans_error = total_line_error / data.lines.size();  // 用线误差代替
  result.avg_rot_error = 0;
  result.num_features = data.lines.size();
  result.num_observations = data.observations.size();
  result.computation_time_ms = time_ms;
  result.converged = (iterations > 0);

  return result;
}

// ============================================================================
// 方案B: 端点 + 点投影
// ============================================================================
ExperimentResult runEndpointMethod(SceneData& data, int maxIter, bool verbose,
                                   bool saveHistory = false) {
  auto start_time = chrono::high_resolution_clock::now();

  SparseOptimizer optimizer;

  OptimizationAlgorithmFactory* solverFactory =
      OptimizationAlgorithmFactory::instance();
  OptimizationAlgorithmProperty solverProperty;
  OptimizationAlgorithm* solver =
      solverFactory->construct("lm_var", solverProperty);
  optimizer.setAlgorithm(solver);

  // 添加相机参数
  // CameraParameters 构造函数: (focal_length, principle_point, baseline)
  // 注意: CameraParameters 假设 fx = fy = focal_length
  CameraParameters* cam =
      new CameraParameters(data.fx, g2o::Vector2(data.cx, data.cy), 0);
  cam->setId(0);
  optimizer.addParameter(cam);

  // 添加位姿顶点 (VertexSE3Expmap)
  // 注意: SE3Quat 存储的是 T_cw (相机坐标系到世界坐标系的变换的逆)
  // SE3Quat::map(p_w) 计算 T_cw * p_w = p_c
  map<int, VertexSE3Expmap*> pose_vertices;

  for (size_t i = 0; i < data.poses.size(); ++i) {
    VertexSE3Expmap* v = new VertexSE3Expmap();
    v->setId(1000 + i);

    // data.poses[i] 是 T_wc (相机在世界中的位姿)
    // SE3Quat 需要 T_cw = T_wc^(-1)
    // 使用真值位姿（固定所有位姿，只优化端点）
    Isometry3d T_cw = data.poses[i].inverse();

    v->setEstimate(SE3Quat(T_cw.rotation(), T_cw.translation()));
    v->setFixed(true);  // 固定所有位姿

    optimizer.addVertex(v);
    pose_vertices[i] = v;
  }

  // 注意：位姿已全部固定，不需要添加位姿边
  // 位姿边只在位姿可优化时才有意义

  // 添加端点顶点
  map<int, pair<VertexPointXYZ*, VertexPointXYZ*>> endpoint_vertices;

  for (size_t i = 0; i < data.lines.size(); ++i) {
    // 使用预生成的带噪声端点（与Plücker方法使用完全相同的噪声）
    Vector3d p1_noisy = data.lines_noisy[i].first;
    Vector3d p2_noisy = data.lines_noisy[i].second;

    VertexPointXYZ* v1 = new VertexPointXYZ();
    v1->setId(i * 2);
    v1->setEstimate(p1_noisy);
    optimizer.addVertex(v1);

    VertexPointXYZ* v2 = new VertexPointXYZ();
    v2->setId(i * 2 + 1);
    v2->setEstimate(p2_noisy);
    optimizer.addVertex(v2);

    endpoint_vertices[i] = {v1, v2};
  }

  // 添加端点观测边 - 使用点到线距离约束（PL-VINS方法）
  for (auto& obs : data.observations) {
    auto endpoints = endpoint_vertices[obs.line_id];

    // 创建点到线距离边（传入相机内参）
    EdgeEndpointToLine2D* e =
        new EdgeEndpointToLine2D(data.fx, data.fy, data.cx, data.cy);
    e->vertices()[0] = endpoints.first;             // 端点1
    e->vertices()[1] = endpoints.second;            // 端点2
    e->vertices()[2] = pose_vertices[obs.pose_id];  // 位姿

    // 测量：两个观测端点的像素坐标
    Vector4d meas;
    meas << obs.pixel1.x(), obs.pixel1.y(), obs.pixel2.x(), obs.pixel2.y();
    e->setMeasurement(meas);

    // 信息矩阵（2x2，对应两个误差分量）
    e->setInformation(Matrix2d::Identity() * 10.0);

    RobustKernelHuber* rk = new RobustKernelHuber;
    rk->setDelta(5.0);
    e->setRobustKernel(rk);
    optimizer.addEdge(e);
  }

  // 打印端点的初始状态
  cout << "\n端点的初始状态 vs 真值:" << endl;
  for (size_t i = 0; i < data.lines.size(); ++i) {
    Vector3d p1_est = endpoint_vertices[i].first->estimate();
    Vector3d p2_est = endpoint_vertices[i].second->estimate();
    double err1 = (data.lines[i].first - p1_est).norm();
    double err2 = (data.lines[i].second - p2_est).norm();
    cout << "  线 " << i << ": 端点1误差=" << err1 << "m, 端点2误差=" << err2 << "m" << endl;
  }

  // 优化
  optimizer.initializeOptimization();
  optimizer.computeActiveErrors();
  double chi2_before = optimizer.chi2();

  // 收敛历史记录
  vector<vector<pair<Vector3d, Vector3d>>> line_history;
  vector<double> chi2_history;

  auto recordCurrentEndpoints = [&]() {
    vector<pair<Vector3d, Vector3d>> current_lines;
    for (size_t i = 0; i < data.lines.size(); ++i) {
      Vector3d p1 = endpoint_vertices[i].first->estimate();
      Vector3d p2 = endpoint_vertices[i].second->estimate();
      current_lines.emplace_back(p1, p2);
    }
    line_history.push_back(current_lines);
    chi2_history.push_back(optimizer.chi2());
  };

  optimizer.setVerbose(verbose);

  int iterations = 0;
  if (saveHistory) {
    // 单步迭代模式：记录每次迭代
    recordCurrentEndpoints();  // 记录初始状态
    for (int iter = 0; iter < maxIter; ++iter) {
      int ret = optimizer.optimize(1);
      if (ret <= 0) break;
      iterations++;
      optimizer.computeActiveErrors();
      recordCurrentEndpoints();
    }
  } else {
    // 正常模式：一次性优化
    iterations = optimizer.optimize(maxIter);
  }

  optimizer.computeActiveErrors();
  double chi2_after = optimizer.chi2();

  // 计算端点误差
  double total_endpoint_error = 0;
  cout << "\n端点的最终状态 vs 真值:" << endl;
  for (size_t i = 0; i < data.lines.size(); ++i) {
    Vector3d p1_est = endpoint_vertices[i].first->estimate();
    Vector3d p2_est = endpoint_vertices[i].second->estimate();

    double err1 = (data.lines[i].first - p1_est).norm();
    double err2 = (data.lines[i].second - p2_est).norm();
    double avg_err = (err1 + err2) / 2.0;
    total_endpoint_error += avg_err;

    cout << "  线 " << i << ": 端点1误差=" << err1 << "m, 端点2误差=" << err2 << "m";
    if (avg_err < 0.1) {
      cout << " [收敛]";
    } else {
      cout << " [未收敛]";
    }
    cout << endl;
  }

  // 保存收敛历史
  if (saveHistory) {
    saveLineConvergenceHistory("line_convergence_endpoint.txt", data.lines,
                               line_history, chi2_history);
  }

  auto end_time = chrono::high_resolution_clock::now();
  double time_ms =
      chrono::duration<double, milli>(end_time - start_time).count();

  ExperimentResult result;
  result.method_name = "端点+点到线距离";
  result.chi2_before = chi2_before;
  result.chi2_after = chi2_after;
  result.avg_trans_error = total_endpoint_error / data.lines.size();  // 用端点误差
  result.avg_rot_error = 0;
  result.avg_line_error = total_endpoint_error / data.lines.size();
  result.num_features = data.lines.size() * 2;
  result.num_observations = data.observations.size();  // 每个观测2个约束
  result.computation_time_ms = time_ms;
  result.converged = (iterations > 0);

  return result;
}

// ============================================================================
// 打印对比表格
// ============================================================================
void printComparisonTable(const vector<ExperimentResult>& results) {
  cout << "\n";
  cout << "===================================================================="
          "================="
       << endl;
  cout << "                                  实 验 结 果 对 比                 "
          "               "
       << endl;
  cout << "===================================================================="
          "================="
       << endl;

  printf("%-20s %12s %12s %12s %8s %8s %12s\n", "Method", "Chi2_Reduce",
         "Trans(m)", "Rot(deg)", "Feature", "Obs", "Time(ms)");
  cout << "--------------------------------------------------------------------"
          "-----------------"
       << endl;

  for (const auto& r : results) {
    double chi2_reduction = 0;
    if (r.chi2_before > 0) {
      chi2_reduction = (r.chi2_before - r.chi2_after) / r.chi2_before * 100.0;
    }

    printf("%-20s %12f%% %12f %12f %8d %8d %12f\n", r.method_name.c_str(),
           chi2_reduction, r.avg_trans_error, r.avg_rot_error, r.num_features,
           r.num_observations, r.computation_time_ms);
  }

  cout << "===================================================================="
          "================="
       << endl;

  // 详细对比
  if (results.size() == 2) {
    printf("\n");
    printf("对比总结：\n");
    printf("  观测数量: 方案A %4d  vs  方案B %4d\n",
           results[0].num_observations, results[1].num_observations);

    if (results[0].avg_trans_error > 0) {
      double trans_diff =
          (results[1].avg_trans_error - results[0].avg_trans_error) /
          results[0].avg_trans_error * 100;
      printf("  位置精度: 方案B %s 方案A %.1f%%\n",
             (trans_diff > 0 ? "劣于" : "优于"), abs(trans_diff));
    }

    if (results[0].avg_rot_error > 0) {
      double rot_diff = (results[1].avg_rot_error - results[0].avg_rot_error) /
                        results[0].avg_rot_error * 100;
      printf("  旋转精度: 方案B %s 方案A %.1f%%\n",
             (rot_diff > 0 ? "劣于" : "优于"), abs(rot_diff));
    }

    if (results[0].computation_time_ms > 0) {
      double time_diff =
          (results[1].computation_time_ms - results[0].computation_time_ms) /
          results[0].computation_time_ms * 100;
      printf("  计算效率: 方案B %s %.1f%%\n", (time_diff > 0 ? "慢" : "快"),
             abs(time_diff));
    }
  }
}

// ============================================================================
// 主函数
// ============================================================================
int main(int argc, char** argv) {
  bool verbose = false;
  int maxIterations = 15;
  bool runBoth = true;
  bool useEndpoint = false;
  bool saveConvergence = false;

  CommandArgs arg;
  arg.param("i", maxIterations, 15, "最大迭代次数");
  arg.param("v", verbose, false, "详细输出");
  arg.param("both", runBoth, true, "运行两种方法对比");
  arg.param("useEndpoint", useEndpoint, false, "仅运行端点方法");
  arg.param("saveConv", saveConvergence, false,
            "保存线收敛历史到文件(用于可视化)");
  arg.parseArgs(argc, argv);

  cout << "=== 线特征表示方法对比实验 ===" << endl;

  // 生成场景数据
  cout << "\n生成仿真数据..." << endl;
  SceneData data;
  data.generateScene(verbose);

  cout << "  线特征数: " << data.lines.size() << endl;
  cout << "  相机位姿数: " << data.poses.size() << endl;

  vector<ExperimentResult> results;

  if (runBoth) {
    // 运行方案A
    cout << "\n运行方案A: Plücker + 线投影..." << endl;
    results.push_back(
        runPluckerMethod(data, maxIterations, verbose, saveConvergence));

    // 运行方案B
    cout << "\n运行方案B: 端点 + 点投影..." << endl;
    results.push_back(
        runEndpointMethod(data, maxIterations, verbose, saveConvergence));

    // 打印对比结果
    printComparisonTable(results);
  } else if (useEndpoint) {
    cout << "\n运行方案B: 端点 + 点投影..." << endl;
    results.push_back(
        runEndpointMethod(data, maxIterations, verbose, saveConvergence));
    printComparisonTable(results);
  } else {
    cout << "\n运行方案A: Plücker + 线投影..." << endl;
    results.push_back(
        runPluckerMethod(data, maxIterations, verbose, saveConvergence));
    printComparisonTable(results);
  }

  return 0;
}
