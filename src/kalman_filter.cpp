#include <math.h>
#include "kalman_filter.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;

KalmanFilter::KalmanFilter() {}

KalmanFilter::~KalmanFilter() {}

void KalmanFilter::Init(VectorXd &x_in, MatrixXd &P_in, MatrixXd &F_in,
                        MatrixXd &H_in, MatrixXd &R_in, MatrixXd &Q_in) {
  x_ = x_in;
  P_ = P_in;
  F_ = F_in;
  H_ = H_in;
  R_ = R_in;
  Q_ = Q_in;
}

void KalmanFilter::Predict() {
  x_ = F_ * x_;
  MatrixXd Ft = F_.transpose();
  P_ = F_ * P_ * Ft + Q_;
}

void KalmanFilter::Update(const VectorXd &z) {
  VectorXd y = z - H_ * x_;
  UpdateKF(y);
}

void KalmanFilter::UpdateEKF(const VectorXd &z) {
  float px = x_(0);
  float py = x_(1);
  float vx = x_(2);
  float vy = x_(3);
  float c = sqrt(px * px + py * py);
  
  VectorXd hx(3);
  hx << c, atan2(py, px), (px * vx + py * vy)/c;

  VectorXd y = z - hx;

  // Adjust phi to be between -pi and pi
  float phi = y(1);
  if (phi < -1 * M_PI) {
    while (phi < -1 * M_PI) {
      phi += 2 * M_PI;
    }
  } else if (M_PI < phi) {
    while (M_PI < phi) {
      phi -= 2 * M_PI;
    }
  }
  y(1) = phi;

  UpdateKF(y);
}

void KalmanFilter::UpdateKF(const VectorXd &y) {
  MatrixXd Ht = H_.transpose();
  MatrixXd S = H_ * P_ * Ht + R_;
  MatrixXd K =  P_ * Ht * S.inverse();
  
  int size = x_.size();
  MatrixXd I = MatrixXd::Identity(size, size);
  x_ = x_ + (K * y);
  P_ = (I - K * H_) * P_;
}