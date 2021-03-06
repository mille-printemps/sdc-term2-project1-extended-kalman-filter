#include "FusionEKF.h"
#include "tools.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/*
 * Constructor
 */
FusionEKF::FusionEKF() {
  is_initialized_ = false;

  previous_timestamp_ = 0;

  // Initializing matrices

  // Measurement covariance matrix - laser
  R_laser_ = MatrixXd::Zero(2, 2);
  R_laser_ << 0.0225, 0,
              0, 0.0225;

  // Measurement covariance matrix - radar
  R_radar_ = MatrixXd::Zero(3, 3);
  R_radar_ << 0.09, 0, 0,
              0, 0.0009, 0,
              0, 0, 0.09;
  
  // Measurement matrix - laser
  H_laser_ = MatrixXd::Zero(2, 4);
  H_laser_ << 1, 0, 0, 0,
              0, 1, 0, 0;
  
  // Mesurement matrix - radar
  Hj_ = MatrixXd::Zero(3, 4);
  
  // State transition matrix
  ekf_.F_ = MatrixXd::Zero(4, 4);
  ekf_.F_ << 1, 0, 1, 0,
             0, 1, 0, 1,
             0, 0, 1, 0,
             0, 0, 0, 1;

  // State covariance matrix
  ekf_.P_ = MatrixXd::Zero(4, 4);
  ekf_.P_ << 1, 0, 0, 0,
             0, 1, 0, 0,
             0, 0, 1000, 0,
             0, 0, 0, 1000;
  
  // Process covariance matrix
  ekf_.Q_ = MatrixXd::Zero(4, 4);

  // Noise for the process covariance matrix
  noise_ax_ = 9;
  noise_ay_ = 9;
}

/**
 * Destructor
 */
FusionEKF::~FusionEKF() {}

void FusionEKF::ProcessMeasurement(const MeasurementPackage &measurement_pack) {

  /*****************************************************************************
   *  Initialization
   ****************************************************************************/
  
  if (!is_initialized_) {
    // First measurement
    cout << "EKF: " << endl;
    ekf_.x_ = VectorXd::Zero(4);
    
    if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
      // Use rho and phi to initialize px and py
      // Use 0 to initialize vx and vy
      ekf_.x_ << measurement_pack.raw_measurements_[0], measurement_pack.raw_measurements_[1], 0, 0;
    }
    else if (measurement_pack.sensor_type_ == MeasurementPackage::LASER) {
      // Use 0 to initialize vx and vy
      ekf_.x_ << measurement_pack.raw_measurements_[0], measurement_pack.raw_measurements_[1], 0, 0;
    }
    
    previous_timestamp_ = measurement_pack.timestamp_;
    
    // Done initializing, no need to predict or update
    is_initialized_ = true;
    return;
  }
    
  /*****************************************************************************
   *  Prediction
   ****************************************************************************/
  
  float dt = (measurement_pack.timestamp_ - previous_timestamp_) / 1000000.0;	//dt - expressed in seconds
  previous_timestamp_ = measurement_pack.timestamp_;
  
  float dt_2 = dt * dt;
  float dt_3 = dt_2 * dt;
  float dt_4 = dt_3 * dt;
  
  // Modify the F matrix so that the time is integrated
  ekf_.F_(0, 2) = dt;
  ekf_.F_(1, 3) = dt;
  
  // Set the process covariance matrix Q
  ekf_.Q_ <<  dt_4/4 * noise_ax_, 0, dt_3/2 * noise_ax_, 0,
              0, dt_4/4 * noise_ay_, 0, dt_3/2 * noise_ay_,
              dt_3/2 * noise_ax_, 0, dt_2 * noise_ax_, 0,
              0, dt_3/2 * noise_ay_, 0, dt_2 * noise_ay_;
  
  ekf_.Predict();

  /*****************************************************************************
   *  Update
   ****************************************************************************/

  // Replace R and H to update the state
  if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
    ekf_.R_ = R_radar_;
    ekf_.H_ = tools.CalculateJacobian(ekf_.x_);
    ekf_.UpdateEKF(measurement_pack.raw_measurements_);
  } else {
    ekf_.R_ = R_laser_;
    ekf_.H_ = H_laser_;
    ekf_.Update(measurement_pack.raw_measurements_);
  }

  // Print the output
  cout << "x_ = " << ekf_.x_ << endl;
  cout << "P_ = " << ekf_.P_ << endl;
}
