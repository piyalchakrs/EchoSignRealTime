#pragma once
#include <Arduino.h>
#include <Wire.h>
#include <MPU6050.h>   // Install "MPU6050" library (e.g., Electronic Cats)
#include "Calib.h"
#include "scaler_params.h"
#include "knn_runtime.h"

// ----------- Sensor + feature helper -----------

class GlovePredictor {
public:
  GlovePredictor()
  : mpu(MPU6050(MPU6050_ADDR))
  {}

  bool begin() {
    Wire.begin(PIN_I2C_SDA, PIN_I2C_SCL);
    mpu.initialize();
    bool ok = mpu.testConnection();
    if (!ok) return false;

    // ESP32 ADC config
    analogReadResolution(12); // 0..4095

    return true;
  }

  // Read a *single* raw frame from flex + IMU.
  // rawFlex[5], ax/ay/az/gx/gy/gz (int16_t)
  void readRawFrame(int rawFlex[5],
                    int16_t& ax, int16_t& ay, int16_t& az,
                    int16_t& gx, int16_t& gy, int16_t& gz) {

    for (int i = 0; i < 5; ++i) {
      rawFlex[i] = analogRead(PIN_FLEX[i]);
    }

    mpu.getMotion6(&ax, &ay, &az, &gx, &gy, &gz);
  }

  // Build a smoothed feature vector by averaging several raw frames
  // over ~windowMs milliseconds.
  //
  // Feature order (strict!):
  // 0: f1 (thumb raw ADC)
  // 1: f2 (index raw ADC)
  // 2: f3 (middle raw ADC)
  // 3: f4 (ring raw ADC)
  // 4: f5 (pinky raw ADC)
  // 5: gdp  (gyro magnitude)
  // 6: ax
  // 7: ay
  // 8: az
  // 9: gx
  // 10: gy
  // 11: gz
  void buildFeatureVector(float feat[NUM_FEATURES],
                          uint32_t windowMs = 200,
                          uint32_t sampleDelayMs = 10) {

    const int maxSamples = 64;
    int count = 0;
    double acc[NUM_FEATURES] = {0};

    uint32_t t0 = millis();

    while (millis() - t0 < windowMs && count < maxSamples) {
      int flex[5];
      int16_t ax, ay, az, gx, gy, gz;
      readRawFrame(flex, ax, ay, az, gx, gy, gz);

      float f1 = (float)flex[0];
      float f2 = (float)flex[1];
      float f3 = (float)flex[2];
      float f4 = (float)flex[3];
      float f5 = (float)flex[4];

      // GDP = gyro magnitude
      float fgx = (float)gx;
      float fgy = (float)gy;
      float fgz = (float)gz;
      float gdp = sqrtf(fgx * fgx + fgy * fgy + fgz * fgz);

      float fax = (float)ax;
      float fay = (float)ay;
      float faz = (float)az;

      float v[NUM_FEATURES] = {
        f1, f2, f3, f4, f5,
        gdp,
        fax, fay, faz,
        fgx, fgy, fgz
      };

      for (int i = 0; i < NUM_FEATURES; ++i) {
        acc[i] += v[i];
      }
      count++;

      delay(sampleDelayMs);
    }

    if (count == 0) {
      for (int i = 0; i < NUM_FEATURES; ++i) feat[i] = 0.0f;
      return;
    }

    for (int i = 0; i < NUM_FEATURES; ++i) {
      feat[i] = (float)(acc[i] / (double)count);
    }
  }

  // Build, standardize, and classify one feature window.
  // Returns label index.
  uint8_t predictGesture(float* outBestDist = nullptr,
                         uint32_t windowMs = 200,
                         uint32_t sampleDelayMs = 10) {

    float feat[NUM_FEATURES];
    buildFeatureVector(feat, windowMs, sampleDelayMs);

    // Normalize with exported scaler parameters
    standardizeFeatures(feat);

    return knn_predict(feat, outBestDist);
  }

private:
  MPU6050 mpu;
};
