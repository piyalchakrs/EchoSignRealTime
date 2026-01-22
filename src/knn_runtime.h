#pragma once
#include <Arduino.h>
#include "glove_knn_model.h"
#include "label_names.h"

// Compute distance between two feature vectors (size NUM_FEATURES)
inline float knn_distance(const float* a, const float* b) {
  if (KNN_METRIC == 0) {
    // Euclidean (squared L2)
    float d = 0.0f;
    for (int i = 0; i < NUM_FEATURES; ++i) {
      float diff = a[i] - b[i];
      d += diff * diff;
    }
    return d;
  } else if (KNN_METRIC == 1) {
    // Manhattan (L1)
    float d = 0.0f;
    for (int i = 0; i < NUM_FEATURES; ++i) {
      float diff = a[i] - b[i];
      d += (diff >= 0 ? diff : -diff);
    }
    return d;
  } else {
    // Chebyshev (L-infinity)
    float dmax = 0.0f;
    for (int i = 0; i < NUM_FEATURES; ++i) {
      float diff = a[i] - b[i];
      float adiff = (diff >= 0 ? diff : -diff);
      if (adiff > dmax) dmax = adiff;
    }
    return dmax;
  }
}

// KNN classification.
// Returns label index (compatible with label_names[]).
// Optionally returns the best (nearest) distance via out_best_dist.
inline uint8_t knn_predict(const float feat[NUM_FEATURES], float* out_best_dist = nullptr) {
  // Store K best neighbors
  float bestDist[KNN_K];
  uint8_t bestLabel[KNN_K];

  for (int k = 0; k < KNN_K; ++k) {
    bestDist[k] = 1e30f;
    bestLabel[k] = 0;
  }

  // Scan all training samples
  for (int i = 0; i < NUM_SAMPLES; ++i) {
    float sample[NUM_FEATURES];
    for (int j = 0; j < NUM_FEATURES; ++j) {
      sample[j] = X_train[i][j];  // PROGMEM is no-op on ESP32
    }

    float d = knn_distance(feat, sample);

    // Insert into sorted bestDist array (largest at the end)
    int idx = -1;
    for (int k = 0; k < KNN_K; ++k) {
      if (d < bestDist[k]) {
        idx = k;
        break;
      }
    }
    if (idx >= 0) {
      for (int k = KNN_K - 1; k > idx; --k) {
        bestDist[k] = bestDist[k - 1];
        bestLabel[k] = bestLabel[k - 1];
      }
      bestDist[idx] = d;
      bestLabel[idx] = y_train[i];
    }
  }

  if (out_best_dist) {
    *out_best_dist = bestDist[0];
  }

  // Voting over the K neighbors
  float votes[NUM_CLASSES];
  for (int c = 0; c < NUM_CLASSES; ++c) votes[c] = 0.0f;

  if (KNN_WEIGHTS == 0) {
    // Uniform weights
    for (int k = 0; k < KNN_K; ++k) {
      uint8_t lbl = bestLabel[k];
      if (lbl < NUM_CLASSES) votes[lbl] += 1.0f;
    }
  } else {
    // Distance weights: weight = 1 / (d + eps)
    const float eps = 1e-3f;
    for (int k = 0; k < KNN_K; ++k) {
      uint8_t lbl = bestLabel[k];
      float w = 1.0f / (bestDist[k] + eps);
      if (lbl < NUM_CLASSES) votes[lbl] += w;
    }
  }

  // Pick label with max vote
  uint8_t bestClass = 0;
  float bestVote = votes[0];
  for (int c = 1; c < NUM_CLASSES; ++c) {
    if (votes[c] > bestVote) {
      bestVote = votes[c];
      bestClass = c;
    }
  }

  return bestClass;
}
