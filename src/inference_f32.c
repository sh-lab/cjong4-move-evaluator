#include "cjong4_move_evaluator/model.h"

#include <math.h>
#include <stddef.h>

static bool dense_relu(const float *input, size_t input_count,
                       const float *weights, const float *biases,
                       size_t output_count, float *output) {
  for (size_t out = 0; out < output_count; ++out) {
    float value = biases[out];
    if (!isfinite(value))
      return false;
    for (size_t in = 0; in < input_count; ++in) {
      const float weight = weights[out * input_count + in];
      if (!isfinite(input[in]) || !isfinite(weight))
        return false;
      value += input[in] * weight;
    }
    if (!isfinite(value))
      return false;
    output[out] = value > 0.0f ? value : 0.0f;
  }
  return true;
}

bool cj4me_infer_f32(const cj4me_model_f32 *model,
                     const float input[CJ4ME_FEATURE_COUNT],
                     cj4me_inference_f32_scratch *scratch, float *output) {
  float value;

  if (!model || !input || !scratch || !output ||
      !dense_relu(input, CJ4ME_FEATURE_COUNT, &model->weights1[0][0],
                  model->biases1, CJ4ME_MODEL_HIDDEN1_COUNT,
                  scratch->hidden1) ||
      !dense_relu(scratch->hidden1, CJ4ME_MODEL_HIDDEN1_COUNT,
                  &model->weights2[0][0], model->biases2,
                  CJ4ME_MODEL_HIDDEN2_COUNT, scratch->hidden2) ||
      !dense_relu(scratch->hidden2, CJ4ME_MODEL_HIDDEN2_COUNT,
                  &model->weights3[0][0], model->biases3,
                  CJ4ME_MODEL_HIDDEN3_COUNT, scratch->hidden3)) {
    return false;
  }

  value = model->biases4[0];
  if (!isfinite(value))
    return false;
  for (size_t in = 0; in < CJ4ME_MODEL_HIDDEN3_COUNT; ++in) {
    const float weight = model->weights4[0][in];
    if (!isfinite(weight))
      return false;
    value += scratch->hidden3[in] * weight;
  }
  if (!isfinite(value))
    return false;
  *output = value;
  return true;
}
