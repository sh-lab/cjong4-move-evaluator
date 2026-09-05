#include "cjong4_move_evaluator/model.h"

#include <math.h>
#include <stddef.h>
#include <string.h>

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

  if (!model || !input || !scratch || !output)
    return false;

  for (size_t tile = 0; tile < CJ4ME_TILE_COUNT; ++tile) {
    const float *tile_input = input + tile * (size_t)CJ4ME_TILE_FEATURE_COUNT;
    float *tile_output =
        scratch->score_input + tile * (size_t)CJ4ME_TILE_EMBEDDING_COUNT;
    if (!dense_relu(tile_input, CJ4ME_TILE_FEATURE_COUNT,
                    &model->tile_weights1[0][0], model->tile_biases1,
                    CJ4ME_MODEL_TILE_HIDDEN_COUNT, scratch->tile_hidden) ||
        !dense_relu(scratch->tile_hidden, CJ4ME_MODEL_TILE_HIDDEN_COUNT,
                    &model->tile_weights2[0][0], model->tile_biases2,
                    CJ4ME_TILE_EMBEDDING_COUNT, tile_output)) {
      return false;
    }
  }
  memcpy(scratch->score_input +
             CJ4ME_TILE_COUNT * (size_t)CJ4ME_TILE_EMBEDDING_COUNT,
         input + CJ4ME_FEATURE_CONTEXT_OFFSET,
         sizeof(float) * CJ4ME_FEATURE_CONTEXT_COUNT);

  if (!dense_relu(scratch->score_input, CJ4ME_SCORE_INPUT_COUNT,
                  &model->score_weights1[0][0], model->score_biases1,
                  CJ4ME_MODEL_HIDDEN1_COUNT, scratch->hidden1) ||
      !dense_relu(scratch->hidden1, CJ4ME_MODEL_HIDDEN1_COUNT,
                  &model->score_weights2[0][0], model->score_biases2,
                  CJ4ME_MODEL_HIDDEN2_COUNT, scratch->hidden2) ||
      !dense_relu(scratch->hidden2, CJ4ME_MODEL_HIDDEN2_COUNT,
                  &model->score_weights3[0][0], model->score_biases3,
                  CJ4ME_MODEL_HIDDEN3_COUNT, scratch->hidden3)) {
    return false;
  }

  value = model->score_biases4[0];
  if (!isfinite(value))
    return false;
  for (size_t in = 0; in < CJ4ME_MODEL_HIDDEN3_COUNT; ++in) {
    const float weight = model->score_weights4[0][in];
    if (!isfinite(weight))
      return false;
    value += scratch->hidden3[in] * weight;
  }
  if (!isfinite(value))
    return false;
  *output = value;
  return true;
}
