#include "cjong4_move_evaluator/model.h"

#include <limits.h>
#include <math.h>
#include <stddef.h>
#include <string.h>

static int8_t saturate_i8(int64_t value) {
  if (value > INT8_MAX)
    return INT8_MAX;
  if (value < INT8_MIN)
    return INT8_MIN;
  return (int8_t)value;
}

static int64_t rounded_shift_right(int64_t value, uint32_t shift) {
  uint64_t magnitude;
  uint64_t rounded;

  if (shift == 0u)
    return value;
  if (shift > 63u)
    return 0;
  magnitude = value < 0 ? (uint64_t)(-(value + 1)) + 1u : (uint64_t)value;
  rounded = (magnitude >> shift) +
            ((magnitude & (UINT64_C(1) << (shift - 1u))) != 0u);
  if (value >= 0)
    return rounded > (uint64_t)INT64_MAX ? INT64_MAX : (int64_t)rounded;
  if (rounded == UINT64_C(1) << 63)
    return INT64_MIN;
  return -(int64_t)rounded;
}

static int8_t requantize(int32_t value, int32_t multiplier, int32_t shift) {
  int64_t scaled = (int64_t)value * (int64_t)multiplier;

  if (shift >= 0)
    return saturate_i8(rounded_shift_right(scaled, (uint32_t)shift));
  if (shift < -62)
    return scaled == 0 ? 0 : (scaled > 0 ? INT8_MAX : INT8_MIN);
  for (int32_t i = 0; i < -shift; ++i) {
    if (scaled > INT8_MAX || scaled < INT8_MIN)
      return scaled > 0 ? INT8_MAX : INT8_MIN;
    scaled *= 2;
  }
  return saturate_i8(scaled);
}

static bool dense_i8_relu(const int8_t *input, size_t input_count,
                          const int8_t *weights, const int32_t *biases,
                          const int32_t *multipliers, const int32_t *shifts,
                          size_t output_count, int8_t *output) {
  for (size_t out = 0; out < output_count; ++out) {
    int64_t sum = biases[out];
    for (size_t in = 0; in < input_count; ++in)
      sum += (int32_t)input[in] * (int32_t)weights[out * input_count + in];
    if (sum < INT32_MIN || sum > INT32_MAX || multipliers[out] <= 0)
      return false;
    output[out] =
        sum <= 0 ? 0 : requantize((int32_t)sum, multipliers[out], shifts[out]);
    if (output[out] < 0)
      output[out] = 0;
  }
  return true;
}

static bool quantize_values(const float *input, size_t count, float scale,
                            int8_t *output) {
  if (!isfinite(scale) || scale <= 0.0f)
    return false;
  for (size_t i = 0; i < count; ++i) {
    float scaled;
    int32_t rounded;
    if (!isfinite(input[i]))
      return false;
    scaled = input[i] / scale;
    if (!isfinite(scaled)) {
      output[i] = scaled > 0.0f ? INT8_MAX : INT8_MIN;
      continue;
    }
    if (scaled >= (float)INT8_MAX) {
      output[i] = INT8_MAX;
      continue;
    }
    if (scaled <= (float)INT8_MIN) {
      output[i] = INT8_MIN;
      continue;
    }
    rounded = (int32_t)scaled;
    if (scaled - (float)rounded >= 0.5f)
      ++rounded;
    else if (scaled - (float)rounded <= -0.5f)
      --rounded;
    output[i] = saturate_i8(rounded);
  }
  return true;
}

bool cj4me_quantize_input_i8(const cj4me_model_i8 *model,
                             const float input[CJ4ME_FEATURE_COUNT],
                             int8_t output[CJ4ME_FEATURE_COUNT]) {
  if (!model || !input || !output)
    return false;
  return quantize_values(input, CJ4ME_FEATURE_TILE_COUNT,
                         model->tile_input_scale, output) &&
         quantize_values(input + CJ4ME_FEATURE_CONTEXT_OFFSET,
                         CJ4ME_FEATURE_CONTEXT_COUNT, model->score_input_scale,
                         output + CJ4ME_FEATURE_CONTEXT_OFFSET);
}

bool cj4me_infer_i8(const cj4me_model_i8 *model,
                    const int8_t input[CJ4ME_FEATURE_COUNT],
                    cj4me_inference_i8_scratch *scratch,
                    cj4me_i8_output *output) {
  int64_t sum;

  if (!model || !input || !scratch || !output ||
      !isfinite(model->output_scale) || model->output_scale <= 0.0f)
    return false;

  for (size_t tile = 0; tile < CJ4ME_TILE_COUNT; ++tile) {
    const int8_t *tile_input = input + tile * (size_t)CJ4ME_TILE_FEATURE_COUNT;
    int8_t *tile_output =
        scratch->score_input + tile * (size_t)CJ4ME_TILE_EMBEDDING_COUNT;
    if (!dense_i8_relu(tile_input, CJ4ME_TILE_FEATURE_COUNT,
                       &model->tile_weights1[0][0], model->tile_biases1,
                       model->tile_requant_multipliers1,
                       model->tile_requant_shifts1,
                       CJ4ME_MODEL_TILE_HIDDEN_COUNT, scratch->tile_hidden) ||
        !dense_i8_relu(scratch->tile_hidden, CJ4ME_MODEL_TILE_HIDDEN_COUNT,
                       &model->tile_weights2[0][0], model->tile_biases2,
                       model->tile_requant_multipliers2,
                       model->tile_requant_shifts2, CJ4ME_TILE_EMBEDDING_COUNT,
                       tile_output)) {
      return false;
    }
  }
  memcpy(scratch->score_input +
             CJ4ME_TILE_COUNT * (size_t)CJ4ME_TILE_EMBEDDING_COUNT,
         input + CJ4ME_FEATURE_CONTEXT_OFFSET,
         sizeof(int8_t) * CJ4ME_FEATURE_CONTEXT_COUNT);

  if (!dense_i8_relu(scratch->score_input, CJ4ME_SCORE_INPUT_COUNT,
                     &model->score_weights1[0][0], model->score_biases1,
                     model->score_requant_multipliers1,
                     model->score_requant_shifts1, CJ4ME_MODEL_HIDDEN1_COUNT,
                     scratch->hidden1) ||
      !dense_i8_relu(scratch->hidden1, CJ4ME_MODEL_HIDDEN1_COUNT,
                     &model->score_weights2[0][0], model->score_biases2,
                     model->score_requant_multipliers2,
                     model->score_requant_shifts2, CJ4ME_MODEL_HIDDEN2_COUNT,
                     scratch->hidden2) ||
      !dense_i8_relu(scratch->hidden2, CJ4ME_MODEL_HIDDEN2_COUNT,
                     &model->score_weights3[0][0], model->score_biases3,
                     model->score_requant_multipliers3,
                     model->score_requant_shifts3, CJ4ME_MODEL_HIDDEN3_COUNT,
                     scratch->hidden3)) {
    return false;
  }

  sum = model->score_biases4[0];
  for (size_t in = 0; in < CJ4ME_MODEL_HIDDEN3_COUNT; ++in)
    sum +=
        (int32_t)scratch->hidden3[in] * (int32_t)model->score_weights4[0][in];
  if (sum < INT32_MIN || sum > INT32_MAX)
    return false;
  output->quantized = (int32_t)sum;
  output->scale = model->output_scale;
  output->value = (float)output->quantized * output->scale;
  return isfinite(output->value);
}
