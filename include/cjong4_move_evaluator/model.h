#ifndef CJ4ME_MODEL_H
#define CJ4ME_MODEL_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "cjong4_move_evaluator/feature.h"

#ifdef __cplusplus
extern "C" {
#endif

#define CJ4ME_MODEL_FORMAT_VERSION 1u
#define CJ4ME_MODEL_HEADER_SIZE 64u
#define CJ4ME_MODEL_LAYER_COUNT 4u

#define CJ4ME_MODEL_HIDDEN1_COUNT 128u
#define CJ4ME_MODEL_HIDDEN2_COUNT 64u
#define CJ4ME_MODEL_HIDDEN3_COUNT 16u
#define CJ4ME_MODEL_OUTPUT_COUNT 1u
#define CJ4ME_MODEL_I8_REQUANT_COUNT                                           \
  (CJ4ME_MODEL_HIDDEN1_COUNT + CJ4ME_MODEL_HIDDEN2_COUNT +                     \
   CJ4ME_MODEL_HIDDEN3_COUNT)

typedef enum {
  CJ4ME_MODEL_KIND_F32 = 1,
  CJ4ME_MODEL_KIND_I8 = 2
} cj4me_model_kind;

typedef enum {
  CJ4ME_MODEL_TENSOR_F32 = 1,
  CJ4ME_MODEL_TENSOR_I8 = 2,
  CJ4ME_MODEL_TENSOR_I32 = 3
} cj4me_model_tensor_type;

typedef enum {
  CJ4ME_MODEL_TENSOR_F32_W1 = 1,
  CJ4ME_MODEL_TENSOR_F32_B1 = 2,
  CJ4ME_MODEL_TENSOR_F32_W2 = 3,
  CJ4ME_MODEL_TENSOR_F32_B2 = 4,
  CJ4ME_MODEL_TENSOR_F32_W3 = 5,
  CJ4ME_MODEL_TENSOR_F32_B3 = 6,
  CJ4ME_MODEL_TENSOR_F32_W4 = 7,
  CJ4ME_MODEL_TENSOR_F32_B4 = 8,

  CJ4ME_MODEL_TENSOR_I8_W1 = 101,
  CJ4ME_MODEL_TENSOR_I8_B1 = 102,
  CJ4ME_MODEL_TENSOR_I8_WS1 = 103,
  CJ4ME_MODEL_TENSOR_I8_W2 = 104,
  CJ4ME_MODEL_TENSOR_I8_B2 = 105,
  CJ4ME_MODEL_TENSOR_I8_WS2 = 106,
  CJ4ME_MODEL_TENSOR_I8_W3 = 107,
  CJ4ME_MODEL_TENSOR_I8_B3 = 108,
  CJ4ME_MODEL_TENSOR_I8_WS3 = 109,
  CJ4ME_MODEL_TENSOR_I8_W4 = 110,
  CJ4ME_MODEL_TENSOR_I8_B4 = 111,
  CJ4ME_MODEL_TENSOR_I8_WS4 = 112,
  CJ4ME_MODEL_TENSOR_I8_ACTIVATION_SCALES = 113,
  CJ4ME_MODEL_TENSOR_I8_REQUANT_MULTIPLIERS = 114,
  CJ4ME_MODEL_TENSOR_I8_REQUANT_SHIFTS = 115
} cj4me_model_tensor_id;

typedef struct {
  float weights1[CJ4ME_MODEL_HIDDEN1_COUNT][CJ4ME_FEATURE_COUNT];
  float biases1[CJ4ME_MODEL_HIDDEN1_COUNT];
  float weights2[CJ4ME_MODEL_HIDDEN2_COUNT][CJ4ME_MODEL_HIDDEN1_COUNT];
  float biases2[CJ4ME_MODEL_HIDDEN2_COUNT];
  float weights3[CJ4ME_MODEL_HIDDEN3_COUNT][CJ4ME_MODEL_HIDDEN2_COUNT];
  float biases3[CJ4ME_MODEL_HIDDEN3_COUNT];
  float weights4[CJ4ME_MODEL_OUTPUT_COUNT][CJ4ME_MODEL_HIDDEN3_COUNT];
  float biases4[CJ4ME_MODEL_OUTPUT_COUNT];
} cj4me_model_f32;

typedef struct {
  float hidden1[CJ4ME_MODEL_HIDDEN1_COUNT];
  float hidden2[CJ4ME_MODEL_HIDDEN2_COUNT];
  float hidden3[CJ4ME_MODEL_HIDDEN3_COUNT];
} cj4me_inference_f32_scratch;

typedef struct {
  int8_t weights1[CJ4ME_MODEL_HIDDEN1_COUNT][CJ4ME_FEATURE_COUNT];
  int32_t biases1[CJ4ME_MODEL_HIDDEN1_COUNT];
  float weight_scales1[CJ4ME_MODEL_HIDDEN1_COUNT];
  int32_t requant_multipliers1[CJ4ME_MODEL_HIDDEN1_COUNT];
  int32_t requant_shifts1[CJ4ME_MODEL_HIDDEN1_COUNT];

  int8_t weights2[CJ4ME_MODEL_HIDDEN2_COUNT][CJ4ME_MODEL_HIDDEN1_COUNT];
  int32_t biases2[CJ4ME_MODEL_HIDDEN2_COUNT];
  float weight_scales2[CJ4ME_MODEL_HIDDEN2_COUNT];
  int32_t requant_multipliers2[CJ4ME_MODEL_HIDDEN2_COUNT];
  int32_t requant_shifts2[CJ4ME_MODEL_HIDDEN2_COUNT];

  int8_t weights3[CJ4ME_MODEL_HIDDEN3_COUNT][CJ4ME_MODEL_HIDDEN2_COUNT];
  int32_t biases3[CJ4ME_MODEL_HIDDEN3_COUNT];
  float weight_scales3[CJ4ME_MODEL_HIDDEN3_COUNT];
  int32_t requant_multipliers3[CJ4ME_MODEL_HIDDEN3_COUNT];
  int32_t requant_shifts3[CJ4ME_MODEL_HIDDEN3_COUNT];

  int8_t weights4[CJ4ME_MODEL_OUTPUT_COUNT][CJ4ME_MODEL_HIDDEN3_COUNT];
  int32_t biases4[CJ4ME_MODEL_OUTPUT_COUNT];
  float weight_scales4[CJ4ME_MODEL_OUTPUT_COUNT];

  float input_scale;
  float hidden1_scale;
  float hidden2_scale;
  float hidden3_scale;
  float output_scale;
} cj4me_model_i8;

typedef struct {
  int8_t input[CJ4ME_FEATURE_COUNT];
  int8_t hidden1[CJ4ME_MODEL_HIDDEN1_COUNT];
  int8_t hidden2[CJ4ME_MODEL_HIDDEN2_COUNT];
  int8_t hidden3[CJ4ME_MODEL_HIDDEN3_COUNT];
} cj4me_inference_i8_scratch;

typedef struct {
  int32_t quantized;
  float scale;
  float value;
} cj4me_i8_output;

/** Runs the fixed float32 network without allocating memory. */
bool cj4me_infer_f32(const cj4me_model_f32 *model,
                     const float input[CJ4ME_FEATURE_COUNT],
                     cj4me_inference_f32_scratch *scratch, float *output);

/**
 * Quantizes finite float features with the model's input scale.
 * Rounding is nearest with ties away from zero, followed by int8 clamp.
 */
bool cj4me_quantize_input_i8(const cj4me_model_i8 *model,
                             const float input[CJ4ME_FEATURE_COUNT],
                             int8_t output[CJ4ME_FEATURE_COUNT]);

/** Runs the fixed INT8 network without allocating memory. */
bool cj4me_infer_i8(const cj4me_model_i8 *model,
                    const int8_t input[CJ4ME_FEATURE_COUNT],
                    cj4me_inference_i8_scratch *scratch,
                    cj4me_i8_output *output);

/**
 * Loads and validates a float32 model from an exact memory byte span.
 * The destination is cleared on failure.
 */
bool cj4me_model_f32_load_memory(cj4me_model_f32 *model, const void *bytes,
                                 size_t length);

/**
 * Loads and validates an INT8 model from an exact memory byte span.
 * The destination is cleared on failure.
 */
bool cj4me_model_i8_load_memory(cj4me_model_i8 *model, const void *bytes,
                                size_t length);

/**
 * Loads a float32 model from a path through the memory loader.
 * Loading may allocate a temporary exact-size byte buffer.
 */
bool cj4me_model_f32_load_file(cj4me_model_f32 *model, const char *path);

/**
 * Loads an INT8 model from a path through the memory loader.
 * Loading may allocate a temporary exact-size byte buffer.
 */
bool cj4me_model_i8_load_file(cj4me_model_i8 *model, const char *path);

#ifdef __cplusplus
}
#endif

#endif /* CJ4ME_MODEL_H */
