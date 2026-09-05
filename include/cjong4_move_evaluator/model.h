#ifndef CJ4ME_MODEL_H
#define CJ4ME_MODEL_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "cjong4_move_evaluator/feature.h"

#ifdef __cplusplus
extern "C" {
#endif

#define CJ4ME_MODEL_FORMAT_VERSION 2u
#define CJ4ME_MODEL_HEADER_SIZE 88u
#define CJ4ME_MODEL_LAYER_COUNT 6u

#define CJ4ME_MODEL_TILE_HIDDEN_COUNT 16u
#define CJ4ME_MODEL_HIDDEN1_COUNT 128u
#define CJ4ME_MODEL_HIDDEN2_COUNT 64u
#define CJ4ME_MODEL_HIDDEN3_COUNT 16u
#define CJ4ME_MODEL_OUTPUT_COUNT 1u
#define CJ4ME_MODEL_I8_REQUANT_COUNT                                           \
  (CJ4ME_MODEL_TILE_HIDDEN_COUNT + CJ4ME_TILE_EMBEDDING_COUNT +                \
   CJ4ME_MODEL_HIDDEN1_COUNT + CJ4ME_MODEL_HIDDEN2_COUNT +                     \
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
  CJ4ME_MODEL_TENSOR_F32_TILE_W1 = 1,
  CJ4ME_MODEL_TENSOR_F32_TILE_B1 = 2,
  CJ4ME_MODEL_TENSOR_F32_TILE_W2 = 3,
  CJ4ME_MODEL_TENSOR_F32_TILE_B2 = 4,
  CJ4ME_MODEL_TENSOR_F32_SCORE_W1 = 5,
  CJ4ME_MODEL_TENSOR_F32_SCORE_B1 = 6,
  CJ4ME_MODEL_TENSOR_F32_SCORE_W2 = 7,
  CJ4ME_MODEL_TENSOR_F32_SCORE_B2 = 8,
  CJ4ME_MODEL_TENSOR_F32_SCORE_W3 = 9,
  CJ4ME_MODEL_TENSOR_F32_SCORE_B3 = 10,
  CJ4ME_MODEL_TENSOR_F32_SCORE_W4 = 11,
  CJ4ME_MODEL_TENSOR_F32_SCORE_B4 = 12,

  CJ4ME_MODEL_TENSOR_I8_TILE_W1 = 101,
  CJ4ME_MODEL_TENSOR_I8_TILE_B1 = 102,
  CJ4ME_MODEL_TENSOR_I8_TILE_WS1 = 103,
  CJ4ME_MODEL_TENSOR_I8_TILE_W2 = 104,
  CJ4ME_MODEL_TENSOR_I8_TILE_B2 = 105,
  CJ4ME_MODEL_TENSOR_I8_TILE_WS2 = 106,
  CJ4ME_MODEL_TENSOR_I8_SCORE_W1 = 107,
  CJ4ME_MODEL_TENSOR_I8_SCORE_B1 = 108,
  CJ4ME_MODEL_TENSOR_I8_SCORE_WS1 = 109,
  CJ4ME_MODEL_TENSOR_I8_SCORE_W2 = 110,
  CJ4ME_MODEL_TENSOR_I8_SCORE_B2 = 111,
  CJ4ME_MODEL_TENSOR_I8_SCORE_WS2 = 112,
  CJ4ME_MODEL_TENSOR_I8_SCORE_W3 = 113,
  CJ4ME_MODEL_TENSOR_I8_SCORE_B3 = 114,
  CJ4ME_MODEL_TENSOR_I8_SCORE_WS3 = 115,
  CJ4ME_MODEL_TENSOR_I8_SCORE_W4 = 116,
  CJ4ME_MODEL_TENSOR_I8_SCORE_B4 = 117,
  CJ4ME_MODEL_TENSOR_I8_SCORE_WS4 = 118,
  CJ4ME_MODEL_TENSOR_I8_ACTIVATION_SCALES = 119,
  CJ4ME_MODEL_TENSOR_I8_REQUANT_MULTIPLIERS = 120,
  CJ4ME_MODEL_TENSOR_I8_REQUANT_SHIFTS = 121
} cj4me_model_tensor_id;

typedef struct {
  float tile_weights1[CJ4ME_MODEL_TILE_HIDDEN_COUNT][CJ4ME_TILE_FEATURE_COUNT];
  float tile_biases1[CJ4ME_MODEL_TILE_HIDDEN_COUNT];
  float tile_weights2[CJ4ME_TILE_EMBEDDING_COUNT]
                     [CJ4ME_MODEL_TILE_HIDDEN_COUNT];
  float tile_biases2[CJ4ME_TILE_EMBEDDING_COUNT];

  float score_weights1[CJ4ME_MODEL_HIDDEN1_COUNT][CJ4ME_SCORE_INPUT_COUNT];
  float score_biases1[CJ4ME_MODEL_HIDDEN1_COUNT];
  float score_weights2[CJ4ME_MODEL_HIDDEN2_COUNT][CJ4ME_MODEL_HIDDEN1_COUNT];
  float score_biases2[CJ4ME_MODEL_HIDDEN2_COUNT];
  float score_weights3[CJ4ME_MODEL_HIDDEN3_COUNT][CJ4ME_MODEL_HIDDEN2_COUNT];
  float score_biases3[CJ4ME_MODEL_HIDDEN3_COUNT];
  float score_weights4[CJ4ME_MODEL_OUTPUT_COUNT][CJ4ME_MODEL_HIDDEN3_COUNT];
  float score_biases4[CJ4ME_MODEL_OUTPUT_COUNT];
} cj4me_model_f32;

typedef struct {
  float tile_hidden[CJ4ME_MODEL_TILE_HIDDEN_COUNT];
  float score_input[CJ4ME_SCORE_INPUT_COUNT];
  float hidden1[CJ4ME_MODEL_HIDDEN1_COUNT];
  float hidden2[CJ4ME_MODEL_HIDDEN2_COUNT];
  float hidden3[CJ4ME_MODEL_HIDDEN3_COUNT];
} cj4me_inference_f32_scratch;

typedef struct {
  int8_t tile_weights1[CJ4ME_MODEL_TILE_HIDDEN_COUNT][CJ4ME_TILE_FEATURE_COUNT];
  int32_t tile_biases1[CJ4ME_MODEL_TILE_HIDDEN_COUNT];
  float tile_weight_scales1[CJ4ME_MODEL_TILE_HIDDEN_COUNT];
  int32_t tile_requant_multipliers1[CJ4ME_MODEL_TILE_HIDDEN_COUNT];
  int32_t tile_requant_shifts1[CJ4ME_MODEL_TILE_HIDDEN_COUNT];

  int8_t tile_weights2[CJ4ME_TILE_EMBEDDING_COUNT]
                      [CJ4ME_MODEL_TILE_HIDDEN_COUNT];
  int32_t tile_biases2[CJ4ME_TILE_EMBEDDING_COUNT];
  float tile_weight_scales2[CJ4ME_TILE_EMBEDDING_COUNT];
  int32_t tile_requant_multipliers2[CJ4ME_TILE_EMBEDDING_COUNT];
  int32_t tile_requant_shifts2[CJ4ME_TILE_EMBEDDING_COUNT];

  int8_t score_weights1[CJ4ME_MODEL_HIDDEN1_COUNT][CJ4ME_SCORE_INPUT_COUNT];
  int32_t score_biases1[CJ4ME_MODEL_HIDDEN1_COUNT];
  float score_weight_scales1[CJ4ME_MODEL_HIDDEN1_COUNT];
  int32_t score_requant_multipliers1[CJ4ME_MODEL_HIDDEN1_COUNT];
  int32_t score_requant_shifts1[CJ4ME_MODEL_HIDDEN1_COUNT];

  int8_t score_weights2[CJ4ME_MODEL_HIDDEN2_COUNT][CJ4ME_MODEL_HIDDEN1_COUNT];
  int32_t score_biases2[CJ4ME_MODEL_HIDDEN2_COUNT];
  float score_weight_scales2[CJ4ME_MODEL_HIDDEN2_COUNT];
  int32_t score_requant_multipliers2[CJ4ME_MODEL_HIDDEN2_COUNT];
  int32_t score_requant_shifts2[CJ4ME_MODEL_HIDDEN2_COUNT];

  int8_t score_weights3[CJ4ME_MODEL_HIDDEN3_COUNT][CJ4ME_MODEL_HIDDEN2_COUNT];
  int32_t score_biases3[CJ4ME_MODEL_HIDDEN3_COUNT];
  float score_weight_scales3[CJ4ME_MODEL_HIDDEN3_COUNT];
  int32_t score_requant_multipliers3[CJ4ME_MODEL_HIDDEN3_COUNT];
  int32_t score_requant_shifts3[CJ4ME_MODEL_HIDDEN3_COUNT];

  int8_t score_weights4[CJ4ME_MODEL_OUTPUT_COUNT][CJ4ME_MODEL_HIDDEN3_COUNT];
  int32_t score_biases4[CJ4ME_MODEL_OUTPUT_COUNT];
  float score_weight_scales4[CJ4ME_MODEL_OUTPUT_COUNT];

  float tile_input_scale;
  float tile_hidden_scale;
  float score_input_scale;
  float hidden1_scale;
  float hidden2_scale;
  float hidden3_scale;
  float output_scale;
} cj4me_model_i8;

typedef struct {
  int8_t input[CJ4ME_FEATURE_COUNT];
  int8_t tile_hidden[CJ4ME_MODEL_TILE_HIDDEN_COUNT];
  int8_t score_input[CJ4ME_SCORE_INPUT_COUNT];
  int8_t hidden1[CJ4ME_MODEL_HIDDEN1_COUNT];
  int8_t hidden2[CJ4ME_MODEL_HIDDEN2_COUNT];
  int8_t hidden3[CJ4ME_MODEL_HIDDEN3_COUNT];
} cj4me_inference_i8_scratch;

typedef struct {
  int32_t quantized;
  float scale;
  float value;
} cj4me_i8_output;

bool cj4me_infer_f32(const cj4me_model_f32 *model,
                     const float input[CJ4ME_FEATURE_COUNT],
                     cj4me_inference_f32_scratch *scratch, float *output);

bool cj4me_quantize_input_i8(const cj4me_model_i8 *model,
                             const float input[CJ4ME_FEATURE_COUNT],
                             int8_t output[CJ4ME_FEATURE_COUNT]);

bool cj4me_infer_i8(const cj4me_model_i8 *model,
                    const int8_t input[CJ4ME_FEATURE_COUNT],
                    cj4me_inference_i8_scratch *scratch,
                    cj4me_i8_output *output);

bool cj4me_model_f32_load_memory(cj4me_model_f32 *model, const void *bytes,
                                 size_t length);
bool cj4me_model_i8_load_memory(cj4me_model_i8 *model, const void *bytes,
                                size_t length);
bool cj4me_model_f32_load_file(cj4me_model_f32 *model, const char *path);
bool cj4me_model_i8_load_file(cj4me_model_i8 *model, const char *path);

#ifdef __cplusplus
}
#endif

#endif /* CJ4ME_MODEL_H */
