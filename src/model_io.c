#include "cjong4_move_evaluator/model.h"

#include <limits.h>
#include <math.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const uint8_t MODEL_MAGIC[8] = {'C', 'J', '4', 'M', 'E', 'M', '0', '1'};

enum {
  F32_TENSOR_COUNT = 8,
  I8_TENSOR_COUNT = 15,
  I8_ACTIVATION_SCALE_COUNT = 5
};

#define TENSOR_HEADER_SIZE 16u
#define F32_ELEMENT_SIZE 4u
#define I32_ELEMENT_SIZE 4u
#define I8_ELEMENT_SIZE 1u

typedef struct {
  const uint8_t *next;
  size_t remaining;
} byte_reader;

_Static_assert(sizeof(float) == 4u, "model format requires 32-bit float");
_Static_assert(sizeof(int8_t) == 1u, "model format requires 8-bit int8_t");
_Static_assert(sizeof(int32_t) == 4u, "model format requires 32-bit int32_t");
_Static_assert(INT8_MIN == -128, "model format requires two's-complement int8");
_Static_assert(INT32_MIN == (-INT32_MAX - 1),
               "model format requires two's-complement int32");

static uint32_t read_u32_le(const uint8_t bytes[4]) {
  return (uint32_t)bytes[0] | ((uint32_t)bytes[1] << 8) |
         ((uint32_t)bytes[2] << 16) | ((uint32_t)bytes[3] << 24);
}

static uint64_t read_u64_le(const uint8_t bytes[8]) {
  return (uint64_t)read_u32_le(bytes) |
         ((uint64_t)read_u32_le(bytes + 4) << 32);
}

static bool take_bytes(byte_reader *reader, size_t count,
                       const uint8_t **bytes) {
  if (count > reader->remaining)
    return false;
  *bytes = reader->next;
  reader->next += count;
  reader->remaining -= count;
  return true;
}

static bool read_header(byte_reader *reader, const void *bytes, size_t length,
                        cj4me_model_kind expected_kind,
                        uint32_t expected_tensor_count) {
  const uint8_t *header;

  if (!bytes || length < CJ4ME_MODEL_HEADER_SIZE)
    return false;
  reader->next = (const uint8_t *)bytes;
  reader->remaining = length;
  if (!take_bytes(reader, CJ4ME_MODEL_HEADER_SIZE, &header) ||
      memcmp(header, MODEL_MAGIC, sizeof(MODEL_MAGIC)) != 0 ||
      read_u32_le(header + 8) != CJ4ME_MODEL_FORMAT_VERSION ||
      read_u32_le(header + 12) != CJ4ME_FEATURE_SCHEMA_VERSION ||
      read_u32_le(header + 16) != (uint32_t)expected_kind ||
      read_u32_le(header + 20) != CJ4ME_MODEL_LAYER_COUNT ||
      read_u32_le(header + 24) != CJ4ME_FEATURE_COUNT ||
      read_u32_le(header + 28) != CJ4ME_MODEL_HIDDEN1_COUNT ||
      read_u32_le(header + 32) != CJ4ME_MODEL_HIDDEN2_COUNT ||
      read_u32_le(header + 36) != CJ4ME_MODEL_HIDDEN3_COUNT ||
      read_u32_le(header + 40) != CJ4ME_MODEL_OUTPUT_COUNT ||
      read_u32_le(header + 44) != expected_tensor_count ||
      read_u64_le(header + 48) != (uint64_t)reader->remaining ||
      read_u32_le(header + 56) != CJ4ME_MODEL_HEADER_SIZE ||
      read_u32_le(header + 60) != 0u) {
    return false;
  }
  return true;
}

static bool read_tensor_header(byte_reader *reader, uint32_t expected_id,
                               cj4me_model_tensor_type expected_type,
                               uint32_t expected_count, const uint8_t **data) {
  const uint8_t *header;
  uint32_t element_size;
  uint64_t expected_bytes;

  if (!take_bytes(reader, 16u, &header))
    return false;
  element_size = expected_type == CJ4ME_MODEL_TENSOR_I8 ? 1u : 4u;
  expected_bytes = (uint64_t)expected_count * element_size;
  if (expected_bytes > UINT32_MAX || read_u32_le(header) != expected_id ||
      read_u32_le(header + 4) != (uint32_t)expected_type ||
      read_u32_le(header + 8) != expected_count ||
      read_u32_le(header + 12) != (uint32_t)expected_bytes) {
    return false;
  }
  return take_bytes(reader, (size_t)expected_bytes, data);
}

static bool read_f32_tensor(byte_reader *reader, uint32_t id, uint32_t count,
                            float *output, bool require_positive) {
  const uint8_t *data;

  if (!read_tensor_header(reader, id, CJ4ME_MODEL_TENSOR_F32, count, &data)) {
    return false;
  }
  for (uint32_t i = 0; i < count; ++i) {
    uint32_t bits = read_u32_le(data + (size_t)i * 4u);
    memcpy(output + i, &bits, sizeof(bits));
    if (!isfinite(output[i]) || (require_positive && output[i] <= 0.0f)) {
      return false;
    }
  }
  return true;
}

static bool read_i32_tensor(byte_reader *reader, uint32_t id, uint32_t count,
                            int32_t *output) {
  const uint8_t *data;

  if (!read_tensor_header(reader, id, CJ4ME_MODEL_TENSOR_I32, count, &data)) {
    return false;
  }
  for (uint32_t i = 0; i < count; ++i) {
    const uint32_t value = read_u32_le(data + (size_t)i * 4u);
    output[i] = value <= INT32_MAX ? (int32_t)value
                                   : -(int32_t)(UINT32_MAX - value) - 1;
  }
  return true;
}

static bool read_i8_tensor(byte_reader *reader, uint32_t id, uint32_t count,
                           int8_t *output) {
  const uint8_t *data;

  if (!read_tensor_header(reader, id, CJ4ME_MODEL_TENSOR_I8, count, &data)) {
    return false;
  }
  for (uint32_t i = 0; i < count; ++i) {
    output[i] = data[i] <= INT8_MAX
                    ? (int8_t)data[i]
                    : (int8_t)(-(int16_t)(UINT8_MAX - data[i]) - 1);
  }
  return true;
}

static bool accumulators_fit_i32(const int8_t *weights, const int32_t *biases,
                                 size_t input_count, size_t output_count) {
  for (size_t out = 0; out < output_count; ++out) {
    int64_t minimum = biases[out];
    int64_t maximum = biases[out];
    for (size_t in = 0; in < input_count; ++in) {
      const int32_t weight = (int32_t)weights[out * input_count + in];
      if (weight >= 0) {
        minimum += -128 * weight;
        maximum += 127 * weight;
      } else {
        minimum += 127 * weight;
        maximum += -128 * weight;
      }
    }
    if (minimum < INT32_MIN || maximum > INT32_MAX)
      return false;
  }
  return true;
}

static bool valid_requantization(const int32_t *multipliers,
                                 const int32_t *shifts, size_t count) {
  for (size_t i = 0; i < count; ++i) {
    if (multipliers[i] <= 0 || shifts[i] < -62 || shifts[i] > 62)
      return false;
  }
  return true;
}

static bool valid_output_scale(float hidden3_scale, float weight_scale,
                               float output_scale) {
  float expected = hidden3_scale * weight_scale;
  float difference;
  float reference;

  if (!isfinite(expected) || expected <= 0.0f)
    return false;
  difference = output_scale >= expected ? output_scale - expected
                                        : expected - output_scale;
  reference = output_scale >= expected ? output_scale : expected;
  return difference <= reference * 1.0e-6f;
}

static size_t f32_file_size(void) {
  const size_t elements =
      CJ4ME_MODEL_HIDDEN1_COUNT * CJ4ME_FEATURE_COUNT +
      CJ4ME_MODEL_HIDDEN1_COUNT +
      CJ4ME_MODEL_HIDDEN2_COUNT * CJ4ME_MODEL_HIDDEN1_COUNT +
      CJ4ME_MODEL_HIDDEN2_COUNT +
      CJ4ME_MODEL_HIDDEN3_COUNT * CJ4ME_MODEL_HIDDEN2_COUNT +
      CJ4ME_MODEL_HIDDEN3_COUNT +
      CJ4ME_MODEL_OUTPUT_COUNT * CJ4ME_MODEL_HIDDEN3_COUNT +
      CJ4ME_MODEL_OUTPUT_COUNT;
  return CJ4ME_MODEL_HEADER_SIZE + F32_TENSOR_COUNT * TENSOR_HEADER_SIZE +
         elements * F32_ELEMENT_SIZE;
}

static size_t i8_file_size(void) {
  const size_t weight_elements =
      CJ4ME_MODEL_HIDDEN1_COUNT * CJ4ME_FEATURE_COUNT +
      CJ4ME_MODEL_HIDDEN2_COUNT * CJ4ME_MODEL_HIDDEN1_COUNT +
      CJ4ME_MODEL_HIDDEN3_COUNT * CJ4ME_MODEL_HIDDEN2_COUNT +
      CJ4ME_MODEL_OUTPUT_COUNT * CJ4ME_MODEL_HIDDEN3_COUNT;
  const size_t channel_count =
      CJ4ME_MODEL_HIDDEN1_COUNT + CJ4ME_MODEL_HIDDEN2_COUNT +
      CJ4ME_MODEL_HIDDEN3_COUNT + CJ4ME_MODEL_OUTPUT_COUNT;
  return CJ4ME_MODEL_HEADER_SIZE + I8_TENSOR_COUNT * TENSOR_HEADER_SIZE +
         weight_elements * I8_ELEMENT_SIZE + channel_count * I32_ELEMENT_SIZE +
         channel_count * F32_ELEMENT_SIZE +
         I8_ACTIVATION_SCALE_COUNT * F32_ELEMENT_SIZE +
         2u * CJ4ME_MODEL_I8_REQUANT_COUNT * I32_ELEMENT_SIZE;
}

static bool read_file_bytes(const char *path, size_t expected_size,
                            uint8_t **bytes) {
  FILE *file;
  long file_size;
  bool ok;

  *bytes = NULL;
  if (!path || expected_size == 0u || expected_size > (size_t)LONG_MAX)
    return false;
  file = fopen(path, "rb");
  if (!file)
    return false;
  ok = fseek(file, 0L, SEEK_END) == 0;
  file_size = ok ? ftell(file) : -1L;
  ok = ok && file_size >= 0 && (size_t)file_size == expected_size &&
       fseek(file, 0L, SEEK_SET) == 0;
  if (ok) {
    *bytes = (uint8_t *)malloc(expected_size);
    ok = *bytes != NULL &&
         fread(*bytes, 1u, expected_size, file) == expected_size &&
         fgetc(file) == EOF && !ferror(file);
  }
  if (fclose(file) != 0)
    ok = false;
  if (!ok) {
    free(*bytes);
    *bytes = NULL;
  }
  return ok;
}

bool cj4me_model_f32_load_memory(cj4me_model_f32 *model, const void *bytes,
                                 size_t length) {
  byte_reader reader;
  bool ok;

  if (!model)
    return false;
  memset(model, 0, sizeof(*model));
  ok = read_header(&reader, bytes, length, CJ4ME_MODEL_KIND_F32,
                   F32_TENSOR_COUNT) &&
       read_f32_tensor(&reader, CJ4ME_MODEL_TENSOR_F32_W1,
                       CJ4ME_MODEL_HIDDEN1_COUNT * CJ4ME_FEATURE_COUNT,
                       &model->weights1[0][0], false) &&
       read_f32_tensor(&reader, CJ4ME_MODEL_TENSOR_F32_B1,
                       CJ4ME_MODEL_HIDDEN1_COUNT, model->biases1, false) &&
       read_f32_tensor(&reader, CJ4ME_MODEL_TENSOR_F32_W2,
                       CJ4ME_MODEL_HIDDEN2_COUNT * CJ4ME_MODEL_HIDDEN1_COUNT,
                       &model->weights2[0][0], false) &&
       read_f32_tensor(&reader, CJ4ME_MODEL_TENSOR_F32_B2,
                       CJ4ME_MODEL_HIDDEN2_COUNT, model->biases2, false) &&
       read_f32_tensor(&reader, CJ4ME_MODEL_TENSOR_F32_W3,
                       CJ4ME_MODEL_HIDDEN3_COUNT * CJ4ME_MODEL_HIDDEN2_COUNT,
                       &model->weights3[0][0], false) &&
       read_f32_tensor(&reader, CJ4ME_MODEL_TENSOR_F32_B3,
                       CJ4ME_MODEL_HIDDEN3_COUNT, model->biases3, false) &&
       read_f32_tensor(&reader, CJ4ME_MODEL_TENSOR_F32_W4,
                       CJ4ME_MODEL_OUTPUT_COUNT * CJ4ME_MODEL_HIDDEN3_COUNT,
                       &model->weights4[0][0], false) &&
       read_f32_tensor(&reader, CJ4ME_MODEL_TENSOR_F32_B4,
                       CJ4ME_MODEL_OUTPUT_COUNT, model->biases4, false) &&
       reader.remaining == 0u;
  if (!ok)
    memset(model, 0, sizeof(*model));
  return ok;
}

bool cj4me_model_i8_load_memory(cj4me_model_i8 *model, const void *bytes,
                                size_t length) {
  byte_reader reader;
  float activation_scales[I8_ACTIVATION_SCALE_COUNT];
  int32_t multipliers[CJ4ME_MODEL_I8_REQUANT_COUNT];
  int32_t shifts[CJ4ME_MODEL_I8_REQUANT_COUNT];
  bool ok;

  if (!model)
    return false;
  memset(model, 0, sizeof(*model));
  ok =
      read_header(&reader, bytes, length, CJ4ME_MODEL_KIND_I8,
                  I8_TENSOR_COUNT) &&
      read_i8_tensor(&reader, CJ4ME_MODEL_TENSOR_I8_W1,
                     CJ4ME_MODEL_HIDDEN1_COUNT * CJ4ME_FEATURE_COUNT,
                     &model->weights1[0][0]) &&
      read_i32_tensor(&reader, CJ4ME_MODEL_TENSOR_I8_B1,
                      CJ4ME_MODEL_HIDDEN1_COUNT, model->biases1) &&
      read_f32_tensor(&reader, CJ4ME_MODEL_TENSOR_I8_WS1,
                      CJ4ME_MODEL_HIDDEN1_COUNT, model->weight_scales1, true) &&
      read_i8_tensor(&reader, CJ4ME_MODEL_TENSOR_I8_W2,
                     CJ4ME_MODEL_HIDDEN2_COUNT * CJ4ME_MODEL_HIDDEN1_COUNT,
                     &model->weights2[0][0]) &&
      read_i32_tensor(&reader, CJ4ME_MODEL_TENSOR_I8_B2,
                      CJ4ME_MODEL_HIDDEN2_COUNT, model->biases2) &&
      read_f32_tensor(&reader, CJ4ME_MODEL_TENSOR_I8_WS2,
                      CJ4ME_MODEL_HIDDEN2_COUNT, model->weight_scales2, true) &&
      read_i8_tensor(&reader, CJ4ME_MODEL_TENSOR_I8_W3,
                     CJ4ME_MODEL_HIDDEN3_COUNT * CJ4ME_MODEL_HIDDEN2_COUNT,
                     &model->weights3[0][0]) &&
      read_i32_tensor(&reader, CJ4ME_MODEL_TENSOR_I8_B3,
                      CJ4ME_MODEL_HIDDEN3_COUNT, model->biases3) &&
      read_f32_tensor(&reader, CJ4ME_MODEL_TENSOR_I8_WS3,
                      CJ4ME_MODEL_HIDDEN3_COUNT, model->weight_scales3, true) &&
      read_i8_tensor(&reader, CJ4ME_MODEL_TENSOR_I8_W4,
                     CJ4ME_MODEL_OUTPUT_COUNT * CJ4ME_MODEL_HIDDEN3_COUNT,
                     &model->weights4[0][0]) &&
      read_i32_tensor(&reader, CJ4ME_MODEL_TENSOR_I8_B4,
                      CJ4ME_MODEL_OUTPUT_COUNT, model->biases4) &&
      read_f32_tensor(&reader, CJ4ME_MODEL_TENSOR_I8_WS4,
                      CJ4ME_MODEL_OUTPUT_COUNT, model->weight_scales4, true) &&
      read_f32_tensor(&reader, CJ4ME_MODEL_TENSOR_I8_ACTIVATION_SCALES,
                      I8_ACTIVATION_SCALE_COUNT, activation_scales, true) &&
      read_i32_tensor(&reader, CJ4ME_MODEL_TENSOR_I8_REQUANT_MULTIPLIERS,
                      CJ4ME_MODEL_I8_REQUANT_COUNT, multipliers) &&
      read_i32_tensor(&reader, CJ4ME_MODEL_TENSOR_I8_REQUANT_SHIFTS,
                      CJ4ME_MODEL_I8_REQUANT_COUNT, shifts) &&
      reader.remaining == 0u &&
      valid_requantization(multipliers, shifts, CJ4ME_MODEL_I8_REQUANT_COUNT) &&
      valid_output_scale(activation_scales[3], model->weight_scales4[0],
                         activation_scales[4]) &&
      accumulators_fit_i32(&model->weights1[0][0], model->biases1,
                           CJ4ME_FEATURE_COUNT, CJ4ME_MODEL_HIDDEN1_COUNT) &&
      accumulators_fit_i32(&model->weights2[0][0], model->biases2,
                           CJ4ME_MODEL_HIDDEN1_COUNT,
                           CJ4ME_MODEL_HIDDEN2_COUNT) &&
      accumulators_fit_i32(&model->weights3[0][0], model->biases3,
                           CJ4ME_MODEL_HIDDEN2_COUNT,
                           CJ4ME_MODEL_HIDDEN3_COUNT) &&
      accumulators_fit_i32(&model->weights4[0][0], model->biases4,
                           CJ4ME_MODEL_HIDDEN3_COUNT, CJ4ME_MODEL_OUTPUT_COUNT);

  if (!ok) {
    memset(model, 0, sizeof(*model));
    return false;
  }

  model->input_scale = activation_scales[0];
  model->hidden1_scale = activation_scales[1];
  model->hidden2_scale = activation_scales[2];
  model->hidden3_scale = activation_scales[3];
  model->output_scale = activation_scales[4];
  memcpy(model->requant_multipliers1, multipliers,
         sizeof(model->requant_multipliers1));
  memcpy(model->requant_multipliers2, multipliers + CJ4ME_MODEL_HIDDEN1_COUNT,
         sizeof(model->requant_multipliers2));
  memcpy(model->requant_multipliers3,
         multipliers + CJ4ME_MODEL_HIDDEN1_COUNT + CJ4ME_MODEL_HIDDEN2_COUNT,
         sizeof(model->requant_multipliers3));
  memcpy(model->requant_shifts1, shifts, sizeof(model->requant_shifts1));
  memcpy(model->requant_shifts2, shifts + CJ4ME_MODEL_HIDDEN1_COUNT,
         sizeof(model->requant_shifts2));
  memcpy(model->requant_shifts3,
         shifts + CJ4ME_MODEL_HIDDEN1_COUNT + CJ4ME_MODEL_HIDDEN2_COUNT,
         sizeof(model->requant_shifts3));
  return true;
}

bool cj4me_model_f32_load_file(cj4me_model_f32 *model, const char *path) {
  uint8_t *bytes;
  const size_t length = f32_file_size();
  bool ok;

  if (!model)
    return false;
  memset(model, 0, sizeof(*model));
  if (!read_file_bytes(path, length, &bytes))
    return false;
  ok = cj4me_model_f32_load_memory(model, bytes, length);
  free(bytes);
  if (!ok)
    memset(model, 0, sizeof(*model));
  return ok;
}

bool cj4me_model_i8_load_file(cj4me_model_i8 *model, const char *path) {
  uint8_t *bytes;
  const size_t length = i8_file_size();
  bool ok;

  if (!model)
    return false;
  memset(model, 0, sizeof(*model));
  if (!read_file_bytes(path, length, &bytes))
    return false;
  ok = cj4me_model_i8_load_memory(model, bytes, length);
  free(bytes);
  if (!ok)
    memset(model, 0, sizeof(*model));
  return ok;
}
