#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "cjong4_move_evaluator/model.h"

typedef struct {
  uint8_t *bytes;
  size_t length;
  size_t capacity;
} test_buffer;

static void put_u32(uint8_t *output, uint32_t value) {
  output[0] = (uint8_t)value;
  output[1] = (uint8_t)(value >> 8);
  output[2] = (uint8_t)(value >> 16);
  output[3] = (uint8_t)(value >> 24);
}

static void put_u64(uint8_t *output, uint64_t value) {
  put_u32(output, (uint32_t)value);
  put_u32(output + 4, (uint32_t)(value >> 32));
}

static void put_f32(uint8_t *output, float value) {
  uint32_t bits;
  memcpy(&bits, &value, sizeof(bits));
  put_u32(output, bits);
}

static void append_bytes(test_buffer *buffer, const void *bytes, size_t count) {
  assert(buffer->length + count <= buffer->capacity);
  memcpy(buffer->bytes + buffer->length, bytes, count);
  buffer->length += count;
}

static void append_tensor_header(test_buffer *buffer, uint32_t id,
                                 uint32_t type, uint32_t count) {
  uint8_t header[16];
  const uint32_t element_size = type == CJ4ME_MODEL_TENSOR_I8 ? 1u : 4u;
  put_u32(header, id);
  put_u32(header + 4, type);
  put_u32(header + 8, count);
  put_u32(header + 12, count * element_size);
  append_bytes(buffer, header, sizeof(header));
}

static void append_zero_tensor(test_buffer *buffer, uint32_t id, uint32_t type,
                               uint32_t count) {
  const size_t element_size = type == CJ4ME_MODEL_TENSOR_I8 ? 1u : 4u;
  append_tensor_header(buffer, id, type, count);
  memset(buffer->bytes + buffer->length, 0, count * element_size);
  buffer->length += count * element_size;
}

static void append_f32_ones(test_buffer *buffer, uint32_t id, uint32_t count) {
  append_tensor_header(buffer, id, CJ4ME_MODEL_TENSOR_F32, count);
  for (uint32_t i = 0; i < count; ++i) {
    uint8_t encoded[4];
    put_f32(encoded, 1.0f);
    append_bytes(buffer, encoded, sizeof(encoded));
  }
}

static void append_i32_ones(test_buffer *buffer, uint32_t id, uint32_t count) {
  append_tensor_header(buffer, id, CJ4ME_MODEL_TENSOR_I32, count);
  for (uint32_t i = 0; i < count; ++i) {
    uint8_t encoded[4];
    put_u32(encoded, 1u);
    append_bytes(buffer, encoded, sizeof(encoded));
  }
}

static void finish_header(test_buffer *buffer, cj4me_model_kind kind,
                          uint32_t tensor_count) {
  memcpy(buffer->bytes, "CJ4MEM02", 8u);
  put_u32(buffer->bytes + 8, CJ4ME_MODEL_FORMAT_VERSION);
  put_u32(buffer->bytes + 12, CJ4ME_FEATURE_SCHEMA_VERSION);
  put_u32(buffer->bytes + 16, (uint32_t)kind);
  put_u32(buffer->bytes + 20, CJ4ME_MODEL_LAYER_COUNT);
  put_u32(buffer->bytes + 24, CJ4ME_TILE_COUNT);
  put_u32(buffer->bytes + 28, CJ4ME_TILE_FEATURE_COUNT);
  put_u32(buffer->bytes + 32, CJ4ME_MODEL_TILE_HIDDEN_COUNT);
  put_u32(buffer->bytes + 36, CJ4ME_TILE_EMBEDDING_COUNT);
  put_u32(buffer->bytes + 40, CJ4ME_FEATURE_STATE_COUNT);
  put_u32(buffer->bytes + 44, CJ4ME_FEATURE_ACTION_COUNT);
  put_u32(buffer->bytes + 48, CJ4ME_SCORE_INPUT_COUNT);
  put_u32(buffer->bytes + 52, CJ4ME_MODEL_HIDDEN1_COUNT);
  put_u32(buffer->bytes + 56, CJ4ME_MODEL_HIDDEN2_COUNT);
  put_u32(buffer->bytes + 60, CJ4ME_MODEL_HIDDEN3_COUNT);
  put_u32(buffer->bytes + 64, CJ4ME_MODEL_OUTPUT_COUNT);
  put_u32(buffer->bytes + 68, tensor_count);
  put_u64(buffer->bytes + 72,
          (uint64_t)(buffer->length - CJ4ME_MODEL_HEADER_SIZE));
  put_u32(buffer->bytes + 80, CJ4ME_MODEL_HEADER_SIZE);
  put_u32(buffer->bytes + 84, 0u);
}

static test_buffer make_f32_model_bytes(void) {
  static const uint32_t counts[] = {
      CJ4ME_MODEL_TILE_HIDDEN_COUNT * CJ4ME_TILE_FEATURE_COUNT,
      CJ4ME_MODEL_TILE_HIDDEN_COUNT,
      CJ4ME_TILE_EMBEDDING_COUNT * CJ4ME_MODEL_TILE_HIDDEN_COUNT,
      CJ4ME_TILE_EMBEDDING_COUNT,
      CJ4ME_MODEL_HIDDEN1_COUNT * CJ4ME_SCORE_INPUT_COUNT,
      CJ4ME_MODEL_HIDDEN1_COUNT,
      CJ4ME_MODEL_HIDDEN2_COUNT * CJ4ME_MODEL_HIDDEN1_COUNT,
      CJ4ME_MODEL_HIDDEN2_COUNT,
      CJ4ME_MODEL_HIDDEN3_COUNT * CJ4ME_MODEL_HIDDEN2_COUNT,
      CJ4ME_MODEL_HIDDEN3_COUNT,
      CJ4ME_MODEL_OUTPUT_COUNT * CJ4ME_MODEL_HIDDEN3_COUNT,
      CJ4ME_MODEL_OUTPUT_COUNT};
  test_buffer buffer = {(uint8_t *)calloc(800000u, 1u), CJ4ME_MODEL_HEADER_SIZE,
                        800000u};
  assert(buffer.bytes != NULL);
  for (uint32_t i = 0; i < 12u; ++i)
    append_zero_tensor(&buffer, i + 1u, CJ4ME_MODEL_TENSOR_F32, counts[i]);
  finish_header(&buffer, CJ4ME_MODEL_KIND_F32, 12u);
  return buffer;
}

static void append_i8_layer(test_buffer *buffer, uint32_t id,
                            uint32_t input_count, uint32_t output_count) {
  append_zero_tensor(buffer, id, CJ4ME_MODEL_TENSOR_I8,
                     input_count * output_count);
  append_zero_tensor(buffer, id + 1u, CJ4ME_MODEL_TENSOR_I32, output_count);
  append_f32_ones(buffer, id + 2u, output_count);
}

static test_buffer make_i8_model_bytes(void) {
  test_buffer buffer = {(uint8_t *)calloc(250000u, 1u), CJ4ME_MODEL_HEADER_SIZE,
                        250000u};
  assert(buffer.bytes != NULL);
  append_i8_layer(&buffer, 101u, CJ4ME_TILE_FEATURE_COUNT,
                  CJ4ME_MODEL_TILE_HIDDEN_COUNT);
  append_i8_layer(&buffer, 104u, CJ4ME_MODEL_TILE_HIDDEN_COUNT,
                  CJ4ME_TILE_EMBEDDING_COUNT);
  append_i8_layer(&buffer, 107u, CJ4ME_SCORE_INPUT_COUNT,
                  CJ4ME_MODEL_HIDDEN1_COUNT);
  append_i8_layer(&buffer, 110u, CJ4ME_MODEL_HIDDEN1_COUNT,
                  CJ4ME_MODEL_HIDDEN2_COUNT);
  append_i8_layer(&buffer, 113u, CJ4ME_MODEL_HIDDEN2_COUNT,
                  CJ4ME_MODEL_HIDDEN3_COUNT);
  append_i8_layer(&buffer, 116u, CJ4ME_MODEL_HIDDEN3_COUNT,
                  CJ4ME_MODEL_OUTPUT_COUNT);
  append_f32_ones(&buffer, CJ4ME_MODEL_TENSOR_I8_ACTIVATION_SCALES, 7u);
  append_i32_ones(&buffer, CJ4ME_MODEL_TENSOR_I8_REQUANT_MULTIPLIERS,
                  CJ4ME_MODEL_I8_REQUANT_COUNT);
  append_zero_tensor(&buffer, CJ4ME_MODEL_TENSOR_I8_REQUANT_SHIFTS,
                     CJ4ME_MODEL_TENSOR_I32, CJ4ME_MODEL_I8_REQUANT_COUNT);
  finish_header(&buffer, CJ4ME_MODEL_KIND_I8, 21u);
  return buffer;
}

static void test_f32_inference(void) {
  cj4me_model_f32 *model = (cj4me_model_f32 *)calloc(1u, sizeof(*model));
  cj4me_inference_f32_scratch scratch;
  float input[CJ4ME_FEATURE_COUNT] = {0};
  float output = 0.0f;
  assert(model != NULL);
  input[0] = 2.0f;
  model->tile_weights1[0][0] = 1.0f;
  model->tile_weights2[0][0] = 1.0f;
  model->score_weights1[0][0] = 1.0f;
  model->score_weights2[0][0] = 1.0f;
  model->score_weights3[0][0] = 1.0f;
  model->score_weights4[0][0] = 1.0f;
  assert(cj4me_infer_f32(model, input, &scratch, &output));
  assert(output == 2.0f);
  free(model);
}

static void test_i8_inference(void) {
  cj4me_model_i8 *model = (cj4me_model_i8 *)calloc(1u, sizeof(*model));
  cj4me_inference_i8_scratch scratch;
  cj4me_i8_output output;
  float input[CJ4ME_FEATURE_COUNT] = {0};
  assert(model != NULL);
  for (size_t i = 0; i < CJ4ME_MODEL_TILE_HIDDEN_COUNT; ++i)
    model->tile_requant_multipliers1[i] = 1;
  for (size_t i = 0; i < CJ4ME_TILE_EMBEDDING_COUNT; ++i)
    model->tile_requant_multipliers2[i] = 1;
  for (size_t i = 0; i < CJ4ME_MODEL_HIDDEN1_COUNT; ++i)
    model->score_requant_multipliers1[i] = 1;
  for (size_t i = 0; i < CJ4ME_MODEL_HIDDEN2_COUNT; ++i)
    model->score_requant_multipliers2[i] = 1;
  for (size_t i = 0; i < CJ4ME_MODEL_HIDDEN3_COUNT; ++i)
    model->score_requant_multipliers3[i] = 1;
  model->tile_input_scale = 0.5f;
  model->score_input_scale = 1.0f;
  model->output_scale = 1.0f;
  input[0] = 2.0f;
  model->tile_weights1[0][0] = 1;
  model->tile_weights2[0][0] = 1;
  model->score_weights1[0][0] = 1;
  model->score_weights2[0][0] = 1;
  model->score_weights3[0][0] = 1;
  model->score_weights4[0][0] = 1;
  assert(cj4me_quantize_input_i8(model, input, scratch.input));
  assert(scratch.input[0] == 4);
  assert(cj4me_infer_i8(model, scratch.input, &scratch, &output));
  assert(output.quantized == 4);
  assert(output.value == 4.0f);
  free(model);
}

static void test_memory_loading(void) {
  test_buffer f32 = make_f32_model_bytes();
  test_buffer i8 = make_i8_model_bytes();
  cj4me_model_f32 *f32_model = (cj4me_model_f32 *)malloc(sizeof(*f32_model));
  cj4me_model_i8 *i8_model = (cj4me_model_i8 *)malloc(sizeof(*i8_model));
  assert(f32_model != NULL && i8_model != NULL);
  assert(cj4me_model_f32_load_memory(f32_model, f32.bytes, f32.length));
  assert(cj4me_model_i8_load_memory(i8_model, i8.bytes, i8.length));
  assert(i8_model->tile_input_scale == 1.0f);
  assert(i8_model->score_requant_multipliers3[15] == 1);

  f32.bytes[0] = 'X';
  assert(!cj4me_model_f32_load_memory(f32_model, f32.bytes, f32.length));
  f32.bytes[0] = 'C';
  put_u32(f32.bytes + 24, CJ4ME_TILE_COUNT - 1u);
  assert(!cj4me_model_f32_load_memory(f32_model, f32.bytes, f32.length));
  put_u32(f32.bytes + 24, CJ4ME_TILE_COUNT);
  assert(!cj4me_model_f32_load_memory(f32_model, f32.bytes, f32.length - 1u));

  free(f32_model);
  free(i8_model);
  free(f32.bytes);
  free(i8.bytes);
}

static void test_file_loading(void) {
  const char *f32_path = "cj4me-test-model-f32.bin";
  const char *i8_path = "cj4me-test-model-i8.bin";
  test_buffer f32 = make_f32_model_bytes();
  test_buffer i8 = make_i8_model_bytes();
  cj4me_model_f32 *f32_model = (cj4me_model_f32 *)malloc(sizeof(*f32_model));
  cj4me_model_i8 *i8_model = (cj4me_model_i8 *)malloc(sizeof(*i8_model));
  FILE *file = fopen(f32_path, "wb");
  assert(f32_model != NULL && i8_model != NULL && file != NULL);
  assert(fwrite(f32.bytes, 1u, f32.length, file) == f32.length);
  assert(fclose(file) == 0);
  file = fopen(i8_path, "wb");
  assert(file != NULL);
  assert(fwrite(i8.bytes, 1u, i8.length, file) == i8.length);
  assert(fclose(file) == 0);
  assert(cj4me_model_f32_load_file(f32_model, f32_path));
  assert(cj4me_model_i8_load_file(i8_model, i8_path));
  assert(remove(f32_path) == 0);
  assert(remove(i8_path) == 0);
  free(f32_model);
  free(i8_model);
  free(f32.bytes);
  free(i8.bytes);
}

void test_model(void) {
  test_f32_inference();
  test_i8_inference();
  test_memory_loading();
  test_file_loading();
}
