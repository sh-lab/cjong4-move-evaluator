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

static uint32_t get_u32(const uint8_t *input) {
  return (uint32_t)input[0] | ((uint32_t)input[1] << 8) |
         ((uint32_t)input[2] << 16) | ((uint32_t)input[3] << 24);
}

static void append_bytes(test_buffer *buffer, const void *bytes, size_t count) {
  assert(buffer->length + count <= buffer->capacity);
  memcpy(buffer->bytes + buffer->length, bytes, count);
  buffer->length += count;
}

static void append_tensor_header(test_buffer *buffer, uint32_t id,
                                 uint32_t type, uint32_t count,
                                 uint32_t byte_length) {
  uint8_t header[16];
  put_u32(header, id);
  put_u32(header + 4, type);
  put_u32(header + 8, count);
  put_u32(header + 12, byte_length);
  append_bytes(buffer, header, sizeof(header));
}

static void append_zero_tensor(test_buffer *buffer, uint32_t id, uint32_t type,
                               uint32_t count) {
  const uint32_t element_size = type == CJ4ME_MODEL_TENSOR_I8 ? 1u : 4u;
  const size_t byte_length = (size_t)count * element_size;
  assert(byte_length <= UINT32_MAX);
  append_tensor_header(buffer, id, type, count, (uint32_t)byte_length);
  memset(buffer->bytes + buffer->length, 0, byte_length);
  buffer->length += byte_length;
}

static void append_f32_ones(test_buffer *buffer, uint32_t id, uint32_t count) {
  append_tensor_header(buffer, id, CJ4ME_MODEL_TENSOR_F32, count, count * 4u);
  for (uint32_t i = 0; i < count; ++i) {
    uint8_t encoded[4];
    put_f32(encoded, 1.0f);
    append_bytes(buffer, encoded, sizeof(encoded));
  }
}

static uint8_t *find_tensor_data(test_buffer *buffer, uint32_t tensor_id) {
  size_t offset = CJ4ME_MODEL_HEADER_SIZE;

  while (offset + 16u <= buffer->length) {
    uint8_t *header = buffer->bytes + offset;
    const uint32_t byte_length = get_u32(header + 12);
    offset += 16u;
    assert(byte_length <= buffer->length - offset);
    if (get_u32(header) == tensor_id)
      return buffer->bytes + offset;
    offset += byte_length;
  }
  return NULL;
}

static void append_i32_ones(test_buffer *buffer, uint32_t id, uint32_t count) {
  append_tensor_header(buffer, id, CJ4ME_MODEL_TENSOR_I32, count, count * 4u);
  for (uint32_t i = 0; i < count; ++i) {
    uint8_t encoded[4];
    put_u32(encoded, 1u);
    append_bytes(buffer, encoded, sizeof(encoded));
  }
}

static void finish_header(test_buffer *buffer, cj4me_model_kind kind,
                          uint32_t tensor_count) {
  memcpy(buffer->bytes, "CJ4MEM01", 8u);
  put_u32(buffer->bytes + 8, CJ4ME_MODEL_FORMAT_VERSION);
  put_u32(buffer->bytes + 12, CJ4ME_FEATURE_SCHEMA_VERSION);
  put_u32(buffer->bytes + 16, (uint32_t)kind);
  put_u32(buffer->bytes + 20, CJ4ME_MODEL_LAYER_COUNT);
  put_u32(buffer->bytes + 24, CJ4ME_FEATURE_COUNT);
  put_u32(buffer->bytes + 28, CJ4ME_MODEL_HIDDEN1_COUNT);
  put_u32(buffer->bytes + 32, CJ4ME_MODEL_HIDDEN2_COUNT);
  put_u32(buffer->bytes + 36, CJ4ME_MODEL_HIDDEN3_COUNT);
  put_u32(buffer->bytes + 40, CJ4ME_MODEL_OUTPUT_COUNT);
  put_u32(buffer->bytes + 44, tensor_count);
  put_u64(buffer->bytes + 48,
          (uint64_t)(buffer->length - CJ4ME_MODEL_HEADER_SIZE));
  put_u32(buffer->bytes + 56, CJ4ME_MODEL_HEADER_SIZE);
  put_u32(buffer->bytes + 60, 0u);
}

static test_buffer make_f32_model_bytes(void) {
  test_buffer buffer;
  buffer.capacity = 800000u;
  buffer.bytes = (uint8_t *)calloc(buffer.capacity, 1u);
  assert(buffer.bytes != NULL);
  buffer.length = CJ4ME_MODEL_HEADER_SIZE;

  append_zero_tensor(&buffer, CJ4ME_MODEL_TENSOR_F32_W1, CJ4ME_MODEL_TENSOR_F32,
                     CJ4ME_MODEL_HIDDEN1_COUNT * CJ4ME_FEATURE_COUNT);
  append_zero_tensor(&buffer, CJ4ME_MODEL_TENSOR_F32_B1, CJ4ME_MODEL_TENSOR_F32,
                     CJ4ME_MODEL_HIDDEN1_COUNT);
  append_zero_tensor(&buffer, CJ4ME_MODEL_TENSOR_F32_W2, CJ4ME_MODEL_TENSOR_F32,
                     CJ4ME_MODEL_HIDDEN2_COUNT * CJ4ME_MODEL_HIDDEN1_COUNT);
  append_zero_tensor(&buffer, CJ4ME_MODEL_TENSOR_F32_B2, CJ4ME_MODEL_TENSOR_F32,
                     CJ4ME_MODEL_HIDDEN2_COUNT);
  append_zero_tensor(&buffer, CJ4ME_MODEL_TENSOR_F32_W3, CJ4ME_MODEL_TENSOR_F32,
                     CJ4ME_MODEL_HIDDEN3_COUNT * CJ4ME_MODEL_HIDDEN2_COUNT);
  append_zero_tensor(&buffer, CJ4ME_MODEL_TENSOR_F32_B3, CJ4ME_MODEL_TENSOR_F32,
                     CJ4ME_MODEL_HIDDEN3_COUNT);
  append_zero_tensor(&buffer, CJ4ME_MODEL_TENSOR_F32_W4, CJ4ME_MODEL_TENSOR_F32,
                     CJ4ME_MODEL_OUTPUT_COUNT * CJ4ME_MODEL_HIDDEN3_COUNT);
  append_zero_tensor(&buffer, CJ4ME_MODEL_TENSOR_F32_B4, CJ4ME_MODEL_TENSOR_F32,
                     CJ4ME_MODEL_OUTPUT_COUNT);
  finish_header(&buffer, CJ4ME_MODEL_KIND_F32, 8u);
  return buffer;
}

static test_buffer make_i8_model_bytes(void) {
  test_buffer buffer;
  buffer.capacity = 200000u;
  buffer.bytes = (uint8_t *)calloc(buffer.capacity, 1u);
  assert(buffer.bytes != NULL);
  buffer.length = CJ4ME_MODEL_HEADER_SIZE;

  append_zero_tensor(&buffer, CJ4ME_MODEL_TENSOR_I8_W1, CJ4ME_MODEL_TENSOR_I8,
                     CJ4ME_MODEL_HIDDEN1_COUNT * CJ4ME_FEATURE_COUNT);
  append_zero_tensor(&buffer, CJ4ME_MODEL_TENSOR_I8_B1, CJ4ME_MODEL_TENSOR_I32,
                     CJ4ME_MODEL_HIDDEN1_COUNT);
  append_f32_ones(&buffer, CJ4ME_MODEL_TENSOR_I8_WS1,
                  CJ4ME_MODEL_HIDDEN1_COUNT);
  append_zero_tensor(&buffer, CJ4ME_MODEL_TENSOR_I8_W2, CJ4ME_MODEL_TENSOR_I8,
                     CJ4ME_MODEL_HIDDEN2_COUNT * CJ4ME_MODEL_HIDDEN1_COUNT);
  append_zero_tensor(&buffer, CJ4ME_MODEL_TENSOR_I8_B2, CJ4ME_MODEL_TENSOR_I32,
                     CJ4ME_MODEL_HIDDEN2_COUNT);
  append_f32_ones(&buffer, CJ4ME_MODEL_TENSOR_I8_WS2,
                  CJ4ME_MODEL_HIDDEN2_COUNT);
  append_zero_tensor(&buffer, CJ4ME_MODEL_TENSOR_I8_W3, CJ4ME_MODEL_TENSOR_I8,
                     CJ4ME_MODEL_HIDDEN3_COUNT * CJ4ME_MODEL_HIDDEN2_COUNT);
  append_zero_tensor(&buffer, CJ4ME_MODEL_TENSOR_I8_B3, CJ4ME_MODEL_TENSOR_I32,
                     CJ4ME_MODEL_HIDDEN3_COUNT);
  append_f32_ones(&buffer, CJ4ME_MODEL_TENSOR_I8_WS3,
                  CJ4ME_MODEL_HIDDEN3_COUNT);
  append_zero_tensor(&buffer, CJ4ME_MODEL_TENSOR_I8_W4, CJ4ME_MODEL_TENSOR_I8,
                     CJ4ME_MODEL_OUTPUT_COUNT * CJ4ME_MODEL_HIDDEN3_COUNT);
  append_zero_tensor(&buffer, CJ4ME_MODEL_TENSOR_I8_B4, CJ4ME_MODEL_TENSOR_I32,
                     CJ4ME_MODEL_OUTPUT_COUNT);
  append_f32_ones(&buffer, CJ4ME_MODEL_TENSOR_I8_WS4, CJ4ME_MODEL_OUTPUT_COUNT);
  append_f32_ones(&buffer, CJ4ME_MODEL_TENSOR_I8_ACTIVATION_SCALES, 5u);
  append_i32_ones(&buffer, CJ4ME_MODEL_TENSOR_I8_REQUANT_MULTIPLIERS,
                  CJ4ME_MODEL_I8_REQUANT_COUNT);
  append_zero_tensor(&buffer, CJ4ME_MODEL_TENSOR_I8_REQUANT_SHIFTS,
                     CJ4ME_MODEL_TENSOR_I32, CJ4ME_MODEL_I8_REQUANT_COUNT);
  finish_header(&buffer, CJ4ME_MODEL_KIND_I8, 15u);
  return buffer;
}

static void test_f32_inference(void) {
  cj4me_model_f32 *model = (cj4me_model_f32 *)calloc(1u, sizeof(*model));
  cj4me_inference_f32_scratch scratch;
  float input[CJ4ME_FEATURE_COUNT] = {0};
  float output = 0.0f;

  assert(model != NULL);
  input[0] = 2.0f;
  input[1] = -3.0f;
  model->weights1[0][0] = 1.0f;
  model->weights1[0][1] = 1.0f;
  model->biases1[0] = 2.0f;
  model->weights1[1][0] = -1.0f;
  model->biases1[1] = 1.0f;
  model->weights2[0][0] = 3.0f;
  model->biases2[0] = -1.0f;
  model->weights3[0][0] = 4.0f;
  model->biases3[0] = 1.0f;
  model->weights4[0][0] = 2.0f;
  model->biases4[0] = -3.0f;

  assert(cj4me_infer_f32(model, input, &scratch, &output));
  assert(output > 14.999999f && output < 15.000001f);
  free(model);
}

static void test_i8_inference(void) {
  cj4me_model_i8 *model = (cj4me_model_i8 *)calloc(1u, sizeof(*model));
  cj4me_inference_i8_scratch scratch;
  cj4me_i8_output output;
  float float_input[CJ4ME_FEATURE_COUNT] = {0};

  assert(model != NULL);
  for (size_t i = 0; i < CJ4ME_MODEL_HIDDEN1_COUNT; ++i)
    model->requant_multipliers1[i] = 1;
  for (size_t i = 0; i < CJ4ME_MODEL_HIDDEN2_COUNT; ++i)
    model->requant_multipliers2[i] = 1;
  for (size_t i = 0; i < CJ4ME_MODEL_HIDDEN3_COUNT; ++i)
    model->requant_multipliers3[i] = 1;
  model->input_scale = 0.5f;
  model->output_scale = 0.25f;

  float_input[0] = 200.0f;
  float_input[1] = -200.0f;
  assert(cj4me_quantize_input_i8(model, float_input, scratch.input));
  assert(scratch.input[0] == INT8_MAX);
  assert(scratch.input[1] == INT8_MIN);

  memset(scratch.input, 0, sizeof(scratch.input));
  scratch.input[0] = 100;
  model->weights1[0][0] = 2;
  model->weights2[0][0] = 2;
  model->weights3[0][0] = 2;
  model->weights4[0][0] = 2;
  assert(cj4me_infer_i8(model, scratch.input, &scratch, &output));
  assert(scratch.hidden1[0] == INT8_MAX);
  assert(scratch.hidden2[0] == INT8_MAX);
  assert(scratch.hidden3[0] == INT8_MAX);
  assert(output.quantized == 254);
  assert(output.scale == 0.25f);
  assert(output.value == 63.5f);

  memset(model->weights1, 0, sizeof(model->weights1));
  memset(model->weights2, 0, sizeof(model->weights2));
  memset(model->weights3, 0, sizeof(model->weights3));
  memset(model->weights4, 0, sizeof(model->weights4));
  memset(scratch.input, 0, sizeof(scratch.input));
  scratch.input[0] = 3;
  model->weights1[0][0] = 1;
  model->requant_shifts1[0] = 1;
  model->weights2[0][0] = 1;
  model->weights3[0][0] = 1;
  model->weights4[0][0] = 1;
  assert(cj4me_infer_i8(model, scratch.input, &scratch, &output));
  assert(output.quantized == 2);
  free(model);
}

static void test_memory_loading(void) {
  test_buffer f32 = make_f32_model_bytes();
  test_buffer i8 = make_i8_model_bytes();
  cj4me_model_f32 *f32_model = (cj4me_model_f32 *)malloc(sizeof(*f32_model));
  cj4me_model_i8 *i8_model = (cj4me_model_i8 *)malloc(sizeof(*i8_model));
  uint8_t saved[8];

  assert(f32_model != NULL);
  assert(i8_model != NULL);
  assert(cj4me_model_f32_load_memory(f32_model, f32.bytes, f32.length));
  assert(cj4me_model_i8_load_memory(i8_model, i8.bytes, i8.length));
  assert(i8_model->input_scale == 1.0f);
  assert(i8_model->requant_multipliers3[15] == 1);

  {
    uint8_t *activation_scales =
        find_tensor_data(&i8, CJ4ME_MODEL_TENSOR_I8_ACTIVATION_SCALES);
    assert(activation_scales != NULL);
    put_f32(activation_scales + 4u * 4u, 1.0000005f);
    assert(cj4me_model_i8_load_memory(i8_model, i8.bytes, i8.length));
    put_f32(activation_scales + 4u * 4u, 1.00001f);
    assert(!cj4me_model_i8_load_memory(i8_model, i8.bytes, i8.length));
    assert(i8_model->output_scale == 0.0f);
    put_f32(activation_scales + 4u * 4u, 1.0f);
    assert(cj4me_model_i8_load_memory(i8_model, i8.bytes, i8.length));
  }

  memcpy(saved, f32.bytes, sizeof(saved));
  f32.bytes[0] = 'X';
  assert(!cj4me_model_f32_load_memory(f32_model, f32.bytes, f32.length));
  memcpy(f32.bytes, saved, sizeof(saved));

  put_u32(f32.bytes + 8, 2u);
  assert(!cj4me_model_f32_load_memory(f32_model, f32.bytes, f32.length));
  put_u32(f32.bytes + 8, CJ4ME_MODEL_FORMAT_VERSION);

  put_u32(f32.bytes + 12, CJ4ME_FEATURE_SCHEMA_VERSION + 1u);
  assert(!cj4me_model_f32_load_memory(f32_model, f32.bytes, f32.length));
  put_u32(f32.bytes + 12, CJ4ME_FEATURE_SCHEMA_VERSION);

  put_u32(f32.bytes + 16, CJ4ME_MODEL_KIND_I8);
  assert(!cj4me_model_f32_load_memory(f32_model, f32.bytes, f32.length));
  put_u32(f32.bytes + 16, CJ4ME_MODEL_KIND_F32);

  put_u32(f32.bytes + 24, CJ4ME_FEATURE_COUNT - 1u);
  assert(!cj4me_model_f32_load_memory(f32_model, f32.bytes, f32.length));
  put_u32(f32.bytes + 24, CJ4ME_FEATURE_COUNT);

  put_u32(f32.bytes + 44, 7u);
  assert(!cj4me_model_f32_load_memory(f32_model, f32.bytes, f32.length));
  put_u32(f32.bytes + 44, 8u);

  put_u32(f32.bytes + CJ4ME_MODEL_HEADER_SIZE + 4u, CJ4ME_MODEL_TENSOR_I32);
  assert(!cj4me_model_f32_load_memory(f32_model, f32.bytes, f32.length));
  put_u32(f32.bytes + CJ4ME_MODEL_HEADER_SIZE + 4u, CJ4ME_MODEL_TENSOR_F32);

  put_u32(f32.bytes + CJ4ME_MODEL_HEADER_SIZE + 8u, 1u);
  assert(!cj4me_model_f32_load_memory(f32_model, f32.bytes, f32.length));
  put_u32(f32.bytes + CJ4ME_MODEL_HEADER_SIZE + 8u,
          CJ4ME_MODEL_HIDDEN1_COUNT * CJ4ME_FEATURE_COUNT);

  assert(!cj4me_model_f32_load_memory(f32_model, f32.bytes, f32.length - 1u));
  f32.bytes[f32.length] = 0u;
  assert(!cj4me_model_f32_load_memory(f32_model, f32.bytes, f32.length + 1u));

  free(f32_model);
  free(i8_model);
  free(f32.bytes);
  free(i8.bytes);
}

static void write_test_file(const char *path, const uint8_t *bytes,
                            size_t length) {
  FILE *file = fopen(path, "wb");
  assert(file != NULL);
  assert(fwrite(bytes, 1u, length, file) == length);
  assert(fclose(file) == 0);
}

static void test_file_loading(void) {
  const char *f32_path = "cj4me-test-model-f32.bin";
  const char *i8_path = "cj4me-test-model-i8.bin";
  const char *bad_path = "cj4me-test-model-bad.bin";
  test_buffer f32 = make_f32_model_bytes();
  test_buffer i8 = make_i8_model_bytes();
  cj4me_model_f32 *f32_model = (cj4me_model_f32 *)malloc(sizeof(*f32_model));
  cj4me_model_i8 *i8_model = (cj4me_model_i8 *)malloc(sizeof(*i8_model));

  assert(f32_model != NULL);
  assert(i8_model != NULL);
  write_test_file(f32_path, f32.bytes, f32.length);
  write_test_file(i8_path, i8.bytes, i8.length);
  assert(cj4me_model_f32_load_file(f32_model, f32_path));
  assert(cj4me_model_i8_load_file(i8_model, i8_path));

  f32_model->biases4[0] = 9.0f;
  assert(
      !cj4me_model_f32_load_file(f32_model, "cj4me-model-does-not-exist.bin"));
  assert(f32_model->biases4[0] == 0.0f);

  write_test_file(bad_path, f32.bytes, f32.length - 1u);
  f32_model->biases4[0] = 9.0f;
  assert(!cj4me_model_f32_load_file(f32_model, bad_path));
  assert(f32_model->biases4[0] == 0.0f);

  f32.bytes[f32.length] = 0u;
  write_test_file(bad_path, f32.bytes, f32.length + 1u);
  assert(!cj4me_model_f32_load_file(f32_model, bad_path));

  f32.bytes[0] = 'X';
  write_test_file(bad_path, f32.bytes, f32.length);
  assert(!cj4me_model_f32_load_file(f32_model, bad_path));

  assert(remove(f32_path) == 0);
  assert(remove(i8_path) == 0);
  assert(remove(bad_path) == 0);
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
