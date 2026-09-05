#include <errno.h>
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "cjong4_move_evaluator/dataset.h"
#include "cjong4_move_evaluator/model.h"
#include "tool_model.h"

static int parse_limit(const char *text, uint32_t *limit) {
  char *end;
  unsigned long value;
  errno = 0;
  value = strtoul(text, &end, 10);
  if (errno != 0 || end == text || *end != '\0' || value > UINT32_MAX) {
    return 0;
  }
  *limit = (uint32_t)value;
  return 1;
}

int main(int argc, char **argv) {
  const char *model_path = NULL;
  const char *dataset_path = NULL;
  uint32_t limit = UINT32_MAX;
  cj4me_model_kind kind;
  cj4me_dataset_reader reader;
  cj4me_dataset_record *record;
  void *model;
  uint32_t index = 0u;
  int result = 1;

  for (int i = 1; i < argc; ++i) {
    if (i + 1 >= argc)
      goto usage;
    if (strcmp(argv[i], "--model") == 0)
      model_path = argv[++i];
    else if (strcmp(argv[i], "--dataset") == 0)
      dataset_path = argv[++i];
    else if (strcmp(argv[i], "--limit") == 0) {
      if (!parse_limit(argv[++i], &limit))
        goto usage;
    } else
      goto usage;
  }
  if (!model_path || !dataset_path)
    goto usage;

  if (!cj4me_tool_detect_model_kind(model_path, &kind)) {
    fprintf(stderr, "cj4me_evaluate: invalid model header\n");
    return 1;
  }
  model = calloc(1u, kind == CJ4ME_MODEL_KIND_F32 ? sizeof(cj4me_model_f32)
                                                  : sizeof(cj4me_model_i8));
  record = (cj4me_dataset_record *)malloc(sizeof(*record));
  if (!model || !record) {
    fprintf(stderr, "cj4me_evaluate: allocation failed\n");
    goto cleanup;
  }
  if ((kind == CJ4ME_MODEL_KIND_F32 &&
       !cj4me_model_f32_load_file((cj4me_model_f32 *)model, model_path)) ||
      (kind == CJ4ME_MODEL_KIND_I8 &&
       !cj4me_model_i8_load_file((cj4me_model_i8 *)model, model_path))) {
    fprintf(stderr, "cj4me_evaluate: model load failed\n");
    goto cleanup;
  }
  if (!cj4me_dataset_reader_open(&reader, dataset_path)) {
    fprintf(stderr, "cj4me_evaluate: dataset load failed\n");
    goto cleanup;
  }

  while (index < limit && cj4me_dataset_reader_next(&reader, record)) {
    if (kind == CJ4ME_MODEL_KIND_F32) {
      cj4me_inference_f32_scratch scratch;
      float score;
      if (!cj4me_infer_f32((const cj4me_model_f32 *)model, record->features,
                           &scratch, &score)) {
        fprintf(stderr, "cj4me_evaluate: inference failed\n");
        cj4me_dataset_reader_close(&reader);
        goto cleanup;
      }
      printf("%" PRIu32 "\t%.9g\t%.9g\n", index, (double)record->target,
             (double)score);
    } else {
      cj4me_inference_i8_scratch scratch;
      cj4me_i8_output score;
      if (!cj4me_quantize_input_i8((const cj4me_model_i8 *)model,
                                   record->features, scratch.input) ||
          !cj4me_infer_i8((const cj4me_model_i8 *)model, scratch.input,
                          &scratch, &score)) {
        fprintf(stderr, "cj4me_evaluate: inference failed\n");
        cj4me_dataset_reader_close(&reader);
        goto cleanup;
      }
      printf("%" PRIu32 "\t%.9g\t%" PRId32 "\t%.9g\n", index,
             (double)record->target, score.quantized, (double)score.value);
    }
    ++index;
  }
  if (reader.failed || (index < reader.record_count && index < limit)) {
    fprintf(stderr, "cj4me_evaluate: record read failed\n");
    cj4me_dataset_reader_close(&reader);
    goto cleanup;
  }
  cj4me_dataset_reader_close(&reader);
  result = 0;
  goto cleanup;

usage:
  fprintf(stderr, "Usage: %s --model PATH --dataset PATH [--limit N]\n",
          argv[0]);
  return 2;

cleanup:
  free(record);
  free(model);
  return result;
}
