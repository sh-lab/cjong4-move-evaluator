#ifndef CJ4ME_TOOL_MODEL_H
#define CJ4ME_TOOL_MODEL_H

#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "cjong4_move_evaluator/model.h"

static uint32_t cj4me_tool_read_u32_le(const unsigned char bytes[4]) {
  return (uint32_t)bytes[0] | ((uint32_t)bytes[1] << 8) |
         ((uint32_t)bytes[2] << 16) | ((uint32_t)bytes[3] << 24);
}

static int cj4me_tool_detect_model_kind(const char *path,
                                        cj4me_model_kind *kind) {
  unsigned char header[20];
  FILE *file;
  size_t count;
  int close_result;
  if (!path || !kind)
    return 0;
  file = fopen(path, "rb");
  if (!file)
    return 0;
  count = fread(header, 1u, sizeof(header), file);
  close_result = fclose(file);
  if (count != sizeof(header) || close_result != 0 ||
      memcmp(header, "CJ4MEM01", 8u) != 0) {
    return 0;
  }
  *kind = (cj4me_model_kind)cj4me_tool_read_u32_le(header + 16);
  return *kind == CJ4ME_MODEL_KIND_F32 || *kind == CJ4ME_MODEL_KIND_I8;
}

#endif /* CJ4ME_TOOL_MODEL_H */
