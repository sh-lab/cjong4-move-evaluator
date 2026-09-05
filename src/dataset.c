#include "cjong4_move_evaluator/dataset.h"

#include <limits.h>
#include <math.h>
#include <stddef.h>
#include <string.h>

static const uint8_t DATASET_MAGIC[8] = {'C', 'J', '4', 'M',
                                         'E', 'D', 'A', '1'};

_Static_assert(sizeof(float) == 4u, "dataset format requires 32-bit float");
_Static_assert(sizeof(cj4me_dataset_record) == CJ4ME_DATASET_RECORD_SIZE,
               "dataset record structure must not contain padding");

static void write_u16_le(uint8_t out[2], uint16_t value) {
  out[0] = (uint8_t)value;
  out[1] = (uint8_t)(value >> 8);
}

static void write_u32_le(uint8_t out[4], uint32_t value) {
  out[0] = (uint8_t)value;
  out[1] = (uint8_t)(value >> 8);
  out[2] = (uint8_t)(value >> 16);
  out[3] = (uint8_t)(value >> 24);
}

static uint16_t read_u16_le(const uint8_t in[2]) {
  return (uint16_t)((uint16_t)in[0] | ((uint16_t)in[1] << 8));
}

static uint32_t read_u32_le(const uint8_t in[4]) {
  return (uint32_t)in[0] | ((uint32_t)in[1] << 8) | ((uint32_t)in[2] << 16) |
         ((uint32_t)in[3] << 24);
}

static void encode_f32_le(uint8_t out[4], float value) {
  uint32_t bits;
  memcpy(&bits, &value, sizeof(bits));
  write_u32_le(out, bits);
}

static bool decode_f32_le(const uint8_t in[4], float *value) {
  uint32_t bits;
  bits = read_u32_le(in);
  memcpy(value, &bits, sizeof(bits));
  return isfinite(*value);
}

static bool write_header(FILE *file, uint32_t record_count) {
  uint8_t header[CJ4ME_DATASET_HEADER_SIZE] = {0};
  memcpy(header, DATASET_MAGIC, sizeof(DATASET_MAGIC));
  write_u32_le(header + 8, CJ4ME_DATASET_FORMAT_VERSION);
  write_u32_le(header + 12, CJ4ME_FEATURE_SCHEMA_VERSION);
  write_u32_le(header + 16, CJ4ME_FEATURE_COUNT);
  write_u32_le(header + 20, record_count);
  write_u32_le(header + 24, CJ4ME_DATASET_RECORD_SIZE);
  write_u32_le(header + 28, 0u);
  return fwrite(header, 1u, sizeof(header), file) == sizeof(header);
}

static bool record_count_fits_file_api(uint32_t record_count) {
  uint64_t length = (uint64_t)CJ4ME_DATASET_HEADER_SIZE +
                    (uint64_t)record_count * CJ4ME_DATASET_RECORD_SIZE;
  return length <= (uint64_t)LONG_MAX;
}

bool cj4me_dataset_writer_open(cj4me_dataset_writer *writer, const char *path) {
  if (!writer || !path)
    return false;
  memset(writer, 0, sizeof(*writer));
  writer->file = fopen(path, "wb+");
  if (!writer->file)
    return false;
  if (!write_header(writer->file, 0u)) {
    fclose(writer->file);
    memset(writer, 0, sizeof(*writer));
    return false;
  }
  return true;
}

bool cj4me_dataset_writer_append(cj4me_dataset_writer *writer,
                                 const cj4me_dataset_record *record) {
  uint8_t bytes[CJ4ME_DATASET_RECORD_SIZE];
  uint8_t metadata[4];
  size_t offset = 0u;
  if (!writer || !writer->file || writer->failed || !record ||
      writer->record_count == UINT32_MAX ||
      !record_count_fits_file_api(writer->record_count + 1u) ||
      !isfinite(record->target) || record->action_player >= CJ4_PLAYER_COUNT ||
      record->action_type > CJ4_ACTION_PASS) {
    return false;
  }
  for (size_t i = 0; i < CJ4ME_FEATURE_COUNT; ++i) {
    if (!isfinite(record->features[i]))
      return false;
    encode_f32_le(bytes + offset, record->features[i]);
    offset += 4u;
  }
  encode_f32_le(bytes + offset, record->target);
  offset += 4u;
  metadata[0] = record->action_player;
  metadata[1] = record->action_type;
  write_u16_le(metadata + 2, record->flags);
  memcpy(bytes + offset, metadata, sizeof(metadata));
  if (fwrite(bytes, 1u, sizeof(bytes), writer->file) != sizeof(bytes)) {
    writer->failed = true;
    return false;
  }
  ++writer->record_count;
  return true;
}

bool cj4me_dataset_writer_close(cj4me_dataset_writer *writer) {
  bool ok;
  if (!writer || !writer->file)
    return false;

  ok = !writer->failed && fseek(writer->file, 0L, SEEK_SET) == 0 &&
       write_header(writer->file, writer->record_count) &&
       fflush(writer->file) == 0;
  if (fclose(writer->file) != 0)
    ok = false;
  memset(writer, 0, sizeof(*writer));
  return ok;
}

bool cj4me_dataset_reader_open(cj4me_dataset_reader *reader, const char *path) {
  uint8_t header[CJ4ME_DATASET_HEADER_SIZE];
  uint32_t count;
  uint64_t expected;
  long length;

  if (!reader || !path)
    return false;
  memset(reader, 0, sizeof(*reader));
  reader->file = fopen(path, "rb");
  if (!reader->file)
    return false;
  if (fread(header, 1u, sizeof(header), reader->file) != sizeof(header) ||
      memcmp(header, DATASET_MAGIC, sizeof(DATASET_MAGIC)) != 0 ||
      read_u32_le(header + 8) != CJ4ME_DATASET_FORMAT_VERSION ||
      read_u32_le(header + 12) != CJ4ME_FEATURE_SCHEMA_VERSION ||
      read_u32_le(header + 16) != CJ4ME_FEATURE_COUNT ||
      read_u32_le(header + 24) != CJ4ME_DATASET_RECORD_SIZE ||
      read_u32_le(header + 28) != 0u) {
    fclose(reader->file);
    memset(reader, 0, sizeof(*reader));
    return false;
  }
  count = read_u32_le(header + 20);
  expected = (uint64_t)CJ4ME_DATASET_HEADER_SIZE +
             (uint64_t)count * CJ4ME_DATASET_RECORD_SIZE;
  if (expected > (uint64_t)LONG_MAX || fseek(reader->file, 0L, SEEK_END) != 0) {
    fclose(reader->file);
    memset(reader, 0, sizeof(*reader));
    return false;
  }
  length = ftell(reader->file);
  if (length < 0 || (uint64_t)length != expected ||
      fseek(reader->file, (long)CJ4ME_DATASET_HEADER_SIZE, SEEK_SET) != 0) {
    fclose(reader->file);
    memset(reader, 0, sizeof(*reader));
    return false;
  }
  reader->record_count = count;
  return true;
}

bool cj4me_dataset_reader_next(cj4me_dataset_reader *reader,
                               cj4me_dataset_record *record) {
  uint8_t bytes[CJ4ME_DATASET_RECORD_SIZE];
  uint8_t metadata[4];
  size_t offset = 0u;
  if (!reader || !reader->file || !record ||
      reader->next_record >= reader->record_count) {
    return false;
  }
  if (fread(bytes, 1u, sizeof(bytes), reader->file) != sizeof(bytes)) {
    reader->failed = true;
    return false;
  }
  for (size_t i = 0; i < CJ4ME_FEATURE_COUNT; ++i) {
    if (!decode_f32_le(bytes + offset, &record->features[i])) {
      reader->failed = true;
      return false;
    }
    offset += 4u;
  }
  if (!decode_f32_le(bytes + offset, &record->target)) {
    reader->failed = true;
    return false;
  }
  offset += 4u;
  memcpy(metadata, bytes + offset, sizeof(metadata));
  record->action_player = metadata[0];
  record->action_type = metadata[1];
  record->flags = read_u16_le(metadata + 2);
  if (record->action_player >= CJ4_PLAYER_COUNT ||
      record->action_type > CJ4_ACTION_PASS) {
    reader->failed = true;
    return false;
  }
  ++reader->next_record;
  return true;
}

void cj4me_dataset_reader_close(cj4me_dataset_reader *reader) {
  if (!reader)
    return;
  if (reader->file)
    fclose(reader->file);
  memset(reader, 0, sizeof(*reader));
}
