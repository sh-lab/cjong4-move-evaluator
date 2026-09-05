#ifndef CJ4ME_DATASET_H
#define CJ4ME_DATASET_H

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>

#include "cjong4_move_evaluator/feature.h"

#ifdef __cplusplus
extern "C" {
#endif

#define CJ4ME_DATASET_FORMAT_VERSION 1u
#define CJ4ME_DATASET_HEADER_SIZE 32u
#define CJ4ME_DATASET_RECORD_SIZE ((uint32_t)(CJ4ME_FEATURE_COUNT * 4u + 8u))

typedef struct {
  float features[CJ4ME_FEATURE_COUNT];
  float target;
  uint8_t action_player;
  uint8_t action_type;
  uint16_t flags;
} cj4me_dataset_record;

typedef struct {
  FILE *file;
  uint32_t record_count;
  bool failed;
} cj4me_dataset_writer;

typedef struct {
  FILE *file;
  uint32_t record_count;
  uint32_t next_record;
  bool failed;
} cj4me_dataset_reader;

/** Opens a new dataset and writes a placeholder header. */
bool cj4me_dataset_writer_open(cj4me_dataset_writer *writer, const char *path);

/** Appends one finite fixed-size record. */
bool cj4me_dataset_writer_append(cj4me_dataset_writer *writer,
                                 const cj4me_dataset_record *record);

/** Finalizes the record count and closes the file. */
bool cj4me_dataset_writer_close(cj4me_dataset_writer *writer);

/** Opens and fully validates a dataset header and file length. */
bool cj4me_dataset_reader_open(cj4me_dataset_reader *reader, const char *path);

/**
 * Reads the next record.
 *
 * Returns false at clean end-of-file or on error. Inspect `failed` to
 * distinguish the two cases.
 */
bool cj4me_dataset_reader_next(cj4me_dataset_reader *reader,
                               cj4me_dataset_record *record);

/** Closes a dataset reader. */
void cj4me_dataset_reader_close(cj4me_dataset_reader *reader);

#ifdef __cplusplus
}
#endif

#endif /* CJ4ME_DATASET_H */
