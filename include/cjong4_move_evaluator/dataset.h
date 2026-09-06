#ifndef CJ4ME_DATASET_H
#define CJ4ME_DATASET_H

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>

#include "cjong4_move_evaluator/feature.h"

#ifdef __cplusplus
extern "C" {
#endif

#define CJ4ME_DATASET_FORMAT_VERSION 2u
#define CJ4ME_DATASET_HEADER_SIZE 32u
#define CJ4ME_DATASET_RECORD_SIZE ((uint32_t)(CJ4ME_FEATURE_COUNT * 4u + 32u))

enum {
  CJ4ME_CALL_AVAILABLE_CHI = 1u << 0,
  CJ4ME_CALL_AVAILABLE_PON = 1u << 1,
  CJ4ME_CALL_AVAILABLE_MINKAN = 1u << 2,
  CJ4ME_CALL_AVAILABLE_MASK = CJ4ME_CALL_AVAILABLE_CHI |
                              CJ4ME_CALL_AVAILABLE_PON |
                              CJ4ME_CALL_AVAILABLE_MINKAN
};

typedef enum {
  CJ4ME_TENPAI_UNKNOWN = 0,
  CJ4ME_TENPAI_NO = 1,
  CJ4ME_TENPAI_YES = 2
} cj4me_tenpai_status;

enum {
  CJ4ME_FACT_WAS_MENZEN = 1u << 0,
  CJ4ME_FACT_OPENED_HAND = 1u << 1,
  CJ4ME_FACT_CALL_AVAILABLE = 1u << 2,
  CJ4ME_FACT_CHOSE_CALL = 1u << 3,
  CJ4ME_FACT_RIICHI_AVAILABLE = 1u << 4,
  CJ4ME_FACT_CHOSE_RIICHI = 1u << 5,
  CJ4ME_FACT_PLAYER_WON = 1u << 6,
  CJ4ME_FACT_PLAYER_DEALT_IN = 1u << 7,
  CJ4ME_FACT_DEAL_IN_ACTION = 1u << 8,
  CJ4ME_FACT_FLAGS_MASK = (1u << 9) - 1u
};

typedef struct {
  float features[CJ4ME_FEATURE_COUNT];
  float target;
  uint8_t action_player;
  uint8_t action_type;
  uint16_t flags;
  int32_t score_delta;
  int32_t settlement_delta;
  int32_t deal_in_points;
  int32_t win_points;
  uint8_t decision_discard_count;
  uint8_t round_discard_count;
  uint8_t discards_until_end;
  uint8_t round_end_type;
  uint8_t available_call_mask;
  uint8_t tenpai_status;
  uint16_t fact_flags;
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
