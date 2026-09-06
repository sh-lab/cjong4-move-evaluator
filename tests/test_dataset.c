#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

#include "cjong4_move_evaluator/dataset.h"

void test_dataset(void) {
  const char *path = "cj4me-test-dataset.bin";
  const char *bad_path = "cj4me-test-dataset-bad.bin";
  cj4me_dataset_writer writer;
  cj4me_dataset_reader reader;
  cj4me_dataset_record input;
  cj4me_dataset_record output;
  FILE *bad;

  memset(&input, 0, sizeof(input));
  input.features[0] = -1.0f;
  input.features[CJ4ME_FEATURE_COUNT - 1] = 1.0f;
  input.target = 0.5f;
  input.action_player = 2;
  input.action_type = 4;
  input.flags = 7;
  input.score_delta = -12000;
  input.settlement_delta = -12000;
  input.deal_in_points = 12000;
  input.decision_discard_count = 10;
  input.round_discard_count = 18;
  input.discards_until_end = 8;
  input.round_end_type = CJ4_ROUND_END_RON;
  input.available_call_mask = CJ4ME_CALL_AVAILABLE_PON;
  input.tenpai_status = CJ4ME_TENPAI_UNKNOWN;
  input.fact_flags = CJ4ME_FACT_PLAYER_DEALT_IN | CJ4ME_FACT_DEAL_IN_ACTION;

  assert(cj4me_dataset_writer_open(&writer, path));
  input.features[10] = NAN;
  assert(!cj4me_dataset_writer_append(&writer, &input));
  assert(!writer.failed);
  input.features[10] = 0.0f;
  assert(cj4me_dataset_writer_append(&writer, &input));
  assert(cj4me_dataset_writer_close(&writer));
  assert(cj4me_dataset_reader_open(&reader, path));
  assert(reader.record_count == 1u);
  assert(cj4me_dataset_reader_next(&reader, &output));
  assert(!cj4me_dataset_reader_next(&reader, &output));
  assert(memcmp(&input, &output, sizeof(input)) == 0);
  cj4me_dataset_reader_close(&reader);
  bad = fopen(path, "ab");
  assert(bad != NULL);
  assert(fwrite("x", 1u, 1u, bad) == 1u);
  assert(fclose(bad) == 0);
  assert(!cj4me_dataset_reader_open(&reader, path));
  assert(remove(path) == 0);

  bad = fopen(bad_path, "wb");
  assert(bad != NULL);
  assert(fwrite("CJ4MEDA2", 1u, 8u, bad) == 8u);
  assert(fclose(bad) == 0);
  assert(!cj4me_dataset_reader_open(&reader, bad_path));
  assert(remove(bad_path) == 0);
}
