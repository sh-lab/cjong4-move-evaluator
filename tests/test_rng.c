#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

#include "../src/rng.h"
#include "../src/selfplay.h"
#include "cjong4/core/rules.h"
#include "cjong4/core/state_init.h"
#include "cjong4/manager/manager.h"
#include "cjong4_move_evaluator/dataset.h"

static int files_equal(const char *first_path, const char *second_path) {
  FILE *first = fopen(first_path, "rb");
  FILE *second = fopen(second_path, "rb");
  int equal = first != NULL && second != NULL;

  while (equal) {
    unsigned char a[4096];
    unsigned char b[4096];
    size_t a_count = fread(a, 1u, sizeof(a), first);
    size_t b_count = fread(b, 1u, sizeof(b), second);
    if (a_count != b_count || memcmp(a, b, a_count) != 0)
      equal = 0;
    if (a_count < sizeof(a))
      break;
  }
  if (first)
    fclose(first);
  if (second)
    fclose(second);
  return equal;
}

void test_rng(void) {
  cj4me_rng first;
  cj4me_rng second;
  cj4_tile_id wall_a[CJ4_TILE_ID_COUNT];
  cj4_tile_id wall_b[CJ4_TILE_ID_COUNT];

  cj4me_rng_seed(&first, 123u);
  cj4me_rng_seed(&second, 123u);
  cj4me_rng_shuffle_wall(&first, wall_a);
  cj4me_rng_shuffle_wall(&second, wall_b);
  assert(memcmp(wall_a, wall_b, sizeof(wall_a)) == 0);
  assert(cj4_wall_is_valid(wall_a));

  {
    const char *path_a = "cj4me-selfplay-a.cj4medata";
    const char *path_b = "cj4me-selfplay-b.cj4medata";
    char error[128];
    cj4me_selfplay_config config = {.games = 1u,
                                    .seed = 99u,
                                    .epsilon = 1.0f,
                                    .reward_scale = 8000.0f,
                                    .max_steps_per_game = 10000u,
                                    .max_records_per_round = 4096u,
                                    .output_path = path_a};
    assert(cj4me_generate_dataset(&config, error, sizeof(error)));
    config.output_path = path_b;
    assert(cj4me_generate_dataset(&config, error, sizeof(error)));
    assert(files_equal(path_a, path_b));
    {
      cj4me_dataset_reader reader;
      cj4me_dataset_record record;
      uint32_t records_read = 0u;
      assert(cj4me_dataset_reader_open(&reader, path_a));
      assert(reader.record_count > 0u);
      while (cj4me_dataset_reader_next(&reader, &record)) {
        const bool call_available = record.available_call_mask != 0u;
        const bool chose_call = record.action_type == CJ4_ACTION_CHI ||
                                record.action_type == CJ4_ACTION_PON ||
                                record.action_type == CJ4_ACTION_MINKAN;
        assert(record.round_end_type != CJ4_ROUND_END_NONE);
        assert(fabsf(record.target - (float)record.score_delta / 8000.0f) <
               0.000001f);
        assert(((record.fact_flags & CJ4ME_FACT_CALL_AVAILABLE) != 0u) ==
               call_available);
        assert(((record.fact_flags & CJ4ME_FACT_CHOSE_CALL) != 0u) ==
               chose_call);
        if ((record.fact_flags & CJ4ME_FACT_OPENED_HAND) != 0u) {
          assert((record.fact_flags & CJ4ME_FACT_WAS_MENZEN) != 0u);
          assert(chose_call);
        }
        if ((record.fact_flags & CJ4ME_FACT_CHOSE_RIICHI) != 0u)
          assert(record.action_type == CJ4_ACTION_RIICHI);
        if ((record.fact_flags & CJ4ME_FACT_DEAL_IN_ACTION) != 0u) {
          assert((record.fact_flags & CJ4ME_FACT_PLAYER_DEALT_IN) != 0u);
          assert(record.round_end_type == CJ4_ROUND_END_RON);
        }
        if ((record.fact_flags & CJ4ME_FACT_PLAYER_WON_KOKUSHI) != 0u)
          assert((record.fact_flags & CJ4ME_FACT_PLAYER_WON) != 0u);
        if (record.round_end_type == CJ4_ROUND_END_EXHAUSTIVE_DRAW)
          assert(record.tenpai_status != CJ4ME_TENPAI_UNKNOWN);
        else
          assert(record.tenpai_status == CJ4ME_TENPAI_UNKNOWN);
        ++records_read;
      }
      assert(!reader.failed);
      assert(records_read == reader.record_count);
      cj4me_dataset_reader_close(&reader);
    }
    assert(remove(path_a) == 0);
    assert(remove(path_b) == 0);
  }

  {
    const char *path_a = "cj4me-rollout-a.cj4medata";
    const char *path_b = "cj4me-rollout-b.cj4medata";
    const char *path_skipped = "cj4me-rollout-skipped.cj4medata";
    cj4_rules rules = cj4_rules_default();
    cj4_mahjong initial;
    cj4_action actions[CJ4M_MAX_ACTIONS];
    cj4me_dataset_reader reader;
    cj4me_dataset_reader skipped_reader;
    cj4me_dataset_record record;
    cj4me_dataset_record previous;
    cj4me_dataset_record skipped_record;
    char error[128];
    uint8_t action_count;
    uint32_t first_game_records;
    uint32_t records_read = 0u;
    int varied_outcome = 0;
    cj4me_selfplay_config config = {.games = 1u,
                                    .seed = 41001u,
                                    .epsilon = 1.0f,
                                    .reward_scale = 8000.0f,
                                    .max_steps_per_game = 10000u,
                                    .max_records_per_round = 4096u,
                                    .rollouts_per_action = 2u,
                                    .max_rollout_decisions_per_game = 1u,
                                    .output_path = path_a};

    cj4me_rng_seed(&first, config.seed);
    cj4me_rng_shuffle_wall(&first, wall_a);
    initial = cj4_create_initial_state(wall_a, &rules);
    action_count = cj4m_collect_actions(&initial, &rules,
                                        cj4_state_current_player(&initial),
                                        actions, CJ4M_MAX_ACTIONS);
    assert(action_count > 1u);
    for (uint8_t i = 0; i < action_count; ++i)
      assert(actions[i].type != CJ4_ACTION_TSUMO &&
             actions[i].type != CJ4_ACTION_RON);

    assert(cj4me_generate_dataset(&config, error, sizeof(error)));
    config.output_path = path_b;
    assert(cj4me_generate_dataset(&config, error, sizeof(error)));
    assert(files_equal(path_a, path_b));
    assert(cj4me_dataset_reader_open(&reader, path_a));
    assert(reader.record_count == (uint32_t)action_count * 2u);
    first_game_records = reader.record_count;
    while (cj4me_dataset_reader_next(&reader, &record)) {
      assert(record.action_player == cj4_state_current_player(&initial));
      assert(record.decision_discard_count == 0u);
      assert(record.round_end_type != CJ4_ROUND_END_NONE);
      assert(fabsf(record.target - (float)record.score_delta / 8000.0f) <
             0.000001f);
      if ((records_read & 1u) == 0u) {
        previous = record;
      } else {
        assert(memcmp(previous.features, record.features,
                      sizeof(record.features)) == 0);
        assert(previous.action_player == record.action_player);
        assert(previous.action_type == record.action_type);
        varied_outcome |= previous.target != record.target;
      }
      ++records_read;
    }
    assert(!reader.failed);
    assert(records_read == reader.record_count);
    assert(varied_outcome);
    cj4me_dataset_reader_close(&reader);

    config.games = 2u;
    config.output_path = path_b;
    assert(cj4me_generate_dataset(&config, error, sizeof(error)));
    config.games = 1u;
    config.skip_games = 1u;
    config.output_path = path_skipped;
    assert(cj4me_generate_dataset(&config, error, sizeof(error)));

    assert(cj4me_dataset_reader_open(&reader, path_b));
    assert(cj4me_dataset_reader_open(&skipped_reader, path_skipped));
    assert(reader.record_count ==
           first_game_records + skipped_reader.record_count);
    for (uint32_t i = 0; i < first_game_records; ++i)
      assert(cj4me_dataset_reader_next(&reader, &record));
    while (cj4me_dataset_reader_next(&skipped_reader, &skipped_record)) {
      assert(cj4me_dataset_reader_next(&reader, &record));
      assert(memcmp(&record, &skipped_record, sizeof(record)) == 0);
    }
    assert(!skipped_reader.failed);
    assert(!cj4me_dataset_reader_next(&reader, &record));
    assert(!reader.failed);
    cj4me_dataset_reader_close(&reader);
    cj4me_dataset_reader_close(&skipped_reader);

    assert(cj4me_dataset_reader_open(&reader, path_skipped));
    assert(reader.record_count > 0u);
    cj4me_dataset_reader_close(&reader);

    assert(remove(path_a) == 0);
    assert(remove(path_b) == 0);
    assert(remove(path_skipped) == 0);
  }
}
