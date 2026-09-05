#ifndef CJ4ME_SELFPLAY_H
#define CJ4ME_SELFPLAY_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "cjong4/core/action.h"
#include "cjong4/core/player.h"
#include "cjong4/core/rules.h"
#include "cjong4/manager/player_view.h"

typedef bool (*cj4me_score_actions_fn)(
    void *context, const cj4_player_view *view, const cj4_rules *rules,
    const cj4_action *actions, uint8_t action_count, uint8_t *out_index);

typedef float (*cj4me_reward_fn)(void *context, cj4_player player,
                                 int32_t score_before, int32_t score_after,
                                 float reward_scale);

typedef struct {
  uint32_t games;
  uint64_t seed;
  float epsilon;
  float reward_scale;
  uint32_t max_steps_per_game;
  uint32_t max_records_per_round;
  const char *output_path;
  cj4me_score_actions_fn score_actions;
  void *score_context;
  cj4me_reward_fn reward;
  void *reward_context;
} cj4me_selfplay_config;

bool cj4me_generate_dataset(const cj4me_selfplay_config *config, char *error,
                            size_t error_size);

#endif /* CJ4ME_SELFPLAY_H */
