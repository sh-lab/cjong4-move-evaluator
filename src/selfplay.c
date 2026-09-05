#include "selfplay.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "cjong4/core/phase.h"
#include "cjong4/core/rules.h"
#include "cjong4/core/state_init.h"
#include "cjong4/core/state_round.h"
#include "cjong4/manager/delegate.h"
#include "cjong4/manager/manager.h"
#include "cjong4_move_evaluator/dataset.h"
#include "cjong4_move_evaluator/feature.h"
#include "rng.h"

typedef struct {
  const cj4me_selfplay_config *config;
  cj4me_rng rng;
  cj4me_dataset_record *pending;
  uint32_t pending_count;
  int32_t round_start_scores[CJ4_PLAYER_COUNT];
  bool failed;
  const char *failure;
  const cj4_rules *rules;
} selfplay_context;

static void set_error(char *error, size_t error_size, const char *message) {
  if (error && error_size > 0u)
    (void)snprintf(error, error_size, "%s", message);
}

static float default_reward(void *context, cj4_player player,
                            int32_t score_before, int32_t score_after,
                            float reward_scale) {
  int64_t difference;
  (void)context;
  (void)player;
  difference = (int64_t)score_after - (int64_t)score_before;
  return (float)difference / reward_scale;
}

static uint8_t forced_or_random_index(selfplay_context *context,
                                      const cj4_player_view *view,
                                      const cj4_action *actions,
                                      uint8_t action_count) {
  uint8_t selected;
  for (uint8_t i = 0; i < action_count; ++i)
    if (actions[i].type == CJ4_ACTION_TSUMO)
      return i;
  for (uint8_t i = 0; i < action_count; ++i)
    if (actions[i].type == CJ4_ACTION_RON)
      return i;

  if (context->config->score_actions &&
      cj4me_rng_unit(&context->rng) >= context->config->epsilon) {
    if (!context->config->score_actions(context->config->score_context, view,
                                        context->rules, actions, action_count,
                                        &selected) ||
        selected >= action_count) {
      context->failed = true;
      context->failure = "model action evaluation failed";
      return 0u;
    }
    return selected;
  }

  return (uint8_t)cj4me_rng_bounded(&context->rng, action_count);
}

static cj4_action selfplay_decide(void *opaque, const cj4_player_view *view,
                                  const cj4_action *actions,
                                  uint8_t action_count) {
  selfplay_context *context = (selfplay_context *)opaque;
  uint8_t selected;
  cj4me_dataset_record *record;

  if (!context || !view || !actions || action_count == 0u ||
      action_count > CJ4M_MAX_ACTIONS) {
    if (context) {
      context->failed = true;
      context->failure = "invalid delegate input";
    }
    return (cj4_action){0};
  }

  selected = forced_or_random_index(context, view, actions, action_count);
  if (context->failed)
    return actions[0];
  if (action_count == 1u || actions[selected].type == CJ4_ACTION_TSUMO ||
      actions[selected].type == CJ4_ACTION_RON) {
    return actions[selected];
  }
  if (context->pending_count >= context->config->max_records_per_round) {
    context->failed = true;
    context->failure = "maximum records per round exceeded";
    return actions[selected];
  }

  record = &context->pending[context->pending_count];
  memset(record, 0, sizeof(*record));
  if (!cj4me_encode_features(view, context->rules, &actions[selected],
                             record->features)) {
    context->failed = true;
    context->failure = "feature encoding failed";
    return actions[selected];
  }
  record->action_player = actions[selected].player;
  record->action_type = (uint8_t)actions[selected].type;
  ++context->pending_count;
  return actions[selected];
}

static bool flush_round(selfplay_context *context, const cj4_mahjong *settled,
                        cj4me_dataset_writer *writer) {
  cj4me_reward_fn reward =
      context->config->reward ? context->config->reward : default_reward;
  for (uint32_t i = 0; i < context->pending_count; ++i) {
    cj4me_dataset_record *record = &context->pending[i];
    cj4_player player = record->action_player;
    if (player >= CJ4_PLAYER_COUNT)
      return false;
    record->target =
        reward(context->config->reward_context, player,
               context->round_start_scores[player], settled->scores[player],
               context->config->reward_scale);
    if (!isfinite(record->target) ||
        !cj4me_dataset_writer_append(writer, record)) {
      return false;
    }
  }
  context->pending_count = 0u;
  return true;
}

bool cj4me_generate_dataset(const cj4me_selfplay_config *config, char *error,
                            size_t error_size) {
  cj4_rules rules;
  cj4me_dataset_writer writer;
  selfplay_context context;
  cj4m_player_delegate delegates[CJ4_PLAYER_COUNT];
  cj4_tile_id wall[CJ4_TILE_ID_COUNT];
  size_t pending_size;
  bool writer_open = false;

  if (!config || !config->output_path || config->games == 0u ||
      !isfinite(config->epsilon) || config->epsilon < 0.0f ||
      config->epsilon > 1.0f || !isfinite(config->reward_scale) ||
      config->reward_scale <= 0.0f || config->max_steps_per_game == 0u ||
      config->max_records_per_round == 0u) {
    set_error(error, error_size, "invalid self-play configuration");
    return false;
  }
  pending_size =
      (size_t)config->max_records_per_round * sizeof(cj4me_dataset_record);
  if (pending_size / sizeof(cj4me_dataset_record) !=
      config->max_records_per_round) {
    set_error(error, error_size, "round buffer size overflows");
    return false;
  }

  memset(&context, 0, sizeof(context));
  context.config = config;
  context.pending = (cj4me_dataset_record *)calloc(
      config->max_records_per_round, sizeof(cj4me_dataset_record));
  if (!context.pending) {
    set_error(error, error_size, "unable to allocate round buffer");
    return false;
  }
  cj4me_rng_seed(&context.rng, config->seed);
  for (cj4_player player = 0; player < CJ4_PLAYER_COUNT; ++player) {
    delegates[player].ctx = &context;
    delegates[player].decide = selfplay_decide;
  }

  rules = cj4_rules_default();
  if (!cj4_rules_validate(&rules)) {
    set_error(error, error_size, "cjong4 default rules are invalid");
    free(context.pending);
    return false;
  }
  context.rules = &rules;
  if (!cj4me_dataset_writer_open(&writer, config->output_path)) {
    set_error(error, error_size, "unable to open output dataset");
    free(context.pending);
    return false;
  }
  writer_open = true;

  for (uint32_t game = 0; game < config->games; ++game) {
    cj4_mahjong state;
    uint32_t steps = 0u;

    cj4me_rng_shuffle_wall(&context.rng, wall);
    state = cj4_create_initial_state(wall, &rules);
    if (cj4_state_phase(&state) == CJ4_PHASE_GAME_END) {
      context.failed = true;
      context.failure = "cjong4 rejected a generated wall";
      break;
    }
    memcpy(context.round_start_scores, state.scores,
           sizeof(context.round_start_scores));
    context.pending_count = 0u;

    while (cj4_state_phase(&state) != CJ4_PHASE_GAME_END) {
      if (steps++ >= config->max_steps_per_game) {
        context.failed = true;
        context.failure = "maximum steps per game exceeded";
        break;
      }

      if (cj4_state_phase(&state) == CJ4_PHASE_SETTLE) {
        if (!flush_round(&context, &state, &writer)) {
          context.failed = true;
          context.failure = "unable to write round records";
          break;
        }
        if (cj4_can_game_end(state)) {
          state = cj4_do_game_end(state);
          continue;
        }
        if (!cj4_can_next_round(state)) {
          context.failed = true;
          context.failure = "settled state cannot advance";
          break;
        }
        cj4me_rng_shuffle_wall(&context.rng, wall);
        state = cj4_do_next_round(state, wall, &rules);
        memcpy(context.round_start_scores, state.scores,
               sizeof(context.round_start_scores));
        continue;
      }

      state = cj4m_step(&state, &rules, delegates);
      if (context.failed)
        break;
    }
    if (context.failed)
      break;
  }

  if (writer_open && !cj4me_dataset_writer_close(&writer)) {
    context.failed = true;
    context.failure = "unable to finalize output dataset";
  }
  if (context.failed) {
    set_error(error, error_size, context.failure);
    (void)remove(config->output_path);
  }
  free(context.pending);
  return !context.failed;
}
