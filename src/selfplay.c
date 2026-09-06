#include "selfplay.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "cjong4/core/hand_analysis.h"
#include "cjong4/core/phase.h"
#include "cjong4/core/rules.h"
#include "cjong4/core/state_init.h"
#include "cjong4/core/state_query.h"
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
  int32_t round_end_scores[CJ4_PLAYER_COUNT];
  uint32_t last_causal_record[CJ4_PLAYER_COUNT];
  bool round_end_scores_valid;
  bool failed;
  const char *failure;
  const cj4_rules *rules;
} selfplay_context;

static bool action_is_open_call(cj4_action_type type) {
  return type == CJ4_ACTION_CHI || type == CJ4_ACTION_PON ||
         type == CJ4_ACTION_MINKAN;
}

static bool action_can_cause_ron(cj4_action_type type) {
  return type == CJ4_ACTION_DISCARD || type == CJ4_ACTION_RIICHI ||
         type == CJ4_ACTION_KAKAN || type == CJ4_ACTION_ANKAN;
}

static uint8_t available_call_mask(const cj4_action *actions,
                                   uint8_t action_count) {
  uint8_t mask = 0u;
  for (uint8_t i = 0; i < action_count; ++i) {
    switch (actions[i].type) {
    case CJ4_ACTION_CHI:
      mask |= CJ4ME_CALL_AVAILABLE_CHI;
      break;
    case CJ4_ACTION_PON:
      mask |= CJ4ME_CALL_AVAILABLE_PON;
      break;
    case CJ4_ACTION_MINKAN:
      mask |= CJ4ME_CALL_AVAILABLE_MINKAN;
      break;
    default:
      break;
    }
  }
  return mask;
}

static bool action_type_available(const cj4_action *actions,
                                  uint8_t action_count, cj4_action_type type) {
  for (uint8_t i = 0; i < action_count; ++i)
    if (actions[i].type == type)
      return true;
  return false;
}

static bool player_is_menzen(const cj4_player_view *view) {
  cj4_meld_list melds =
      cj4_location_collect_melds(view->locations, view->player);
  for (uint8_t i = 0; i < melds.count; ++i)
    if (melds.items[i].type != CJ4_MELD_ANKAN)
      return false;
  return true;
}

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
  if (actions[selected].player >= CJ4_PLAYER_COUNT) {
    context->failed = true;
    context->failure = "selected action has invalid player";
    return actions[0];
  }
  if (action_can_cause_ron(actions[selected].type))
    context->last_causal_record[actions[selected].player] = UINT32_MAX;
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
  record->decision_discard_count =
      cj4_location_collect_discards(view->locations).count;
  record->available_call_mask = available_call_mask(actions, action_count);
  if (player_is_menzen(view))
    record->fact_flags |= CJ4ME_FACT_WAS_MENZEN;
  if (action_is_open_call(actions[selected].type)) {
    record->fact_flags |= CJ4ME_FACT_CHOSE_CALL;
    if ((record->fact_flags & CJ4ME_FACT_WAS_MENZEN) != 0u)
      record->fact_flags |= CJ4ME_FACT_OPENED_HAND;
  }
  if (record->available_call_mask != 0u)
    record->fact_flags |= CJ4ME_FACT_CALL_AVAILABLE;
  if (action_type_available(actions, action_count, CJ4_ACTION_RIICHI))
    record->fact_flags |= CJ4ME_FACT_RIICHI_AVAILABLE;
  if (actions[selected].type == CJ4_ACTION_RIICHI)
    record->fact_flags |= CJ4ME_FACT_CHOSE_RIICHI;
  if (action_can_cause_ron(actions[selected].type))
    context->last_causal_record[actions[selected].player] =
        context->pending_count;
  ++context->pending_count;
  return actions[selected];
}

static cj4me_tenpai_status tenpai_status_at_draw(const cj4_mahjong *settled,
                                                 cj4_player player,
                                                 int32_t settlement_delta,
                                                 bool has_nagashi_mangan) {
  if (cj4_state_round_end_type(settled) != CJ4_ROUND_END_EXHAUSTIVE_DRAW)
    return CJ4ME_TENPAI_UNKNOWN;
  if (has_nagashi_mangan)
    return cj4_is_shape_tenpai(settled, player) ? CJ4ME_TENPAI_YES
                                                : CJ4ME_TENPAI_NO;
  if (settlement_delta > 0)
    return CJ4ME_TENPAI_YES;
  if (settlement_delta < 0)
    return CJ4ME_TENPAI_NO;
  return settled->next_dealer == settled->dealer ? CJ4ME_TENPAI_YES
                                                 : CJ4ME_TENPAI_NO;
}

static uint32_t deal_in_action_index(const selfplay_context *context,
                                     const cj4_mahjong *settled) {
  cj4_player discarder;
  if (cj4_state_round_end_type(settled) != CJ4_ROUND_END_RON)
    return UINT32_MAX;
  discarder = cj4_state_current_player(settled);
  return context->last_causal_record[discarder];
}

static bool flush_round(selfplay_context *context, const cj4_mahjong *settled,
                        cj4me_dataset_writer *writer) {
  cj4me_reward_fn reward =
      context->config->reward ? context->config->reward : default_reward;
  int32_t score_deltas[CJ4_PLAYER_COUNT];
  int32_t settlement_deltas[CJ4_PLAYER_COUNT];
  uint8_t tenpai_statuses[CJ4_PLAYER_COUNT];
  uint32_t cause_index;
  cj4_player discarder = CJ4_PLAYER_COUNT;
  bool has_nagashi_mangan = false;
  if (!context->round_end_scores_valid)
    return false;
  cause_index = deal_in_action_index(context, settled);
  if (cj4_state_round_end_type(settled) == CJ4_ROUND_END_RON)
    discarder = cj4_state_current_player(settled);
  if (cj4_state_round_end_type(settled) == CJ4_ROUND_END_EXHAUSTIVE_DRAW) {
    for (cj4_player player = 0; player < CJ4_PLAYER_COUNT; ++player)
      has_nagashi_mangan |= cj4_is_nagashi_mangan(settled, player);
  }
  for (cj4_player player = 0; player < CJ4_PLAYER_COUNT; ++player) {
    int64_t score_delta = (int64_t)settled->scores[player] -
                          (int64_t)context->round_start_scores[player];
    int64_t settlement_delta = (int64_t)settled->scores[player] -
                               (int64_t)context->round_end_scores[player];
    if (score_delta < INT32_MIN || score_delta > INT32_MAX ||
        settlement_delta < INT32_MIN || settlement_delta > INT32_MAX)
      return false;
    score_deltas[player] = (int32_t)score_delta;
    settlement_deltas[player] = (int32_t)settlement_delta;
    tenpai_statuses[player] = (uint8_t)tenpai_status_at_draw(
        settled, player, settlement_deltas[player], has_nagashi_mangan);
  }
  for (uint32_t i = 0; i < context->pending_count; ++i) {
    cj4me_dataset_record *record = &context->pending[i];
    cj4_player player = record->action_player;
    if (player >= CJ4_PLAYER_COUNT)
      return false;
    record->target =
        reward(context->config->reward_context, player,
               context->round_start_scores[player], settled->scores[player],
               context->config->reward_scale);
    record->score_delta = score_deltas[player];
    record->settlement_delta = settlement_deltas[player];
    record->round_discard_count = settled->discard_count;
    if (record->decision_discard_count > record->round_discard_count)
      return false;
    record->discards_until_end =
        record->round_discard_count - record->decision_discard_count;
    record->round_end_type = (uint8_t)cj4_state_round_end_type(settled);
    record->tenpai_status = tenpai_statuses[player];
    if (cj4_state_is_winner(settled, player)) {
      record->fact_flags |= CJ4ME_FACT_PLAYER_WON;
      if (record->settlement_delta > 0)
        record->win_points = record->settlement_delta;
    }
    if (player == discarder) {
      record->fact_flags |= CJ4ME_FACT_PLAYER_DEALT_IN;
      if (record->settlement_delta < 0)
        record->deal_in_points = -record->settlement_delta;
    }
    if (i == cause_index)
      record->fact_flags |= CJ4ME_FACT_DEAL_IN_ACTION;
    if (!isfinite(record->target) ||
        !cj4me_dataset_writer_append(writer, record)) {
      return false;
    }
  }
  context->pending_count = 0u;
  context->round_end_scores_valid = false;
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
    context.round_end_scores_valid = false;
    for (cj4_player player = 0; player < CJ4_PLAYER_COUNT; ++player)
      context.last_causal_record[player] = UINT32_MAX;

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
        context.round_end_scores_valid = false;
        for (cj4_player player = 0; player < CJ4_PLAYER_COUNT; ++player)
          context.last_causal_record[player] = UINT32_MAX;
        continue;
      }

      if (cj4_state_phase(&state) == CJ4_PHASE_ROUND_END) {
        memcpy(context.round_end_scores, state.scores,
               sizeof(context.round_end_scores));
        context.round_end_scores_valid = true;
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
