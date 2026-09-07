#include "cjong4_move_evaluator/policy_filter.h"

#include <math.h>
#include <string.h>

#include "cjong4/core/state_query.h"
#include "cjong4/core/tile_const.h"
#include "cjong4/player/hand_analysis.h"

static bool action_is_open_call(cj4_action_type type) {
  return type == CJ4_ACTION_CHI || type == CJ4_ACTION_PON ||
         type == CJ4_ACTION_MINKAN;
}

static bool action_is_discard_choice(cj4_action_type type) {
  return type == CJ4_ACTION_DISCARD || type == CJ4_ACTION_RIICHI;
}

static int shanten_minimum(const cj4_shanten_result *result) {
  int minimum = CJ4_SHANTEN_NOT_APPLICABLE;
  if (result->standard < minimum)
    minimum = result->standard;
  if (result->chiitoitsu < minimum)
    minimum = result->chiitoitsu;
  if (result->kokushi < minimum)
    minimum = result->kokushi;
  return minimum;
}

static int find_forced_action(const cj4_action *actions, uint8_t action_count) {
  for (uint8_t i = 0; i < action_count; ++i)
    if (actions[i].type == CJ4_ACTION_TSUMO)
      return i;
  for (uint8_t i = 0; i < action_count; ++i)
    if (actions[i].type == CJ4_ACTION_RON)
      return i;
  return -1;
}

static bool meld_is_called_yakuhai(const cj4_player_view *view,
                                   cj4_player player, const cj4_meld *meld) {
  cj4_tile_type type;
  cj4_tile_type seat_wind;
  cj4_tile_type round_wind;
  if (!meld || meld->size == 0u ||
      (meld->type != CJ4_MELD_PON && meld->type != CJ4_MELD_MINKAN &&
       meld->type != CJ4_MELD_KAKAN) ||
      !cj4_tile_id_is_valid(meld->tiles[0])) {
    return false;
  }
  type = cj4_tile_get_type(meld->tiles[0]);
  seat_wind = (cj4_tile_type)(CJ4_TILE_TYPE_EAST +
                              (player + CJ4_PLAYER_COUNT - view->dealer) %
                                  CJ4_PLAYER_COUNT);
  round_wind =
      (cj4_tile_type)(CJ4_TILE_TYPE_EAST + (cj4_tile_type)view->round_wind);
  return type >= CJ4_TILE_TYPE_HAKU || type == seat_wind || type == round_wind;
}

static bool player_has_called_yakuhai(const cj4_player_view *view,
                                      cj4_player player) {
  cj4_meld_list melds = cj4_location_collect_melds(view->locations, player);
  for (uint8_t i = 0; i < melds.count; ++i)
    if (meld_is_called_yakuhai(view, player, &melds.items[i]))
      return true;
  return false;
}

static bool player_has_tsumogiri_streak(const cj4_player_view *view,
                                        cj4_player player, uint8_t required) {
  cj4_discard_list discards;
  if (required == 0u)
    return false;
  discards = cj4_location_collect_player_discards(view->locations, player);
  if (discards.count < required)
    return false;
  for (uint8_t i = 0; i < required; ++i)
    if (!discards.items[discards.count - 1u - i].is_tsumogiri)
      return false;
  return true;
}

static uint8_t collect_threat_mask(const cj4me_policy_filter_config *config,
                                   const cj4_player_view *view) {
  uint8_t mask = 0u;
  for (cj4_player player = 0; player < CJ4_PLAYER_COUNT; ++player) {
    bool late_tsumogiri;
    if (player == view->player)
      continue;
    late_tsumogiri =
        view->live_wall_remaining <= config->safe_live_wall_threshold &&
        player_has_tsumogiri_streak(view, player,
                                    config->safe_tsumogiri_streak);
    if (view->is_riichi[player] || player_has_called_yakuhai(view, player) ||
        late_tsumogiri) {
      mask |= (uint8_t)(1u << player);
    }
  }
  return mask;
}

static bool player_has_discarded_type(const cj4_player_view *view,
                                      cj4_player player, cj4_tile_type type) {
  cj4_discard_list discards =
      cj4_location_collect_player_discards(view->locations, player);
  for (uint8_t i = 0; i < discards.count; ++i)
    if (cj4_tile_id_is_valid(discards.items[i].tile) &&
        cj4_tile_get_type(discards.items[i].tile) == type) {
      return true;
    }
  return false;
}

static uint8_t genbutsu_coverage(const cj4_player_view *view, cj4_tile_id tile,
                                 uint8_t threat_mask) {
  uint8_t coverage = 0u;
  cj4_tile_type type;
  if (!cj4_tile_id_is_valid(tile))
    return 0u;
  type = cj4_tile_get_type(tile);
  for (cj4_player player = 0; player < CJ4_PLAYER_COUNT; ++player)
    if ((threat_mask & (uint8_t)(1u << player)) != 0u &&
        player_has_discarded_type(view, player, type)) {
      ++coverage;
    }
  return coverage;
}

static uint8_t unique_yaochu_count(const cj4_player_view *view) {
  bool present[CJ4_TILE_TYPE_COUNT] = {false};
  uint8_t count = 0u;
  for (cj4_tile_id tile = CJ4_TILE_ID_MIN; tile <= CJ4_TILE_ID_MAX; ++tile) {
    cj4_tile_type type;
    const cj4_location *location = &view->locations[tile];
    if (!cj4_location_is_hand(location->placement) ||
        cj4_location_placement_player(location->placement) != view->player) {
      continue;
    }
    type = cj4_tile_get_type(tile);
    if (cj4_tile_type_is_yaochu(type) && !present[type]) {
      present[type] = true;
      ++count;
    }
  }
  return count;
}

static bool kokushi_is_promising(const cj4me_policy_filter_config *config,
                                 const cj4_player_view *view,
                                 bool *out_promising) {
  cj4_shanten_result result;
  int best_other;
  *out_promising = false;
  if (unique_yaochu_count(view) < config->kokushi_min_unique_yaochu)
    return true;
  if (!cj4p_calculate_shanten(view, &result))
    return false;
  if (result.kokushi == CJ4_SHANTEN_NOT_APPLICABLE)
    return true;
  best_other = result.standard;
  if (result.chiitoitsu < best_other)
    best_other = result.chiitoitsu;
  *out_promising =
      result.kokushi <= best_other + config->kokushi_max_shanten_gap;
  return true;
}

static uint8_t bit_count(uint8_t value) {
  uint8_t count = 0u;
  while (value != 0u) {
    count += value & 1u;
    value >>= 1u;
  }
  return count;
}

static bool action_breaks_kokushi(cj4_action_type type) {
  return type == CJ4_ACTION_CHI || type == CJ4_ACTION_PON ||
         type == CJ4_ACTION_ANKAN || type == CJ4_ACTION_MINKAN ||
         type == CJ4_ACTION_KAKAN || type == CJ4_ACTION_ABORTIVE_DRAW;
}

static void allow_all(cj4me_policy_filter_result *result,
                      uint8_t action_count) {
  result->allowed_count = action_count;
  for (uint8_t i = 0; i < action_count; ++i)
    result->allowed[i] = true;
}

static void recount_allowed(cj4me_policy_filter_result *result,
                            uint8_t action_count) {
  result->allowed_count = 0u;
  for (uint8_t i = 0; i < action_count; ++i)
    if (result->allowed[i])
      ++result->allowed_count;
}

cj4me_policy_filter_config cj4me_policy_filter_default(void) {
  cj4me_policy_filter_config config;
  memset(&config, 0, sizeof(config));
  return config;
}

bool cj4me_policy_filter_config_validate(
    const cj4me_policy_filter_config *config) {
  return config && !(config->exclude_open_calls && config->require_open_call) &&
         !(config->exclude_riichi && config->require_riichi) &&
         isfinite(config->open_call_bias) &&
         isfinite(config->pass_with_call_bias) &&
         isfinite(config->riichi_bias) &&
         isfinite(config->best_shanten_discard_bias) &&
         isfinite(config->safe_discard_bias) &&
         isfinite(config->kokushi_discard_bias) &&
         config->safe_live_wall_threshold <= CJ4_MAX_DRAWS &&
         config->safe_tsumogiri_streak <= CJ4_DISCARD_INDEX_MAX + 1u &&
         config->kokushi_min_unique_yaochu <= 13u &&
         config->kokushi_max_shanten_gap <= 13u;
}

bool cj4me_policy_filter_preset(cj4me_policy_profile profile,
                                cj4me_policy_filter_mode mode,
                                float soft_strength,
                                cj4me_policy_filter_config *out_config) {
  cj4me_policy_filter_config config = cj4me_policy_filter_default();
  if (!out_config || profile < CJ4ME_POLICY_STANDARD ||
      profile > CJ4ME_POLICY_DAMA ||
      (mode != CJ4ME_POLICY_FILTER_HARD && mode != CJ4ME_POLICY_FILTER_SOFT) ||
      !isfinite(soft_strength) || soft_strength < 0.0f) {
    return false;
  }

  switch (profile) {
  case CJ4ME_POLICY_SAFE:
    config.safe_live_wall_threshold = 40u;
    config.safe_tsumogiri_streak = 3u;
    if (mode == CJ4ME_POLICY_FILTER_HARD)
      config.prefer_genbutsu_when_threatened = true;
    else
      config.safe_discard_bias = soft_strength;
    break;
  case CJ4ME_POLICY_MENZEN:
    if (mode == CJ4ME_POLICY_FILTER_HARD)
      config.exclude_open_calls = true;
    else
      config.open_call_bias = -soft_strength;
    break;
  case CJ4ME_POLICY_CALL:
    if (mode == CJ4ME_POLICY_FILTER_HARD)
      config.require_open_call = true;
    else
      config.open_call_bias = soft_strength;
    break;
  case CJ4ME_POLICY_SPEED:
    if (mode == CJ4ME_POLICY_FILTER_HARD)
      config.keep_best_shanten_discards = true;
    else
      config.best_shanten_discard_bias = soft_strength;
    break;
  case CJ4ME_POLICY_KOKUSHI:
    config.kokushi_min_unique_yaochu = 9u;
    config.kokushi_max_shanten_gap = 1u;
    if (mode == CJ4ME_POLICY_FILTER_HARD)
      config.prefer_kokushi_when_promising = true;
    else
      config.kokushi_discard_bias = soft_strength;
    break;
  case CJ4ME_POLICY_RIICHI:
    if (mode == CJ4ME_POLICY_FILTER_HARD)
      config.require_riichi = true;
    else
      config.riichi_bias = soft_strength;
    break;
  case CJ4ME_POLICY_DAMA:
    if (mode == CJ4ME_POLICY_FILTER_HARD)
      config.exclude_riichi = true;
    else
      config.riichi_bias = -soft_strength;
    break;
  case CJ4ME_POLICY_STANDARD:
    break;
  }
  *out_config = config;
  return true;
}

bool cj4me_policy_filter_apply(const cj4me_policy_filter_config *config,
                               const cj4_player_view *view,
                               const cj4_action *actions, uint8_t action_count,
                               cj4me_policy_filter_result *out_result) {
  cj4me_policy_filter_config no_filter;
  int shanten[CJ4M_MAX_ACTIONS];
  int kokushi_shanten[CJ4M_MAX_ACTIONS];
  uint8_t safety[CJ4M_MAX_ACTIONS] = {0};
  int best_shanten = CJ4_SHANTEN_NOT_APPLICABLE;
  int best_kokushi_shanten = CJ4_SHANTEN_NOT_APPLICABLE;
  uint8_t best_safety = 0u;
  uint8_t threat_count;
  bool has_open_call = false;
  bool has_riichi = false;
  bool need_shanten;
  int forced;

  if (!config) {
    no_filter = cj4me_policy_filter_default();
    config = &no_filter;
  }
  if (!view || view->player >= CJ4_PLAYER_COUNT ||
      view->dealer >= CJ4_PLAYER_COUNT || view->round_wind >= CJ4_WIND_COUNT ||
      view->live_wall_remaining > CJ4_MAX_DRAWS || !actions || !out_result ||
      action_count == 0u || action_count > CJ4M_MAX_ACTIONS ||
      !cj4me_policy_filter_config_validate(config)) {
    return false;
  }
  memset(out_result, 0, sizeof(*out_result));
  allow_all(out_result, action_count);

  forced = find_forced_action(actions, action_count);
  if (forced >= 0) {
    memset(out_result->allowed, 0, sizeof(out_result->allowed));
    out_result->allowed[forced] = true;
    out_result->allowed_count = 1u;
    return true;
  }

  for (uint8_t i = 0; i < action_count; ++i) {
    if (actions[i].player >= CJ4_PLAYER_COUNT)
      return false;
    has_open_call |= action_is_open_call(actions[i].type);
    has_riichi |= actions[i].type == CJ4_ACTION_RIICHI;
    shanten[i] = CJ4_SHANTEN_NOT_APPLICABLE;
    kokushi_shanten[i] = CJ4_SHANTEN_NOT_APPLICABLE;
  }

  if (config->prefer_genbutsu_when_threatened ||
      config->safe_discard_bias != 0.0f) {
    out_result->threat_mask = collect_threat_mask(config, view);
  }
  threat_count = bit_count(out_result->threat_mask);
  if (threat_count > 0u) {
    for (uint8_t i = 0; i < action_count; ++i) {
      if (!action_is_discard_choice(actions[i].type))
        continue;
      safety[i] =
          genbutsu_coverage(view, actions[i].tile, out_result->threat_mask);
      if (safety[i] > best_safety)
        best_safety = safety[i];
    }
  }

  if ((config->prefer_kokushi_when_promising ||
       config->kokushi_discard_bias != 0.0f) &&
      !kokushi_is_promising(config, view, &out_result->kokushi_active)) {
    return false;
  }

  need_shanten = config->keep_best_shanten_discards ||
                 config->best_shanten_discard_bias != 0.0f ||
                 out_result->kokushi_active;
  if (need_shanten) {
    for (uint8_t i = 0; i < action_count; ++i) {
      cj4_shanten_result result;
      if (!action_is_discard_choice(actions[i].type))
        continue;
      if (!cj4p_calculate_shanten_after_discard(view, actions[i].tile,
                                                &result)) {
        return false;
      }
      shanten[i] = shanten_minimum(&result);
      kokushi_shanten[i] = result.kokushi;
      if (shanten[i] < best_shanten)
        best_shanten = shanten[i];
      if (kokushi_shanten[i] < best_kokushi_shanten)
        best_kokushi_shanten = kokushi_shanten[i];
    }
  }

  for (uint8_t i = 0; i < action_count; ++i) {
    const cj4_action_type type = actions[i].type;
    if (config->exclude_open_calls && action_is_open_call(type))
      out_result->allowed[i] = false;
    if (config->require_open_call && has_open_call &&
        !action_is_open_call(type))
      out_result->allowed[i] = false;
    if (config->exclude_riichi && type == CJ4_ACTION_RIICHI)
      out_result->allowed[i] = false;
    if (config->require_riichi && has_riichi && type != CJ4_ACTION_RIICHI)
      out_result->allowed[i] = false;
    if (config->keep_best_shanten_discards &&
        shanten[i] != CJ4_SHANTEN_NOT_APPLICABLE && shanten[i] > best_shanten) {
      out_result->allowed[i] = false;
    }
    if (config->prefer_genbutsu_when_threatened && best_safety > 0u &&
        action_is_discard_choice(type) && safety[i] < best_safety) {
      out_result->allowed[i] = false;
    }
    if (out_result->kokushi_active && config->prefer_kokushi_when_promising) {
      if (action_breaks_kokushi(type))
        out_result->allowed[i] = false;
      if (action_is_discard_choice(type) &&
          kokushi_shanten[i] > best_kokushi_shanten) {
        out_result->allowed[i] = false;
      }
    }

    if (action_is_open_call(type))
      out_result->score_bias[i] += config->open_call_bias;
    if (type == CJ4_ACTION_PASS && has_open_call)
      out_result->score_bias[i] += config->pass_with_call_bias;
    if (type == CJ4_ACTION_RIICHI)
      out_result->score_bias[i] += config->riichi_bias;
    if (shanten[i] != CJ4_SHANTEN_NOT_APPLICABLE &&
        shanten[i] == best_shanten) {
      out_result->score_bias[i] += config->best_shanten_discard_bias;
    }
    if (threat_count > 0u && action_is_discard_choice(type)) {
      out_result->score_bias[i] +=
          config->safe_discard_bias * (float)safety[i] / (float)threat_count;
    }
    if (out_result->kokushi_active) {
      if (action_is_discard_choice(type) &&
          kokushi_shanten[i] == best_kokushi_shanten) {
        out_result->score_bias[i] += config->kokushi_discard_bias;
      }
      if (action_breaks_kokushi(type))
        out_result->score_bias[i] -= config->kokushi_discard_bias;
    }
  }

  recount_allowed(out_result, action_count);
  if (out_result->allowed_count == 0u) {
    allow_all(out_result, action_count);
    out_result->used_fallback = true;
  }
  return true;
}
