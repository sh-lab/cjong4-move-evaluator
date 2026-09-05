#include "cjong4_move_evaluator/feature.h"

#include <stddef.h>
#include <string.h>

#include "cjong4/core/location.h"
#include "cjong4/core/phase.h"
#include "cjong4/core/player.h"
#include "cjong4/core/state_query.h"
#include "cjong4/core/tile.h"
#include "cjong4/core/wind.h"

static uint8_t relative_player(cj4_player absolute, cj4_player observer) {
  return (uint8_t)((absolute + CJ4_PLAYER_COUNT - observer) % CJ4_PLAYER_COUNT);
}

static float clamp_unit(float value) {
  if (value < -1.0f)
    return -1.0f;
  if (value > 1.0f)
    return 1.0f;
  return value;
}

static bool tile_is_aka(const cj4_rules *rules, cj4_tile_id tile) {
  return rules->aka_tiles[tile] != 0u;
}

static bool encode_optional_tile(cj4_tile_id tile, const cj4_rules *rules,
                                 float *out, float *out_aka) {
  if (tile == CJ4_TILE_ID_INVALID)
    return true;
  if (!cj4_tile_id_is_valid(tile))
    return false;

  out[0] = 1.0f;
  out[1u + cj4_tile_get_type(tile)] = 1.0f;
  *out_aka = tile_is_aka(rules, tile) ? 1.0f : 0.0f;
  return true;
}

bool cj4me_encode_features(const cj4_player_view *view, const cj4_rules *rules,
                           const cj4_action *action,
                           float out_features[CJ4ME_FEATURE_COUNT]) {
  cj4_hand hand;
  cj4_discard_list discards;
  cj4_dora_indicator_list dora;
  uint8_t latest_discard_history[CJ4ME_FEATURE_DISCARD_ORDER_COUNT];
  bool has_discard_history[CJ4ME_FEATURE_DISCARD_ORDER_COUNT];

  if (!out_features)
    return false;
  memset(out_features, 0, sizeof(float) * CJ4ME_FEATURE_COUNT);
  memset(latest_discard_history, 0, sizeof(latest_discard_history));
  memset(has_discard_history, 0, sizeof(has_discard_history));

  if (!view || !rules || !action || !cj4_rules_validate(rules) ||
      view->player >= CJ4_PLAYER_COUNT ||
      view->current_player >= CJ4_PLAYER_COUNT ||
      view->dealer >= CJ4_PLAYER_COUNT || view->round_wind >= CJ4_WIND_COUNT ||
      (unsigned)view->phase > (unsigned)CJ4_PHASE_GAME_END ||
      (unsigned)action->type > (unsigned)CJ4_ACTION_PASS ||
      action->player >= CJ4_PLAYER_COUNT || action->tile_count > 4u) {
    return false;
  }

  hand = cj4_location_collect_hand(view->locations, view->player);
  if (hand.count > CJ4_MAX_HAND_TILES)
    return false;
  for (uint8_t i = 0; i < hand.count; ++i) {
    if (!cj4_tile_id_is_valid(hand.items[i]))
      return false;
    out_features[CJ4ME_FEATURE_HAND_OFFSET +
                 cj4_tile_get_type(hand.items[i])] += 0.25f;
    if (tile_is_aka(rules, hand.items[i])) {
      out_features[CJ4ME_FEATURE_HAND_AKA_OFFSET +
                   cj4_tile_get_type(hand.items[i])] += 0.25f;
    }
  }

  discards = cj4_location_collect_discards(view->locations);
  if (discards.count > CJ4_MAX_DISCARDS)
    return false;
  for (uint8_t i = 0; i < discards.count; ++i) {
    const cj4_discard *discard = &discards.items[i];
    uint8_t relative;
    uint8_t history_index;
    cj4_tile_type type;
    size_t index;

    if (discard->player >= CJ4_PLAYER_COUNT ||
        !cj4_tile_id_is_valid(discard->tile))
      return false;
    relative = relative_player(discard->player, view->player);
    type = cj4_tile_get_type(discard->tile);
    history_index = cj4_location_discard_history_index(
        view->locations[discard->tile].discard_history);
    if (history_index > CJ4_DISCARD_HISTORY_MAX)
      return false;
    index = (size_t)relative * CJ4_TILE_TYPE_COUNT + type;
    out_features[CJ4ME_FEATURE_DISCARD_COUNT_OFFSET + index] += 0.25f;
    if (tile_is_aka(rules, discard->tile))
      out_features[CJ4ME_FEATURE_DISCARD_AKA_OFFSET + index] += 0.25f;
    if (discard->is_tsumogiri)
      out_features[CJ4ME_FEATURE_DISCARD_TSUMOGIRI_OFFSET + index] += 0.25f;
    if (discard->is_riichi)
      out_features[CJ4ME_FEATURE_DISCARD_RIICHI_OFFSET + index] += 0.25f;
    if (!has_discard_history[index] ||
        history_index > latest_discard_history[index]) {
      latest_discard_history[index] = history_index;
      has_discard_history[index] = true;
    }
  }
  for (size_t index = 0; index < CJ4ME_FEATURE_DISCARD_ORDER_COUNT; ++index) {
    if (has_discard_history[index]) {
      out_features[CJ4ME_FEATURE_DISCARD_ORDER_OFFSET + index] =
          (2.0f * (float)latest_discard_history[index] /
           (float)CJ4_DISCARD_HISTORY_MAX) -
          1.0f;
    }
  }

  for (cj4_player player = 0; player < CJ4_PLAYER_COUNT; ++player) {
    cj4_meld_list melds = cj4_location_collect_melds(view->locations, player);
    uint8_t relative = relative_player(player, view->player);

    if (melds.count > CJ4_MAX_MELDS)
      return false;
    for (uint8_t i = 0; i < melds.count; ++i) {
      const cj4_meld *meld = &melds.items[i];
      if (meld->size > 4u || meld->type > CJ4_MELD_KAKAN)
        return false;
      out_features[CJ4ME_FEATURE_MELD_TYPE_OFFSET + (size_t)relative * 5u +
                   meld->type] += 0.25f;
      for (uint8_t j = 0; j < meld->size; ++j) {
        if (!cj4_tile_id_is_valid(meld->tiles[j]))
          return false;
        out_features[CJ4ME_FEATURE_MELD_TILE_OFFSET +
                     (size_t)relative * CJ4_TILE_TYPE_COUNT +
                     cj4_tile_get_type(meld->tiles[j])] += 0.25f;
        if (tile_is_aka(rules, meld->tiles[j])) {
          out_features[CJ4ME_FEATURE_MELD_AKA_OFFSET +
                       (size_t)relative * CJ4_TILE_TYPE_COUNT +
                       cj4_tile_get_type(meld->tiles[j])] += 0.25f;
        }
      }
    }
  }

  dora = cj4_location_collect_dora_indicators(view->locations);
  if (dora.count > CJ4_MAX_DORA_INDICATORS)
    return false;
  for (uint8_t i = 0; i < dora.count; ++i) {
    if (!cj4_tile_id_is_valid(dora.items[i]))
      return false;
    out_features[CJ4ME_FEATURE_DORA_OFFSET +
                 cj4_tile_get_type(dora.items[i])] += 0.25f;
    if (tile_is_aka(rules, dora.items[i])) {
      out_features[CJ4ME_FEATURE_DORA_AKA_OFFSET +
                   cj4_tile_get_type(dora.items[i])] += 0.25f;
    }
  }

  if (!encode_optional_tile(
          view->draw_tile, rules, &out_features[CJ4ME_FEATURE_DRAW_TILE_OFFSET],
          &out_features[CJ4ME_FEATURE_DRAW_TILE_AKA_OFFSET]) ||
      !encode_optional_tile(
          view->last_discard, rules,
          &out_features[CJ4ME_FEATURE_LAST_DISCARD_OFFSET],
          &out_features[CJ4ME_FEATURE_LAST_DISCARD_AKA_OFFSET]) ||
      !encode_optional_tile(view->kan_tile, rules,
                            &out_features[CJ4ME_FEATURE_KAN_TILE_OFFSET],
                            &out_features[CJ4ME_FEATURE_KAN_TILE_AKA_OFFSET])) {
    return false;
  }

  for (cj4_player player = 0; player < CJ4_PLAYER_COUNT; ++player) {
    uint8_t relative = relative_player(player, view->player);
    out_features[CJ4ME_FEATURE_SCORE_OFFSET + relative] =
        clamp_unit((float)view->scores[player] / 100000.0f);
    out_features[CJ4ME_FEATURE_RIICHI_STATE_OFFSET + relative] =
        view->is_riichi[player] ? 1.0f : 0.0f;
  }

  out_features[CJ4ME_FEATURE_CURRENT_PLAYER_OFFSET +
               relative_player(view->current_player, view->player)] = 1.0f;
  out_features[CJ4ME_FEATURE_DEALER_OFFSET +
               relative_player(view->dealer, view->player)] = 1.0f;
  out_features[CJ4ME_FEATURE_ROUND_WIND_OFFSET + view->round_wind] = 1.0f;
  out_features[CJ4ME_FEATURE_PHASE_OFFSET + view->phase] = 1.0f;
  out_features[CJ4ME_FEATURE_HONBA_OFFSET] =
      clamp_unit((float)view->honba / 16.0f);
  out_features[CJ4ME_FEATURE_RIICHI_STICKS_OFFSET] =
      clamp_unit((float)view->riichi_sticks / 16.0f);
  out_features[CJ4ME_FEATURE_SELF_FLAGS_OFFSET] =
      view->temporary_furiten ? 1.0f : 0.0f;
  out_features[CJ4ME_FEATURE_SELF_FLAGS_OFFSET + 1u] =
      view->riichi_furiten ? 1.0f : 0.0f;
  out_features[CJ4ME_FEATURE_SELF_FLAGS_OFFSET + 2u] =
      view->first_turn_uninterrupted ? 1.0f : 0.0f;

  out_features[CJ4ME_FEATURE_ACTION_TYPE_OFFSET + action->type] = 1.0f;
  out_features[CJ4ME_FEATURE_ACTION_PLAYER_OFFSET +
               relative_player(action->player, view->player)] = 1.0f;
  if (!encode_optional_tile(
          action->tile, rules, &out_features[CJ4ME_FEATURE_ACTION_TILE_OFFSET],
          &out_features[CJ4ME_FEATURE_ACTION_TILE_AKA_OFFSET])) {
    return false;
  }
  for (uint8_t i = 0; i < action->tile_count; ++i) {
    if (!cj4_tile_id_is_valid(action->tiles[i]))
      return false;
    out_features[CJ4ME_FEATURE_ACTION_TILES_OFFSET +
                 cj4_tile_get_type(action->tiles[i])] += 0.25f;
    if (tile_is_aka(rules, action->tiles[i])) {
      out_features[CJ4ME_FEATURE_ACTION_TILES_AKA_OFFSET +
                   cj4_tile_get_type(action->tiles[i])] += 0.25f;
    }
  }
  out_features[CJ4ME_FEATURE_ACTION_TILE_COUNT_OFFSET] =
      (float)action->tile_count * 0.25f;

  return true;
}
