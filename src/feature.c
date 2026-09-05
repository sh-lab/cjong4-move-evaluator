#include "cjong4_move_evaluator/feature.h"

#include <stddef.h>
#include <string.h>

#include "cjong4/core/location.h"
#include "cjong4/core/phase.h"
#include "cjong4/core/player.h"
#include "cjong4/core/tile.h"
#include "cjong4/core/wind.h"

enum {
  WALL_NONE_CATEGORY = CJ4_TILE_ID_COUNT,
  DISCARD_CATEGORY_COUNT = CJ4_PLAYER_COUNT * (CJ4_DISCARD_INDEX_MAX + 1),
  DISCARD_NONE_CATEGORY = DISCARD_CATEGORY_COUNT,
  DISCARD_HISTORY_NONE_CATEGORY = CJ4_DISCARD_HISTORY_MAX + 1,
  HAND_NONE_CATEGORY = CJ4_PLAYER_COUNT,
  MELD_CATEGORY_COUNT =
      CJ4_PLAYER_COUNT * (CJ4_MELD_GROUP_MAX + 1) * (CJ4_MELD_KAKAN + 1),
  MELD_NONE_CATEGORY = MELD_CATEGORY_COUNT
};

static uint8_t relative_player(cj4_player absolute, cj4_player observer) {
  return (uint8_t)((absolute + CJ4_PLAYER_COUNT - observer) % CJ4_PLAYER_COUNT);
}

static float normalize_category(uint16_t category, uint16_t maximum) {
  return 2.0f * (float)category / (float)maximum - 1.0f;
}

static bool optional_tile_is_valid(cj4_tile_id tile) {
  return tile == CJ4_TILE_ID_INVALID || cj4_tile_id_is_valid(tile);
}

static bool action_contains_tile(const cj4_action *action, cj4_tile_id tile) {
  for (uint8_t index = 0; index < action->tile_count; ++index)
    if (action->tiles[index] == tile)
      return true;
  return false;
}

static bool encode_tile(const cj4_player_view *view, const cj4_rules *rules,
                        const cj4_action *action, cj4_tile_id tile,
                        float *output) {
  const cj4_location *location = &view->locations[tile];
  uint16_t wall_category = WALL_NONE_CATEGORY;
  uint16_t discard_category = DISCARD_NONE_CATEGORY;
  uint16_t history_category = DISCARD_HISTORY_NONE_CATEGORY;
  uint16_t hand_category = HAND_NONE_CATEGORY;
  uint16_t meld_category = MELD_NONE_CATEGORY;

  if (location->wall != CJ4_LOCATION_NONE) {
    if (location->wall > CJ4_TILE_ID_MAX)
      return false;
    wall_category = location->wall;
  }

  if (cj4_location_is_discard(location->discard)) {
    cj4_player player = cj4_location_discard_player(location->discard);
    uint8_t index = cj4_location_discard_index(location->discard);
    if (player >= CJ4_PLAYER_COUNT || index > CJ4_DISCARD_INDEX_MAX)
      return false;
    discard_category = (uint16_t)relative_player(player, view->player) *
                           (CJ4_DISCARD_INDEX_MAX + 1u) +
                       index;
    output[CJ4ME_TILE_FEATURE_TSUMOGIRI] =
        cj4_location_discard_is_tsumogiri(location->discard) ? 1.0f : 0.0f;
  } else if (location->discard != CJ4_LOCATION_NONE) {
    return false;
  }

  if (cj4_location_is_discard_history(location->discard_history)) {
    uint8_t index =
        cj4_location_discard_history_index(location->discard_history);
    if (index > CJ4_DISCARD_HISTORY_MAX)
      return false;
    history_category = index;
    output[CJ4ME_TILE_FEATURE_RIICHI_DISCARD] =
        cj4_location_discard_is_riichi(location->discard_history) ? 1.0f : 0.0f;
  } else if (location->discard_history != CJ4_LOCATION_NONE) {
    return false;
  }

  if (cj4_location_is_hand(location->placement)) {
    cj4_player player = cj4_location_placement_player(location->placement);
    if (player >= CJ4_PLAYER_COUNT)
      return false;
    hand_category = relative_player(player, view->player);
  } else if (cj4_location_is_meld(location->placement)) {
    cj4_player player = cj4_location_placement_player(location->placement);
    uint8_t group = cj4_location_meld_group(location->placement);
    cj4_meld_type type = cj4_location_meld_type(location->placement);
    if (player >= CJ4_PLAYER_COUNT || group > CJ4_MELD_GROUP_MAX ||
        type > CJ4_MELD_KAKAN)
      return false;
    meld_category = ((uint16_t)relative_player(player, view->player) *
                         (CJ4_MELD_GROUP_MAX + 1u) +
                     group) *
                        (CJ4_MELD_KAKAN + 1u) +
                    type;
  } else if (location->placement != CJ4_LOCATION_NONE) {
    return false;
  }

  output[CJ4ME_TILE_FEATURE_WALL] =
      normalize_category(wall_category, WALL_NONE_CATEGORY);
  output[CJ4ME_TILE_FEATURE_DISCARD] =
      normalize_category(discard_category, DISCARD_NONE_CATEGORY);
  output[CJ4ME_TILE_FEATURE_DISCARD_HISTORY] =
      normalize_category(history_category, DISCARD_HISTORY_NONE_CATEGORY);
  output[CJ4ME_TILE_FEATURE_HAND_OWNER] =
      normalize_category(hand_category, HAND_NONE_CATEGORY);
  output[CJ4ME_TILE_FEATURE_MELD] =
      normalize_category(meld_category, MELD_NONE_CATEGORY);
  output[CJ4ME_TILE_FEATURE_AKA] = rules->aka_tiles[tile] ? 1.0f : 0.0f;
  output[CJ4ME_TILE_FEATURE_DRAW] = view->draw_tile == tile ? 1.0f : 0.0f;
  output[CJ4ME_TILE_FEATURE_LAST_DISCARD] =
      view->last_discard == tile ? 1.0f : 0.0f;
  output[CJ4ME_TILE_FEATURE_KAN] = view->kan_tile == tile ? 1.0f : 0.0f;
  output[CJ4ME_TILE_FEATURE_ACTION_PRIMARY] =
      action->tile == tile ? 1.0f : 0.0f;
  output[CJ4ME_TILE_FEATURE_ACTION_MEMBER] =
      action_contains_tile(action, tile) ? 1.0f : 0.0f;
  return true;
}

bool cj4me_encode_features(const cj4_player_view *view, const cj4_rules *rules,
                           const cj4_action *action,
                           float out_features[CJ4ME_FEATURE_COUNT]) {
  if (!out_features)
    return false;
  memset(out_features, 0, sizeof(float) * CJ4ME_FEATURE_COUNT);

  if (!view || !rules || !action || !cj4_rules_validate(rules) ||
      view->player >= CJ4_PLAYER_COUNT ||
      view->current_player >= CJ4_PLAYER_COUNT ||
      view->dealer >= CJ4_PLAYER_COUNT || view->round_wind >= CJ4_WIND_COUNT ||
      (unsigned)view->phase > (unsigned)CJ4_PHASE_GAME_END ||
      (unsigned)action->type > (unsigned)CJ4_ACTION_PASS ||
      action->player != view->player || action->tile_count > 4u ||
      !optional_tile_is_valid(view->draw_tile) ||
      !optional_tile_is_valid(view->last_discard) ||
      !optional_tile_is_valid(view->kan_tile) ||
      !optional_tile_is_valid(action->tile)) {
    return false;
  }
  for (uint8_t index = 0; index < action->tile_count; ++index)
    if (!cj4_tile_id_is_valid(action->tiles[index]))
      return false;

  for (uint16_t tile = 0; tile < CJ4_TILE_ID_COUNT; ++tile) {
    float *tile_features = &out_features[CJ4ME_TILE_FEATURE_INDEX(tile, 0)];
    if (!encode_tile(view, rules, action, (cj4_tile_id)tile, tile_features)) {
      memset(out_features, 0, sizeof(float) * CJ4ME_FEATURE_COUNT);
      return false;
    }
  }

  for (cj4_player player = 0; player < CJ4_PLAYER_COUNT; ++player) {
    uint8_t relative = relative_player(player, view->player);
    out_features[CJ4ME_FEATURE_SCORE_OFFSET + relative] =
        (float)view->scores[player] / 100000.0f;
    out_features[CJ4ME_FEATURE_RIICHI_STATE_OFFSET + relative] =
        view->is_riichi[player] ? 1.0f : 0.0f;
  }
  out_features[CJ4ME_FEATURE_PHASE_OFFSET + view->phase] = 1.0f;
  out_features[CJ4ME_FEATURE_CURRENT_PLAYER_OFFSET +
               relative_player(view->current_player, view->player)] = 1.0f;
  out_features[CJ4ME_FEATURE_DEALER_OFFSET +
               relative_player(view->dealer, view->player)] = 1.0f;
  out_features[CJ4ME_FEATURE_ROUND_WIND_OFFSET + view->round_wind] = 1.0f;
  out_features[CJ4ME_FEATURE_HONBA_OFFSET] = (float)view->honba / 16.0f;
  out_features[CJ4ME_FEATURE_RIICHI_STICKS_OFFSET] =
      (float)view->riichi_sticks / 16.0f;
  out_features[CJ4ME_FEATURE_SELF_FLAGS_OFFSET] =
      view->temporary_furiten ? 1.0f : 0.0f;
  out_features[CJ4ME_FEATURE_SELF_FLAGS_OFFSET + 1u] =
      view->riichi_furiten ? 1.0f : 0.0f;
  out_features[CJ4ME_FEATURE_SELF_FLAGS_OFFSET + 2u] =
      view->first_turn_uninterrupted ? 1.0f : 0.0f;

  out_features[CJ4ME_FEATURE_ACTION_TYPE_OFFSET] =
      2.0f * (float)action->type / (float)CJ4_ACTION_PASS - 1.0f;
  out_features[CJ4ME_FEATURE_ACTION_TILE_COUNT_OFFSET] =
      2.0f * (float)action->tile_count / 4.0f - 1.0f;
  return true;
}
