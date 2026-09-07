#include <assert.h>
#include <math.h>
#include <string.h>

#include "cjong4/core/location.h"
#include "cjong4/core/tile_const.h"
#include "cjong4/player/hand_analysis.h"
#include "cjong4_move_evaluator/policy_filter.h"

static cj4_player_view empty_view(void) {
  cj4_player_view view;
  memset(&view, 0, sizeof(view));
  memset(view.locations, CJ4_LOCATION_NONE, sizeof(view.locations));
  view.draw_tile = CJ4_TILE_ID_INVALID;
  view.last_discard = CJ4_TILE_ID_INVALID;
  view.kan_tile = CJ4_TILE_ID_INVALID;
  return view;
}

static void put_in_hand(cj4_player_view *view, uint8_t type, uint8_t copy) {
  cj4_tile_id tile = cj4_tile_make((cj4_tile_type)type, copy);
  view->locations[tile].placement =
      (uint8_t)(view->player << CJ4_LOCATION_PLAYER_SHIFT);
}

static void put_discard(cj4_player_view *view, cj4_player player, uint8_t type,
                        uint8_t copy, uint8_t player_index,
                        uint8_t history_index, bool tsumogiri) {
  cj4_tile_id tile = cj4_tile_make((cj4_tile_type)type, copy);
  view->locations[tile].discard =
      (uint8_t)((player << CJ4_LOCATION_PLAYER_SHIFT) | player_index |
                (tsumogiri ? CJ4_LOCATION_DISCARD_TSUMOGIRI_FLAG : 0u));
  view->locations[tile].discard_history = history_index;
}

static void put_meld(cj4_player_view *view, cj4_player player, uint8_t type,
                     uint8_t copy, uint8_t group, cj4_meld_type meld_type) {
  cj4_tile_id tile = cj4_tile_make((cj4_tile_type)type, copy);
  view->locations[tile].placement =
      (uint8_t)(CJ4_LOCATION_PLACEMENT_MELD_FLAG |
                (player << CJ4_LOCATION_PLAYER_SHIFT) |
                (group << CJ4_LOCATION_MELD_GROUP_SHIFT) | meld_type);
}

static int shanten_minimum(const cj4_shanten_result *result) {
  int minimum = result->standard;
  if (result->chiitoitsu < minimum)
    minimum = result->chiitoitsu;
  if (result->kokushi < minimum)
    minimum = result->kokushi;
  return minimum;
}

static void test_profile_filters(void) {
  cj4_player_view view = empty_view();
  cj4_action actions[7];
  cj4me_policy_filter_config config;
  cj4me_policy_filter_result result;

  memset(actions, 0, sizeof(actions));
  actions[0].type = CJ4_ACTION_PASS;
  actions[1].type = CJ4_ACTION_CHI;
  actions[2].type = CJ4_ACTION_PON;
  actions[3].type = CJ4_ACTION_MINKAN;
  actions[4].type = CJ4_ACTION_ANKAN;
  actions[5].type = CJ4_ACTION_RIICHI;
  actions[6].type = CJ4_ACTION_RON;
  for (uint8_t i = 0; i < 7u; ++i)
    actions[i].player = view.player;

  config = cj4me_policy_filter_default();
  assert(cj4me_policy_filter_apply(&config, &view, actions, 6u, &result));
  assert(result.allowed_count == 6u);
  assert(!result.used_fallback);

  assert(cj4me_policy_filter_preset(CJ4ME_POLICY_MENZEN,
                                    CJ4ME_POLICY_FILTER_HARD, 0.0f, &config));
  assert(cj4me_policy_filter_apply(&config, &view, actions, 6u, &result));
  assert(result.allowed_count == 3u);
  assert(result.allowed[0] && result.allowed[4] && result.allowed[5]);

  assert(cj4me_policy_filter_preset(CJ4ME_POLICY_CALL, CJ4ME_POLICY_FILTER_HARD,
                                    0.0f, &config));
  assert(cj4me_policy_filter_apply(&config, &view, actions, 6u, &result));
  assert(result.allowed_count == 3u);
  assert(result.allowed[1] && result.allowed[2] && result.allowed[3]);

  assert(cj4me_policy_filter_preset(CJ4ME_POLICY_RIICHI,
                                    CJ4ME_POLICY_FILTER_HARD, 0.0f, &config));
  assert(cj4me_policy_filter_apply(&config, &view, actions, 6u, &result));
  assert(result.allowed_count == 1u && result.allowed[5]);

  assert(cj4me_policy_filter_preset(CJ4ME_POLICY_DAMA, CJ4ME_POLICY_FILTER_HARD,
                                    0.0f, &config));
  assert(cj4me_policy_filter_apply(&config, &view, actions, 6u, &result));
  assert(result.allowed_count == 5u && !result.allowed[5]);

  assert(cj4me_policy_filter_apply(&config, &view, actions, 7u, &result));
  assert(result.allowed_count == 1u && result.allowed[6]);

  config = cj4me_policy_filter_default();
  config.exclude_riichi = true;
  assert(cj4me_policy_filter_apply(&config, &view, &actions[5], 1u, &result));
  assert(result.allowed_count == 1u && result.allowed[0]);
  assert(result.used_fallback);

  assert(cj4me_policy_filter_preset(CJ4ME_POLICY_CALL, CJ4ME_POLICY_FILTER_SOFT,
                                    0.25f, &config));
  assert(cj4me_policy_filter_apply(&config, &view, actions, 6u, &result));
  assert(result.allowed_count == 6u);
  assert(fabsf(result.score_bias[0]) < 0.000001f);
  assert(fabsf(result.score_bias[1] - 0.25f) < 0.000001f);
}

static void test_best_shanten_discard_filter(void) {
  cj4_player_view view = empty_view();
  const uint8_t hand[][2] = {{0, 0},  {1, 0},  {2, 0},  {9, 0},  {10, 0},
                             {11, 0}, {18, 0}, {19, 0}, {20, 0}, {3, 0},
                             {4, 0},  {15, 0}, {15, 1}, {33, 0}};
  cj4_action actions[2];
  cj4_shanten_result good;
  cj4_shanten_result bad;
  cj4me_policy_filter_config config;
  cj4me_policy_filter_result result;

  for (size_t i = 0; i < sizeof(hand) / sizeof(hand[0]); ++i)
    put_in_hand(&view, hand[i][0], hand[i][1]);
  memset(actions, 0, sizeof(actions));
  actions[0].type = CJ4_ACTION_DISCARD;
  actions[0].player = view.player;
  actions[0].tile = cj4_tile_make((cj4_tile_type)33, 0);
  actions[1].type = CJ4_ACTION_DISCARD;
  actions[1].player = view.player;
  actions[1].tile = cj4_tile_make((cj4_tile_type)0, 0);

  assert(cj4p_calculate_shanten_after_discard(&view, actions[0].tile, &good));
  assert(cj4p_calculate_shanten_after_discard(&view, actions[1].tile, &bad));
  assert(shanten_minimum(&good) < shanten_minimum(&bad));
  assert(cj4me_policy_filter_preset(CJ4ME_POLICY_SPEED,
                                    CJ4ME_POLICY_FILTER_HARD, 0.0f, &config));
  assert(cj4me_policy_filter_apply(&config, &view, actions, 2u, &result));
  assert(result.allowed_count == 1u);
  assert(result.allowed[0] && !result.allowed[1]);
}

static void assert_safe_filter(cj4_player_view *view, cj4_player threat,
                               cj4_tile_id safe, cj4_tile_id unsafe) {
  cj4_action actions[2] = {{0}};
  cj4me_policy_filter_config config;
  cj4me_policy_filter_result result;

  actions[0].type = CJ4_ACTION_DISCARD;
  actions[0].player = view->player;
  actions[0].tile = safe;
  actions[1].type = CJ4_ACTION_DISCARD;
  actions[1].player = view->player;
  actions[1].tile = unsafe;
  assert(cj4me_policy_filter_preset(CJ4ME_POLICY_SAFE, CJ4ME_POLICY_FILTER_HARD,
                                    0.0f, &config));
  assert(config.safe_live_wall_threshold == 40u);
  assert(config.safe_tsumogiri_streak == 3u);
  assert(cj4me_policy_filter_apply(&config, view, actions, 2u, &result));
  assert(result.threat_mask == (uint8_t)(1u << threat));
  assert(result.allowed_count == 1u);
  assert(result.allowed[0] && !result.allowed[1]);

  assert(cj4me_policy_filter_preset(CJ4ME_POLICY_SAFE, CJ4ME_POLICY_FILTER_SOFT,
                                    0.25f, &config));
  assert(cj4me_policy_filter_apply(&config, view, actions, 2u, &result));
  assert(result.allowed_count == 2u);
  assert(fabsf(result.score_bias[0] - 0.25f) < 0.000001f);
  assert(fabsf(result.score_bias[1]) < 0.000001f);
}

static void test_safe_filter(void) {
  cj4_player_view view = empty_view();
  cj4_tile_id safe = cj4_tile_make(CJ4_TILE_TYPE_5M, 0);
  cj4_tile_id unsafe = cj4_tile_make(CJ4_TILE_TYPE_6M, 0);

  view.is_riichi[CJ4_PLAYER_1] = 1u;
  put_discard(&view, CJ4_PLAYER_1, CJ4_TILE_TYPE_5M, 1u, 0u, 0u, false);
  assert_safe_filter(&view, CJ4_PLAYER_1, safe, unsafe);

  view = empty_view();
  for (uint8_t copy = 0; copy < 3u; ++copy)
    put_meld(&view, CJ4_PLAYER_2, CJ4_TILE_TYPE_HAKU, copy, 0u, CJ4_MELD_PON);
  put_discard(&view, CJ4_PLAYER_2, CJ4_TILE_TYPE_5M, 1u, 0u, 0u, false);
  assert_safe_filter(&view, CJ4_PLAYER_2, safe, unsafe);

  view = empty_view();
  view.live_wall_remaining = 40u;
  put_discard(&view, CJ4_PLAYER_3, CJ4_TILE_TYPE_5M, 1u, 0u, 0u, true);
  put_discard(&view, CJ4_PLAYER_3, CJ4_TILE_TYPE_7M, 0u, 1u, 1u, true);
  put_discard(&view, CJ4_PLAYER_3, CJ4_TILE_TYPE_8M, 0u, 2u, 2u, true);
  assert_safe_filter(&view, CJ4_PLAYER_3, safe, unsafe);
}

static void test_kokushi_filter(void) {
  const uint8_t yaochu[] = {
      CJ4_TILE_TYPE_1M,    CJ4_TILE_TYPE_9M,    CJ4_TILE_TYPE_1P,
      CJ4_TILE_TYPE_9P,    CJ4_TILE_TYPE_1S,    CJ4_TILE_TYPE_9S,
      CJ4_TILE_TYPE_EAST,  CJ4_TILE_TYPE_SOUTH, CJ4_TILE_TYPE_WEST,
      CJ4_TILE_TYPE_NORTH, CJ4_TILE_TYPE_HAKU,  CJ4_TILE_TYPE_HATSU,
      CJ4_TILE_TYPE_CHUN};
  cj4_player_view view = empty_view();
  cj4_action actions[2] = {{0}};
  cj4_shanten_result good;
  cj4_shanten_result bad;
  cj4me_policy_filter_config config;
  cj4me_policy_filter_result result;

  for (size_t i = 0; i < sizeof(yaochu) / sizeof(yaochu[0]) - 1u; ++i)
    put_in_hand(&view, yaochu[i], 0u);
  put_in_hand(&view, CJ4_TILE_TYPE_1M, 1u);
  put_in_hand(&view, CJ4_TILE_TYPE_5M, 0u);
  actions[0].type = CJ4_ACTION_DISCARD;
  actions[0].player = view.player;
  actions[0].tile = cj4_tile_make(CJ4_TILE_TYPE_5M, 0u);
  actions[1].type = CJ4_ACTION_DISCARD;
  actions[1].player = view.player;
  actions[1].tile = cj4_tile_make(CJ4_TILE_TYPE_9M, 0u);

  assert(cj4p_calculate_shanten_after_discard(&view, actions[0].tile, &good));
  assert(cj4p_calculate_shanten_after_discard(&view, actions[1].tile, &bad));
  assert(good.kokushi < bad.kokushi);
  assert(cj4me_policy_filter_preset(CJ4ME_POLICY_KOKUSHI,
                                    CJ4ME_POLICY_FILTER_HARD, 0.0f, &config));
  assert(config.kokushi_min_unique_yaochu == 9u);
  assert(config.kokushi_max_shanten_gap == 1u);
  assert(cj4me_policy_filter_apply(&config, &view, actions, 2u, &result));
  assert(result.kokushi_active);
  assert(result.allowed_count == 1u);
  assert(result.allowed[0] && !result.allowed[1]);

  assert(cj4me_policy_filter_preset(CJ4ME_POLICY_KOKUSHI,
                                    CJ4ME_POLICY_FILTER_SOFT, 0.25f, &config));
  assert(cj4me_policy_filter_apply(&config, &view, actions, 2u, &result));
  assert(result.kokushi_active);
  assert(result.allowed_count == 2u);
  assert(fabsf(result.score_bias[0] - 0.25f) < 0.000001f);
  assert(fabsf(result.score_bias[1]) < 0.000001f);

  view.locations[cj4_tile_make(CJ4_TILE_TYPE_5M, 0u)].placement =
      CJ4_LOCATION_NONE;
  actions[0].type = CJ4_ACTION_PASS;
  actions[0].tile = CJ4_TILE_ID_INVALID;
  actions[1].type = CJ4_ACTION_PON;
  actions[1].tile = cj4_tile_make(CJ4_TILE_TYPE_CHUN, 0u);
  assert(cj4me_policy_filter_preset(CJ4ME_POLICY_KOKUSHI,
                                    CJ4ME_POLICY_FILTER_HARD, 0.0f, &config));
  assert(cj4me_policy_filter_apply(&config, &view, actions, 2u, &result));
  assert(result.kokushi_active);
  assert(result.allowed_count == 1u);
  assert(result.allowed[0] && !result.allowed[1]);

  view = empty_view();
  for (size_t i = 0; i < 8u; ++i)
    put_in_hand(&view, yaochu[i], 0u);
  assert(cj4me_policy_filter_apply(&config, &view, actions, 2u, &result));
  assert(!result.kokushi_active);
  assert(result.allowed_count == 2u);
}

void test_policy_filter(void) {
  test_profile_filters();
  test_best_shanten_discard_filter();
  test_safe_filter();
  test_kokushi_filter();
}
