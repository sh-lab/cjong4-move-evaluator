#include <assert.h>
#include <math.h>
#include <string.h>

#include "cjong4/core/location.h"
#include "cjong4/core/tile.h"
#include "cjong4_move_evaluator/feature.h"

_Static_assert(CJ4ME_TILE_FEATURE_COUNT == 13, "tile schema changed");
_Static_assert(CJ4ME_FEATURE_STATE_COUNT == 33, "state schema changed");
_Static_assert(CJ4ME_FEATURE_ACTION_COUNT == 2, "action schema changed");
_Static_assert(CJ4ME_FEATURE_COUNT == 1803, "feature schema changed");
_Static_assert(CJ4ME_SCORE_INPUT_COUNT == 1123, "score input changed");

static float normalized(unsigned category, unsigned maximum) {
  return 2.0f * (float)category / (float)maximum - 1.0f;
}

static void set_unknown_locations(cj4_location locations[CJ4_TILE_ID_COUNT]) {
  memset(locations, CJ4_LOCATION_NONE,
         sizeof(cj4_location) * CJ4_TILE_ID_COUNT);
}

void test_feature(void) {
  cj4_player_view view;
  cj4_rules rules = cj4_rules_default();
  cj4_action action;
  float features[CJ4ME_FEATURE_COUNT];
  cj4_tile_id hand_tile = cj4_tile_make(2, 3);
  cj4_tile_id discard_tile = cj4_tile_make(5, 2);
  cj4_tile_id action_member = cj4_tile_make(4, 0);
  size_t hand = CJ4ME_TILE_FEATURE_INDEX(hand_tile, 0);
  size_t discard = CJ4ME_TILE_FEATURE_INDEX(discard_tile, 0);
  size_t member = CJ4ME_TILE_FEATURE_INDEX(action_member, 0);

  memset(&view, 0, sizeof(view));
  memset(&action, 0, sizeof(action));
  set_unknown_locations(view.locations);

  view.player = CJ4_PLAYER_1;
  view.current_player = CJ4_PLAYER_2;
  view.dealer = CJ4_PLAYER_0;
  view.round_wind = CJ4_WIND_SOUTH;
  view.phase = CJ4_PHASE_DISCARD;
  view.draw_tile = hand_tile;
  view.last_discard = discard_tile;
  view.kan_tile = CJ4_TILE_ID_INVALID;
  view.scores[CJ4_PLAYER_1] = 25000;
  view.is_riichi[CJ4_PLAYER_2] = 1;

  view.locations[hand_tile].wall = 12u;
  view.locations[hand_tile].placement =
      (uint8_t)(CJ4_PLAYER_1 << CJ4_LOCATION_PLAYER_SHIFT);
  view.locations[discard_tile].discard =
      (uint8_t)((CJ4_PLAYER_2 << CJ4_LOCATION_PLAYER_SHIFT) | 3u |
                CJ4_LOCATION_DISCARD_TSUMOGIRI_FLAG);
  view.locations[discard_tile].discard_history =
      (uint8_t)(17u | CJ4_LOCATION_DISCARD_RIICHI_FLAG);
  view.locations[discard_tile].placement =
      (uint8_t)(CJ4_LOCATION_PLACEMENT_MELD_FLAG |
                (CJ4_PLAYER_0 << CJ4_LOCATION_PLAYER_SHIFT) |
                (2u << CJ4_LOCATION_MELD_GROUP_SHIFT) | CJ4_MELD_CHI);

  action.type = CJ4_ACTION_CHI;
  action.player = CJ4_PLAYER_1;
  action.tile = discard_tile;
  action.tiles[0] = action_member;
  action.tiles[1] = hand_tile;
  action.tile_count = 2;

  assert(cj4me_encode_features(&view, &rules, &action, features));
  assert(fabsf(features[hand + CJ4ME_TILE_FEATURE_WALL] -
               normalized(12u, 136u)) < 1.0e-6f);
  assert(features[hand + CJ4ME_TILE_FEATURE_HAND_OWNER] == -1.0f);
  assert(features[hand + CJ4ME_TILE_FEATURE_DRAW] == 1.0f);
  assert(features[hand + CJ4ME_TILE_FEATURE_ACTION_MEMBER] == 1.0f);
  assert(features[discard + CJ4ME_TILE_FEATURE_DISCARD] ==
         normalized(34u, 124u));
  assert(features[discard + CJ4ME_TILE_FEATURE_TSUMOGIRI] == 1.0f);
  assert(features[discard + CJ4ME_TILE_FEATURE_DISCARD_HISTORY] ==
         normalized(17u, 86u));
  assert(features[discard + CJ4ME_TILE_FEATURE_RIICHI_DISCARD] == 1.0f);
  assert(features[discard + CJ4ME_TILE_FEATURE_HAND_OWNER] == 1.0f);
  assert(features[discard + CJ4ME_TILE_FEATURE_MELD] == normalized(70u, 80u));
  assert(features[discard + CJ4ME_TILE_FEATURE_LAST_DISCARD] == 1.0f);
  assert(features[discard + CJ4ME_TILE_FEATURE_ACTION_PRIMARY] == 1.0f);
  assert(features[member + CJ4ME_TILE_FEATURE_AKA] == 1.0f);
  assert(features[member + CJ4ME_TILE_FEATURE_ACTION_MEMBER] == 1.0f);
  assert(features[CJ4ME_FEATURE_SCORE_OFFSET] == 0.25f);
  assert(features[CJ4ME_FEATURE_CURRENT_PLAYER_OFFSET + 1u] == 1.0f);
  assert(features[CJ4ME_FEATURE_DEALER_OFFSET + 3u] == 1.0f);
  assert(features[CJ4ME_FEATURE_PHASE_OFFSET + CJ4_PHASE_DISCARD] == 1.0f);
  assert(features[CJ4ME_FEATURE_ROUND_WIND_OFFSET + CJ4_WIND_SOUTH] == 1.0f);
  assert(features[CJ4ME_FEATURE_RIICHI_STATE_OFFSET + 1u] == 1.0f);
  assert(fabsf(features[CJ4ME_FEATURE_ACTION_TYPE_OFFSET] + 0.8f) < 1.0e-6f);
  assert(features[CJ4ME_FEATURE_ACTION_TILE_COUNT_OFFSET] == 0.0f);

  action.player = CJ4_PLAYER_2;
  assert(!cj4me_encode_features(&view, &rules, &action, features));
  action.player = CJ4_PLAYER_1;
  action.tile_count = 5;
  assert(!cj4me_encode_features(&view, &rules, &action, features));
  action.tile_count = 2;
  view.locations[hand_tile].wall = 200u;
  assert(!cj4me_encode_features(&view, &rules, &action, features));
}
