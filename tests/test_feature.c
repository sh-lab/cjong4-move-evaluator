#include <assert.h>
#include <string.h>

#include "cjong4/core/location.h"
#include "cjong4/core/tile.h"
#include "cjong4_move_evaluator/feature.h"

_Static_assert(CJ4ME_FEATURE_COUNT == 1369, "feature schema changed");
_Static_assert(CJ4ME_FEATURE_ACTION_TILES_AKA_OFFSET +
                       CJ4ME_FEATURE_ACTION_TILES_AKA_COUNT ==
                   CJ4ME_FEATURE_COUNT,
               "feature regions must be contiguous");

static void set_unknown_locations(cj4_location locations[CJ4_TILE_ID_COUNT]) {
  memset(locations, CJ4_LOCATION_NONE,
         sizeof(cj4_location) * CJ4_TILE_ID_COUNT);
}

void test_feature(void) {
  cj4_player_view view;
  cj4_rules rules = cj4_rules_default();
  cj4_action action;
  float features[CJ4ME_FEATURE_COUNT];
  float normal_features[CJ4ME_FEATURE_COUNT];
  cj4_tile_id hand_tile = cj4_tile_make(2, 3);
  cj4_tile_id hidden_other_tile = cj4_tile_make(3, 1);
  cj4_tile_id discard_tile = cj4_tile_make(5, 2);
  cj4_tile_id red_five = cj4_tile_make(4, 0);

  memset(&view, 0, sizeof(view));
  memset(&action, 0, sizeof(action));
  set_unknown_locations(view.locations);

  view.player = CJ4_PLAYER_1;
  view.current_player = CJ4_PLAYER_2;
  view.dealer = CJ4_PLAYER_0;
  view.round_wind = CJ4_WIND_SOUTH;
  view.phase = CJ4_PHASE_DISCARD;
  view.draw_tile = CJ4_TILE_ID_INVALID;
  view.last_discard = discard_tile;
  view.kan_tile = CJ4_TILE_ID_INVALID;
  view.scores[CJ4_PLAYER_1] = 25000;
  view.is_riichi[CJ4_PLAYER_2] = 1;

  view.locations[hand_tile].placement =
      (uint8_t)(CJ4_PLAYER_1 << CJ4_LOCATION_PLAYER_SHIFT);
  view.locations[hidden_other_tile].placement =
      (uint8_t)(CJ4_PLAYER_2 << CJ4_LOCATION_PLAYER_SHIFT);
  view.locations[discard_tile].discard =
      (uint8_t)((CJ4_PLAYER_2 << CJ4_LOCATION_PLAYER_SHIFT) | 0u |
                CJ4_LOCATION_DISCARD_TSUMOGIRI_FLAG);
  view.locations[discard_tile].discard_history =
      CJ4_LOCATION_DISCARD_RIICHI_FLAG;
  /* A called discard remains part of public discard history. */
  view.locations[discard_tile].placement =
      (uint8_t)(CJ4_LOCATION_PLACEMENT_MELD_FLAG |
                (CJ4_PLAYER_0 << CJ4_LOCATION_PLAYER_SHIFT) | CJ4_MELD_CHI);

  action.type = CJ4_ACTION_CHI;
  action.player = CJ4_PLAYER_2;
  action.tile = discard_tile;
  action.tiles[0] = cj4_tile_make(4, 0);
  action.tiles[1] = discard_tile;
  action.tiles[2] = cj4_tile_make(6, 1);
  action.tile_count = 3;

  assert(cj4me_encode_features(&view, &rules, &action, features));
  assert(features[CJ4ME_FEATURE_HAND_OFFSET + 2] == 0.25f);
  assert(features[CJ4ME_FEATURE_HAND_OFFSET + 3] == 0.0f);
  assert(features[CJ4ME_FEATURE_CURRENT_PLAYER_OFFSET + 1] == 1.0f);
  assert(features[CJ4ME_FEATURE_DEALER_OFFSET + 3] == 1.0f);
  assert(features[CJ4ME_FEATURE_DISCARD_COUNT_OFFSET + 34 + 5] == 0.25f);
  assert(features[CJ4ME_FEATURE_DISCARD_TSUMOGIRI_OFFSET + 34 + 5] == 0.25f);
  assert(features[CJ4ME_FEATURE_DISCARD_RIICHI_OFFSET + 34 + 5] == 0.25f);
  assert(features[CJ4ME_FEATURE_ACTION_TYPE_OFFSET + CJ4_ACTION_CHI] == 1.0f);
  assert(features[CJ4ME_FEATURE_ACTION_PLAYER_OFFSET + 1] == 1.0f);
  assert(features[CJ4ME_FEATURE_ACTION_TILE_OFFSET] == 1.0f);
  assert(features[CJ4ME_FEATURE_ACTION_TILE_OFFSET + 1 + 5] == 1.0f);
  assert(features[CJ4ME_FEATURE_ACTION_TILES_OFFSET + 4] == 0.25f);
  assert(features[CJ4ME_FEATURE_ACTION_TILES_OFFSET + 5] == 0.25f);
  assert(features[CJ4ME_FEATURE_ACTION_TILES_OFFSET + 6] == 0.25f);
  assert(features[CJ4ME_FEATURE_ACTION_TILE_COUNT_OFFSET] == 0.75f);

  action.type = CJ4_ACTION_DISCARD;
  action.tile = red_five;
  action.tile_count = 0u;
  assert(cj4me_encode_features(&view, &rules, &action, features));
  action.tile = cj4_tile_make(4u, 1u);
  assert(cj4me_encode_features(&view, &rules, &action, normal_features));
  assert(features[CJ4ME_FEATURE_ACTION_TILE_AKA_OFFSET] == 1.0f);
  assert(normal_features[CJ4ME_FEATURE_ACTION_TILE_AKA_OFFSET] == 0.0f);

  action.tile_count = 5;
  assert(!cj4me_encode_features(&view, &rules, &action, features));
  action.tile_count = 3;
  view.phase = (cj4_phase)-1;
  assert(!cj4me_encode_features(&view, &rules, &action, features));
}
