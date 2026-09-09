#include "../tools/battle_policy.h"
#include "cjong4/core/state_init.h"
#include <assert.h>

void test_battle(void) {
  cj4me_rng rng;
  cj4_tile_id wall[CJ4_TILE_ID_COUNT];
  cj4_rules rules = cj4_rules_default();
  cj4me_rng_seed(&rng, 123);
  cj4me_rng_shuffle_wall(&rng, wall);
  cj4_mahjong state = cj4_create_initial_state(wall, &rules);
  cj4_player p = cj4_state_current_player(&state);
  cj4_player_view view = cj4m_make_player_view(&state, p);
  cj4_action actions[CJ4M_MAX_ACTIONS], selected;
  uint8_t count =
      cj4m_collect_actions(&state, &rules, p, actions, CJ4M_MAX_ACTIONS);
  assert(battle_baseline(&rng, false, &view, actions, count, &selected));
  assert(selected.type == CJ4_ACTION_DISCARD);
  cj4_shanten_result chosen;
  assert(cj4p_calculate_shanten_after_discard(&view, selected.tile, &chosen));
  int best = chosen.standard;
  if (chosen.chiitoitsu < best)
    best = chosen.chiitoitsu;
  if (chosen.kokushi < best)
    best = chosen.kokushi;
  for (uint8_t i = 0; i < count; ++i) {
    if (actions[i].type != CJ4_ACTION_DISCARD)
      continue;
    cj4_shanten_result s;
    assert(cj4p_calculate_shanten_after_discard(&view, actions[i].tile, &s));
    assert(best <= s.standard && best <= s.chiitoitsu && best <= s.kokushi);
  }
  actions[0] = (cj4_action){.type = CJ4_ACTION_PON, .player = p};
  actions[1] = (cj4_action){.type = CJ4_ACTION_PASS, .player = p};
  assert(battle_baseline(&rng, false, &view, actions, 2, &selected));
  assert(selected.type == CJ4_ACTION_PASS);
  actions[2] = (cj4_action){.type = CJ4_ACTION_RON, .player = p};
  assert(battle_baseline(&rng, false, &view, actions, 3, &selected));
  assert(selected.type == CJ4_ACTION_RON);
  assert(!battle_baseline(&rng, false, &view, actions, 0, &selected));
}
