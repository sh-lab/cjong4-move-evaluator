#include "../tools/battle_diagnostics.h"
#include "../tools/battle_policy.h"
#include "cjong4/core/state_init.h"
#include <assert.h>
#include <string.h>

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
  battle_diagnostics diagnostics = {0};
  assert(battle_diagnostics_record(&diagnostics, &view, actions, count,
                                   &selected));
  assert(diagnostics.discard_decisions == 1);
  assert(diagnostics.discard_choices == 1);
  assert(diagnostics.gap_sum == 0 && diagnostics.worse_choices == 0);
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
  assert(battle_diagnostics_record(&diagnostics, &view, actions, 2, &selected));
  assert(diagnostics.call_opportunities == 1 && diagnostics.call_pass == 1);
  assert(
      battle_diagnostics_record(&diagnostics, &view, actions, 2, &actions[0]));
  assert(diagnostics.call_selected == 1);
  actions[2] = (cj4_action){.type = CJ4_ACTION_RON, .player = p};
  assert(battle_baseline(&rng, false, &view, actions, 3, &selected));
  assert(selected.type == CJ4_ACTION_RON);
  assert(battle_diagnostics_record(&diagnostics, &view, actions, 3, &selected));
  assert(diagnostics.call_opportunities == 2);
  assert(diagnostics.actions[CJ4_ACTION_RON] == 1);
  uint64_t decisions = diagnostics.decisions;
  assert(
      !battle_diagnostics_record(&diagnostics, &view, actions, 0, &selected));
  assert(diagnostics.decisions == decisions);
  assert(!battle_baseline(&rng, false, &view, actions, 0, &selected));

  /* 123m 123p 123s 78s EE W: discard W keeps tenpai; E breaks it. */
  memset(&view, 0, sizeof(view));
  memset(view.locations, 0xff, sizeof(view.locations));
  const cj4_tile_id hand[] = {0,  4,  8,  36,  40,  44,  72,
                              76, 80, 96, 100, 108, 109, 116};
  for (unsigned i = 0; i < sizeof(hand) / sizeof(hand[0]); ++i)
    view.locations[hand[i]].placement = 0; /* player 0 concealed hand */
  actions[0] = (cj4_action){.type = CJ4_ACTION_DISCARD, .tile = 116};
  actions[1] = (cj4_action){.type = CJ4_ACTION_DISCARD, .tile = 108};
  diagnostics = (battle_diagnostics){0};
  assert(
      battle_diagnostics_record(&diagnostics, &view, actions, 2, &actions[1]));
  assert(diagnostics.best_sum == 0 && diagnostics.after_sum == 1);
  assert(diagnostics.worse_choices == 1 && diagnostics.gap_sum == 1);
  assert(diagnostics.tenpai_available == 1 && diagnostics.tenpai_kept == 0);
  assert(
      battle_diagnostics_record(&diagnostics, &view, actions, 2, &actions[0]));
  assert(diagnostics.tenpai_kept == 1);

  /* Riichi/discard on the same physical tile is one discard candidate. */
  actions[1] = actions[0];
  actions[1].type = CJ4_ACTION_RIICHI;
  diagnostics = (battle_diagnostics){0};
  assert(
      battle_diagnostics_record(&diagnostics, &view, actions, 2, &actions[1]));
  assert(diagnostics.discard_choices == 0 &&
         diagnostics.discard_decisions == 1);
  assert(diagnostics.riichi_opportunities == 1 &&
         diagnostics.riichi_selected == 1);
}
