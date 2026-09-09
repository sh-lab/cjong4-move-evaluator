#include <errno.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "battle_diagnostics.h"
#include "battle_policy.h"
#include "cjong4/core/state_init.h"
#include "cjong4/core/state_query.h"
#include "cjong4/core/state_round.h"
#include "cjong4_move_evaluator/evaluator.h"
#include "tool_model.h"

typedef struct {
  cj4me_rng rng;
  cj4me_evaluator_context *model;
  bool random;
  bool failed;
  battle_diagnostics *diagnostics;
} player_context;

static cj4_action decide(void *opaque, const cj4_player_view *view,
                         const cj4_action *actions, uint8_t count) {
  player_context *p = opaque;
  cj4_action result = {0};
  if (p->model)
    result = cj4me_evaluator_decide(p->model, view, actions, count);
  else if (!battle_baseline(&p->rng, p->random, view, actions, count, &result))
    p->failed = true;
  if (!p->failed && !(p->model && p->model->failed) && p->diagnostics &&
      !battle_diagnostics_record(p->diagnostics, view, actions, count, &result))
    p->failed = true;
  return result;
}

static bool parse_number(const char *s, uint64_t *out) {
  char *end;
  if (!s[0] || s[0] == '-')
    return false;
  errno = 0;
  uintmax_t n = strtoumax(s, &end, 10);
  if (errno || *end || n > UINT64_MAX)
    return false;
  *out = (uint64_t)n;
  return true;
}

int main(int argc, char **argv) {
  const char *path = NULL;
  bool random = false, baseline_only = false;
  bool diagnostics_enabled = false;
  battle_diagnostics diagnostics = {0};
  uint64_t seeds = 10, seed = 1;
  cj4_rules rules = cj4_rules_default();
  cj4me_model_kind kind;
  void *model = NULL;
  cj4me_evaluator_context *eval = NULL;
  uint64_t games = 0, rounds = 0, wins = 0, dealt_in = 0;
  double score_sum = 0, rank_sum = 0;
  int rc = 1;
  for (int i = 1; i < argc; ++i) {
    if (!strcmp(argv[i], "--diagnostics")) {
      diagnostics_enabled = true;
      continue;
    }
    if (!strcmp(argv[i], "--baseline-only")) {
      baseline_only = true;
      continue;
    }
    if (i + 1 >= argc)
      goto usage;
    if (!strcmp(argv[i], "--model"))
      path = argv[++i];
    else if (!strcmp(argv[i], "--seeds")) {
      if (!parse_number(argv[++i], &seeds) || !seeds || seeds > UINT32_MAX)
        goto usage;
    } else if (!strcmp(argv[i], "--seed")) {
      if (!parse_number(argv[++i], &seed))
        goto usage;
    } else if (!strcmp(argv[i], "--opponent")) {
      ++i;
      if (!strcmp(argv[i], "random"))
        random = true;
      else if (!strcmp(argv[i], "shanten"))
        random = false;
      else
        goto usage;
    } else
      goto usage;
  }
  if ((!path && !baseline_only) || (path && baseline_only) ||
      seed > UINT64_MAX - (seeds - 1))
    goto usage;
  if (path) {
    if (!cj4me_tool_detect_model_kind(path, &kind))
      goto fail;
    size_t size = kind == CJ4ME_MODEL_KIND_F32 ? sizeof(cj4me_model_f32)
                                               : sizeof(cj4me_model_i8);
    model = malloc(size);
    eval = malloc(sizeof(*eval));
    if (!model || !eval)
      goto fail;
    if (kind == CJ4ME_MODEL_KIND_F32) {
      if (!cj4me_model_f32_load_file(model, path))
        goto fail;
    } else if (!cj4me_model_i8_load_file(model, path))
      goto fail;
    if (!cj4me_evaluator_context_init(eval, kind, model, &rules))
      goto fail;
  }
  puts("seed\tseat\tscore_delta\trank\trounds\twins\tdealt_in");
  for (uint64_t k = 0; k < seeds; ++k) {
    for (cj4_player seat = 0; seat < CJ4_PLAYER_COUNT; ++seat) {
      cj4me_rng wall_rng;
      cj4_tile_id wall[CJ4_TILE_ID_COUNT];
      cj4m_player_delegate delegates[CJ4_PLAYER_COUNT];
      player_context players[CJ4_PLAYER_COUNT] = {0};
      uint64_t nr = 0, nw = 0, nd = 0;
      uint32_t steps = 0;
      cj4me_rng_seed(&wall_rng, seed + k);
      cj4me_rng_shuffle_wall(&wall_rng, wall);
      cj4_mahjong state = cj4_create_initial_state(wall, &rules);
      int32_t initial = state.scores[seat];
      for (cj4_player p = 0; p < CJ4_PLAYER_COUNT; ++p) {
        cj4me_rng_seed(&players[p].rng,
                       (seed + k) ^ (UINT64_C(0x9e3779b97f4a7c15) * (p + 1)));
        players[p].random = p != seat && random;
        players[p].model = p == seat ? eval : NULL;
        players[p].diagnostics =
            diagnostics_enabled && p == seat ? &diagnostics : NULL;
        delegates[p] =
            (cj4m_player_delegate){.ctx = &players[p], .decide = decide};
      }
      while (cj4_state_phase(&state) != CJ4_PHASE_GAME_END) {
        cj4_mahjong next;
        if (++steps > 100000u)
          goto fail;
        if (cj4_state_phase(&state) == CJ4_PHASE_SETTLE) {
          if (cj4_can_game_end(state))
            next = cj4_do_game_end(state);
          else {
            if (!cj4_can_next_round(state))
              goto fail;
            cj4me_rng_shuffle_wall(&wall_rng, wall);
            next = cj4_do_next_round(state, wall, &rules);
          }
        } else {
          if (cj4_state_phase(&state) == CJ4_PHASE_ROUND_END) {
            ++nr;
            if (cj4_state_is_winner(&state, seat))
              ++nw;
            if (cj4_state_round_end_type(&state) == CJ4_ROUND_END_RON &&
                cj4_state_current_player(&state) == seat)
              ++nd;
          }
          next = cj4m_step(&state, &rules, delegates);
        }
        if (eval && eval->failed)
          goto fail;
        for (cj4_player p = 0; p < CJ4_PLAYER_COUNT; ++p)
          if (players[p].failed)
            goto fail;
        if (!memcmp(&next, &state, sizeof(state)))
          goto fail;
        state = next;
      }
      double rank = 1;
      for (cj4_player p = 0; p < CJ4_PLAYER_COUNT; ++p) {
        if (p == seat)
          continue;
        if (state.scores[p] > state.scores[seat])
          rank += 1;
        else if (state.scores[p] == state.scores[seat])
          rank += 0.5;
      }
      int64_t delta = (int64_t)state.scores[seat] - initial;
      printf("%" PRIu64 "\t%u\t%" PRId64 "\t%.1f\t%" PRIu64 "\t%" PRIu64
             "\t%" PRIu64 "\n",
             seed + k, (unsigned)seat, delta, rank, nr, nw, nd);
      fflush(stdout);
      ++games;
      rounds += nr;
      wins += nw;
      dealt_in += nd;
      score_sum += (double)delta;
      rank_sum += rank;
    }
  }
  fprintf(stderr,
          "games=%" PRIu64 " rounds=%" PRIu64
          " mean_score_delta=%.2f mean_rank=%.3f win_rate=%.2f%% "
          "deal_in_rate=%.2f%%\n",
          games, rounds, score_sum / games, rank_sum / games,
          rounds ? 100.0 * wins / rounds : 0,
          rounds ? 100.0 * dealt_in / rounds : 0);
  rc = 0;
  if (diagnostics_enabled)
    battle_diagnostics_print(&diagnostics);
  goto done;
usage:
  fprintf(
      stderr,
      "Usage: cj4me_battle (--model PATH | --baseline-only) "
      "[--opponent shanten|random] [--seeds N] [--seed N] [--diagnostics]\n");
  rc = 2;
  goto done;
fail:
  fprintf(stderr, "cj4me_battle: model load, policy or state transition "
                  "failed; results are incomplete\n");
done:
  free(eval);
  free(model);
  return rc;
}
