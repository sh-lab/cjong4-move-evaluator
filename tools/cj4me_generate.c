#include <errno.h>
#include <inttypes.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "cjong4_move_evaluator/evaluator.h"
#include "selfplay.h"
#include "tool_model.h"

typedef struct {
  cj4me_selfplay_config selfplay;
  const char *model_path;
} cli_config;

typedef struct {
  cj4me_model_kind kind;
  union {
    cj4me_model_f32 *f32;
    cj4me_model_i8 *i8;
  } model;
  union {
    cj4me_inference_f32_scratch f32;
    cj4me_inference_i8_scratch i8;
  } scratch;
} model_policy;

static bool score_actions(void *opaque, const cj4_player_view *view,
                          const cj4_rules *rules, const cj4_action *actions,
                          uint8_t action_count, uint8_t *out_index) {
  model_policy *policy = (model_policy *)opaque;
  if (policy->kind == CJ4ME_MODEL_KIND_F32) {
    float score;
    return cj4me_select_action_f32(policy->model.f32, view, rules, actions,
                                   action_count, &policy->scratch.f32,
                                   out_index, &score);
  }
  if (policy->kind == CJ4ME_MODEL_KIND_I8) {
    cj4me_i8_output score;
    return cj4me_select_action_i8(policy->model.i8, view, rules, actions,
                                  action_count, &policy->scratch.i8, out_index,
                                  &score);
  }
  return false;
}

static int load_model(const char *path, model_policy *policy) {
  memset(policy, 0, sizeof(*policy));
  if (!cj4me_tool_detect_model_kind(path, &policy->kind))
    return 0;
  if (policy->kind == CJ4ME_MODEL_KIND_F32) {
    policy->model.f32 = (cj4me_model_f32 *)malloc(sizeof(*policy->model.f32));
    return policy->model.f32 &&
           cj4me_model_f32_load_file(policy->model.f32, path);
  }
  policy->model.i8 = (cj4me_model_i8 *)malloc(sizeof(*policy->model.i8));
  return policy->model.i8 && cj4me_model_i8_load_file(policy->model.i8, path);
}

static void free_model(model_policy *policy) {
  if (policy->kind == CJ4ME_MODEL_KIND_F32)
    free(policy->model.f32);
  else if (policy->kind == CJ4ME_MODEL_KIND_I8)
    free(policy->model.i8);
  memset(policy, 0, sizeof(*policy));
}

static void print_usage(const char *program) {
  fprintf(stderr,
          "Usage: %s --games N --seed N --epsilon F --reward-scale F "
          "--output PATH [--model PATH] [--max-steps N] "
          "[--max-records-per-round N]\n",
          program);
}

static int parse_u32(const char *text, uint32_t *value) {
  char *end;
  unsigned long parsed;
  errno = 0;
  parsed = strtoul(text, &end, 10);
  if (errno != 0 || end == text || *end != '\0' || parsed > UINT32_MAX)
    return 0;
  *value = (uint32_t)parsed;
  return 1;
}

static int parse_u64(const char *text, uint64_t *value) {
  char *end;
  uintmax_t parsed;
  errno = 0;
  parsed = strtoumax(text, &end, 10);
  if (errno != 0 || end == text || *end != '\0' || parsed > UINT64_MAX)
    return 0;
  *value = (uint64_t)parsed;
  return 1;
}

static int parse_float(const char *text, float *value) {
  char *end;
  float parsed;
  errno = 0;
  parsed = strtof(text, &end);
  if (errno != 0 || end == text || *end != '\0' || !isfinite(parsed))
    return 0;
  *value = parsed;
  return 1;
}

static int parse_arguments(int argc, char **argv, cli_config *config) {
  memset(config, 0, sizeof(*config));
  config->selfplay.games = 1u;
  config->selfplay.seed = 1u;
  config->selfplay.epsilon = 1.0f;
  config->selfplay.reward_scale = 8000.0f;
  config->selfplay.max_steps_per_game = 10000u;
  config->selfplay.max_records_per_round = 4096u;

  for (int i = 1; i < argc; ++i) {
    if (strcmp(argv[i], "--help") == 0)
      return 0;
    if (i + 1 >= argc)
      return -1;
    if (strcmp(argv[i], "--games") == 0) {
      if (!parse_u32(argv[++i], &config->selfplay.games))
        return -1;
    } else if (strcmp(argv[i], "--seed") == 0) {
      if (!parse_u64(argv[++i], &config->selfplay.seed))
        return -1;
    } else if (strcmp(argv[i], "--epsilon") == 0) {
      if (!parse_float(argv[++i], &config->selfplay.epsilon))
        return -1;
    } else if (strcmp(argv[i], "--reward-scale") == 0) {
      if (!parse_float(argv[++i], &config->selfplay.reward_scale))
        return -1;
    } else if (strcmp(argv[i], "--output") == 0) {
      config->selfplay.output_path = argv[++i];
    } else if (strcmp(argv[i], "--model") == 0) {
      config->model_path = argv[++i];
    } else if (strcmp(argv[i], "--max-steps") == 0) {
      if (!parse_u32(argv[++i], &config->selfplay.max_steps_per_game)) {
        return -1;
      }
    } else if (strcmp(argv[i], "--max-records-per-round") == 0) {
      if (!parse_u32(argv[++i], &config->selfplay.max_records_per_round)) {
        return -1;
      }
    } else {
      return -1;
    }
  }
  return config->selfplay.output_path ? 1 : -1;
}

int main(int argc, char **argv) {
  cli_config config;
  model_policy policy;
  char error[256];
  int parsed = parse_arguments(argc, argv, &config);

  if (parsed <= 0) {
    print_usage(argv[0]);
    return parsed == 0 ? 0 : 2;
  }
  if (config.model_path) {
    if (!load_model(config.model_path, &policy)) {
      fprintf(stderr, "cj4me_generate: unable to load model\n");
      free_model(&policy);
      return 1;
    }
    config.selfplay.score_actions = score_actions;
    config.selfplay.score_context = &policy;
  } else {
    memset(&policy, 0, sizeof(policy));
  }
  if (!cj4me_generate_dataset(&config.selfplay, error, sizeof(error))) {
    fprintf(stderr, "cj4me_generate: %s\n", error);
    free_model(&policy);
    return 1;
  }

  printf("generated %" PRIu32 " game(s) with seed %" PRIu64 " to %s\n",
         config.selfplay.games, config.selfplay.seed,
         config.selfplay.output_path);
  free_model(&policy);
  return 0;
}
