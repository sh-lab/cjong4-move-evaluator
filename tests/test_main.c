void test_dataset(void);
void test_evaluator(void);
void test_feature(void);
void test_model(void);
void test_policy_filter(void);
void test_rng(void);
void test_battle(void);

int main(void) {
  test_dataset();
  test_evaluator();
  test_feature();
  test_model();
  test_policy_filter();
  test_rng();
  test_battle();
  return 0;
}
