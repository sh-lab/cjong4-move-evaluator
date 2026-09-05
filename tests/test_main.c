void test_dataset(void);
void test_evaluator(void);
void test_feature(void);
void test_model(void);
void test_rng(void);

int main(void) {
  test_dataset();
  test_evaluator();
  test_feature();
  test_model();
  test_rng();
  return 0;
}
