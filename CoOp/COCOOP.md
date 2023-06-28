## How to Run

The running scripts are provided in `scripts/cocoop/`, which allow you to reproduce the results on the CVPR'22 paper.

Make sure you change the path in `DATA` and run the commands under the main directory `CoOp/`.

### Generalization From Base to New Classes

This corresponds to the experiments in Section 4.1, i.e., Table 1.

You will need both `scripts/cocoop/base2new_train.sh` and `scripts/cocoop/base2new_test.sh`. The former trains a model on bash classes while the latter evaluates the trained model on new classes. Both scripts have two input arguments, i.e., `DATASET` and `SEED`.

`DATASET` takes as input a dataset name, like `plant_village` or `plant_village`. The valid names are the files' names in `CoOp/configs/datasets/`.

Below we provide an example on how to evaluate the model on plant_village.

```bash
# seed=1
bash scripts/cocoop/base2new_train.sh plant_village 1
bash scripts/cocoop/base2new_test.sh plant_village 1

# seed=2
bash scripts/cocoop/base2new_train.sh plant_village 2
bash scripts/cocoop/base2new_test.sh plant_village 2

# seed=3
bash scripts/cocoop/base2new_train.sh plant_village 3
bash scripts/cocoop/base2new_test.sh plant_village 3
```

When the evaluation is done, you can use `parse_test_res.py` to automatically calculate the average results. For instance, after you finish the evaluation (including `base2new_train.sh` and `base2new_test.sh`) on plant_village using the aforementioned commands, you would get

```
output
|–– base2new/
|   |–– test_new/
|   |   |–– plant_village/
|   |   |   |–– shots_16/
|   |   |   |   |–– CoCoOp/
|   |   |   |   |   |–– vit_b16_c4_ep10_batch1_ctxv1/
|   |   |   |   |   |   |–– seed1/
|   |   |   |   |   |   |–– seed2/
|   |   |   |   |   |   |–– seed3/
|   |–– train_base/
|   |   |–– plant_village/
|   |   |   |–– shots_16/
|   |   |   |   |–– CoCoOp/
|   |   |   |   |   |–– vit_b16_c4_ep10_batch1_ctxv1/
|   |   |   |   |   |   |–– seed1/
|   |   |   |   |   |   |–– seed2/
|   |   |   |   |   |   |–– seed3/
```

Then, to get the average performance on the base classes, run

```bash
python parse_test_res.py output/base2new/train_base/plant_village/shots_16/CoCoOp/vit_b16_c4_ep10_batch1_ctxv1
```

To get the average performance on the new classes, run

```bash
python parse_test_res.py output/base2new/test_new/plant_village/shots_16/CoCoOp/vit_b16_c4_ep10_batch1_ctxv1 --test-log
```