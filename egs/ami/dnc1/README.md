# Documentation for reproduction

## Step 1: Prepare meeting-level dvector dictionary
```sh
mkdir augmented_data
python3 datapreperation/gen_dvecdict.py \
	--input-scps data/train.scp \
	--input-mlfs data/train.mlf \
	--filtEncomp \
	--segLenConstraint 100 \
	--meetingLevelDict \
	augmented_data/dvecdict.meeting.split100
```

## Step 2: Prepare augmented training and validation data for pre-training
1. Maximum length 50
```sh
python3 datapreperation/gen_augment_data.py \
	--maxprocesses 16 \
	--input-scps data/train.scp \
	--input-mlfs data/train.mlf \
	--filtEncomp \
	--maxlen 50 \
	--augment 5000 \
	--varnormalise \
	--randomspeaker  \
	--dvectordict augmented_data/dvecdict.meeting.split100/train.npz \
	augmented_data/m50.meeting.augment

python3 datapreperation/gen_augment_data.py \
	--maxprocesses 16 \
	--input-scps data/dev.scp \
	--input-mlfs data/dev.mlf \
	--filtEncomp \
	--maxlen 50 \
	--augment 5000 \
	--varnormalise \
	augmented_data/m50.meeting.augment
```

2. Maximum length 200
```sh
python3 datapreperation/gen_augment_data.py \
	--maxprocesses 16 \
	--input-scps data/train.scp \
	--input-mlfs data/train.mlf \
	--filtEncomp \
	--maxlen 200 \
	--augment 10000 \
	--variableL 0.5 1.0 \
	--varnormalise \
	--randomspeaker  \
	--dvectordict augmented_data/dvecdict.meeting.split100/train.npz \
	augmented_data/m200.meeting.augment

python3 datapreperation/gen_augment_data.py \
	--maxprocesses 16 \
	--input-scps data/dev.scp \
	--input-mlfs data/dev.mlf \
	--filtEncomp \
	--maxlen 200 \
	--augment 1000 \
	--variableL 0.5 1.0 \
	--varnormalise \
	augmented_data/m200.meeting.augment
```

3. Maximum length 500
```sh
python3 datapreperation/gen_augment_data.py \
	--maxprocesses 16 \
	--input-scps data/train.scp \
	--input-mlfs data/train.mlf \
	--filtEncomp \
	--maxlen 500 \
	--augment 10000 \
	--variableL 0.5 1.0 \
	--varnormalise \
	--randomspeaker  \
	--dvectordict augmented_data/dvecdict.meeting.split100/train.npz \
	augmented_data/m500.meeting.augment

python3 datapreperation/gen_augment_data.py \
	--maxprocesses 16 \
	--input-scps data/dev.scp \
	--input-mlfs data/dev.mlf \
	--filtEncomp \
	--maxlen 500 \
	--augment 1000 \
	--variableL 0.5 1.0 \
	--varnormalise \
	augmented_data/m500.meeting.augment
```

4. Full length
```sh
python3 datapreperation/gen_augment_data.py \
	--maxprocesses 16 \
	--input-scps data/train.scp \
	--input-mlfs data/train.mlf \
	--filtEncomp \
	--augment 10000 \
	--variableL 0.5 1.0 \
	--varnormalise \
	--randomspeaker  \
	--dvectordict augmented_data/dvecdict.meeting.split100/train.npz \
	augmented_data/mFull.meeting.augment

python3 datapreperation/gen_augment_data.py \
	--maxprocesses 16 \
	--input-scps data/dev.scp \
	--input-mlfs data/dev.mlf \
	--filtEncomp \
	--augment 1000 \
	--variableL 0.5 1.0 \
	--varnormalise \
	augmented_data/mFull.meeting.augment
```

## Step 3: Prepare data for finetuning and evaluation
For fine-tuning on the full lengths
```sh
python3 datapreperation/gen_augment_data.py \
	--maxprocesses 16 \
	--input-scps data/train.scp \
	--input-mlfs data/train.mlf \
	--filtEncomp \
	--augment 1000 \
	--variableL 0.5 1.0 \
	--varnormalise \
	augmented_data/mFull.real.augment

python3 datapreperation/gen_augment_data.py \
	--maxprocesses 16 \
	--input-scps data/dev.scp \
	--input-mlfs data/dev.mlf \
	--filtEncomp \
	--augment 1000 \
	--variableL 0.5 1.0 \
	--varnormalise \
	augmented_data/mFull.real.augment
```

For final evaluation
```sh
python3 datapreperation/gen_augment_data.py \
	--maxprocesses 16 \
	--input-scps data/eval.scp \
	--input-mlfs data/eval.mlf \
	--filtEncomp \
	--augment 0 \
	--varnormalise \
	data/mFull.real
```

## Step 4: Train DNC models
First, go to the ESPnet directory
```sh
cd espnet/egs/ami/dnc1
```
### Pre-training on augmented data (curriculum leanrning)
1. Max length 50 (c.f. row 1 in Table 3)
```sh
./run.sh \
	--stage 4 \
	--stop_stage 4 \
	--train_config conf/tuning/train_transformer.yaml \
	--train_json ../../../../augmented_data/m50.meeting.augment/train.json \
	--dev_json ../../../../augmented_data/m50.meeting.augment/dev.json \
	--tag m50.meeting.diac.pt
```

2. Max length 200 (c.f. row 2 in Table 3)
```sh
./run.sh \
	--stage 4 \
	--stop_stage 4 \
	--train_config conf/tuning/train_transformer_cl.yaml \
	--train_json ../../../../augmented_data/m200.meeting.augment/train.json \
	--dev_json ../../../../augmented_data/m200.meeting.augment/dev.json \
	--train_sample 0.1 \
	--init_model exp/mdm_train_pytorch_m50.meeting.diac.pt/results/model.acc.best \
	--tag m200.meeting.diac.pt
```

3. Max length 500 (c.f. row 3 in Table 3)
```sh
./run.sh \
	--stage 4 \
	--stop_stage 4 \
	--train_config conf/tuning/train_transformer_cl.yaml \
	--train_json ../../../../augmented_data/m500.meeting.augment/train.json \
	--dev_json ../../../../augmented_data/m500.meeting.augment/dev.json \
	--train_sample 0.1 \
	--init_model exp/mdm_train_pytorch_m200.meeting.diac.pt/results/model.acc.best \
	--tag m500.meeting.diac.pt
```

4. Full length (c.f. row 4 in Table 3)
```sh
./run.sh \
	--stage 4 \
	--stop_stage 4 \
	--train_config conf/tuning/train_transformer_cl.yaml \
	--train_json ../../../../augmented_data/mFull.meeting.augment/train.json \
	--dev_json ../../../../augmented_data/mFull.meeting.augment/dev.json \
	--train_sample 0.02 \
	--init_model exp/mdm_train_pytorch_m500.meeting.diac.pt/results/model.acc.best \
	--tag mFull.meeting.diac.pt
```

### Fine-tuning on real data
```sh
./run.sh \
	--stage 4 \
	--stop_stage 4 \
	--train_config conf/tuning/train_transformer_ft.yaml \
	--train_json ../../../../augmented_data/mFull.meeting.real/train.json \
	--dev_json ../../../../augmented_data/mFull.meeting.real/dev.json \
	--train_sample 0.05 \
	--init_model exp/mdm_train_pytorch_mFull.meeting.diac.pt/results/model.acc.best \
	--tag mFull.diac.ft
```


## Step 5: Decode DNC models
```sh
./run.sh \
	--stage 5 \
	--stop_stage 5 \
	--nj 16 \
	--decode_set eval \
	--decode_json data/mFull.real/eval.json \
	--tag mFull.diac.ft \
	--decode_config conf/decode.yaml
```

## Step 6: Evaluate DNC models
```sh
cd ../../../../

python3 scoring/gen_rttm.py \
	--input-scp data/eval.scp \
	--js-dir espnet/egs/ami/dnc1/exp/mdm_train_pytorch_mFull.diac.ft/decode_mdm_eval_decode/ \
	--js-num 16 \
	--rttm-name eval_dnc

python3 scoring/split_rttm.py \
	--submeeting-rttm espnet/egs/ami/dnc1/exp/mdm_train_pytorch_mFull.diac.ft/decode_mdm_eval_decode/eval_dnc.rttm \
	--input-rttm scoring/refoutputeval.rttm \
	--output-rttm espnet/egs/ami/dnc1/exp/mdm_train_pytorch_mFull.diac.ft/decode_mdm_eval_decode/reference.rttm

python3 scoring/score_rttm.py \
	--score-rttm $(pwd)/espnet/egs/ami/dnc1/exp/mdm_train_pytorch_mFull.diac.ft/decode_mdm_eval_decode/eval_dnc.rttm \
	--ref-rttm $(pwd)/espnet/egs/ami/dnc1/exp/mdm_train_pytorch_mFull.diac.ft/decode_mdm_eval_decode/reference.rttm \
	--output-scoredir $(pwd)/espnet/egs/ami/dnc1/exp/mdm_train_pytorch_mFull.diac.ft/decode_mdm_eval_decode/scoring
```

