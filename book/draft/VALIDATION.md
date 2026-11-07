# Evidence for the public draft

The digits experiment runs on real 8×8 images with 1,078 training, 359 validation and 360 test examples. Chapter 6’s linear model classified 347/359 validation examples correctly and the MLP classified 350/359. Chapter 7 uses a separately declared momentum policy and tests exact epoch-boundary resumption.

Chapter 9 ran 150 updates per policy on 800 SMS training examples, then chose a checkpoint using 200 validation examples. On the 200-example test set the selected last-layer BERT had macro-F1 0.953, with one false positive and three false negatives. Its saved model and tokenizer reproduced the probe logits. These numbers describe a small historical dataset and a specific configuration.

The independent compact TF-IDF/PyTorch SMS service had test accuracy 0.970 and spam recall 0.80. All 200 test predictions matched after restoration in a fresh process. [Recorded runtime evidence](../../data/projects/sms-runtime-evidence.json).

Chapter 1 and Chapter 6 have completed independent editorial review. The later RAG, GPT, PEFT and integrated capstone studies remain under development. The offline suite and executed notebooks check software behavior; the curriculum review continues.

Public-edition verification: 291 tests passed and two pretrained tests skipped in the offline suite. With the verified BERT and SMS assets available, Chapters 6, 8 and 9 each passed their separate tests and fresh-kernel execution. All 24 worked cells ran across those three notebooks, with four unfinished exercises intentionally skipped. All 26 notebooks passed schema and manuscript parity checks.
