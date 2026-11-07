# Data and model sources

The digit study uses scikit-learn’s bundled subset of Alpaydin and Kaynak’s Optical Recognition of Handwritten Digits. Pixel values are scaled by 16 and the original row IDs are split for this teaching experiment. [UCI dataset and CC BY 4.0 attribution](https://archive.ics.uci.edu/dataset/80/optical+recognition+of+handwritten+digits).

The correspondence study uses Almeida and Hidalgo’s SMS Spam Collection. Canonical duplicates are removed before creating the 800/200/200 teaching splits. [UCI dataset and CC BY 4.0 attribution](https://archive.ics.uci.edu/dataset/228/sms+spam+collection). [Source and split metadata](data/projects/sms-source.json).

The pretrained study uses [bert-tiny](https://huggingface.co/prajjwal1/bert-tiny) at recorded revision `6f75de8b60a9f8a2fdf7b69cbd86d9e64bcb3837` (MIT per model card). The asset helper also supports [MiniLM](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) and [FLAN-T5 small](https://huggingface.co/google/flan-t5-small) (Apache 2.0 per model cards). Weights and raw SMS messages are downloaded separately.
