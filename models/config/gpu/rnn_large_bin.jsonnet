local EMBD = 256;
local HDN = 128;

{
  "dataset_reader": {
    "type": "twireader",
    "token_indexers": {
      "tokens": {
        "type": "single_id",
        "lowercase_tokens": true
      }
    },
    "max_sequence_length": 128
  },
  "vocabulary":{
    "directory_path": "models/data/vocab/"
  },
  "train_data_path": "models/data/binary_label/train.jsonl",
  "validation_data_path": "models/data/binary_label/valid.jsonl",
  "model": {
    "type": "rnn_clf",
    "text_field_embedder": {
      "token_embedders": {
        "tokens": {
          "type": "embedding",
          "embedding_dim": EMBD
        }
      }
    },
    "seq2vec_encoder": {
      "type": "lstm",
      "bidirectional": true,
      "input_size": EMBD,
      "hidden_size": HDN,
      "num_layers": 1
    },
    "dropout": 0.5
  },
  "iterator": {
    "type": "bucket",
    "batch_size": 2,
    "sorting_keys": [["tokens", "num_tokens"]],
    "padding_noise": 1e-3
  },

  "trainer": {
    "num_epochs": 1000,
    "patience": 30, 
    "cuda_device": 0,
    "model_save_interval": 2000,
    "summary_interval": 100,
    "histogram_interval": 100,
    "should_log_learning_rate": true,
    "grad_clipping": 1.0,
    "validation_metric": "-loss",
    "optimizer": {
      "type": "adam",
      "lr": 5e-4,
      "weight_decay": 0.001
    },
    "learning_rate_scheduler": {
        "type": "cosine",
        "t_initial": 25
    }
  }
}