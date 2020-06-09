 
{
  "dataset_reader": {
    "type": "twireader",
    "token_indexers": {
      "tokens": {
        "type": "single_id",
        "lowercase_tokens": true
      }
    },
    "max_sequence_length": 300
  },
  "train_data_path": "models/data/debug.jsonl",
  "validation_data_path": "models/data/debug.jsonl",
  "model": {
    "type": "rnn_clf",
    "text_field_embedder": {
      "token_embedders": {
        "tokens": {
          "type": "embedding",
          "embedding_dim": 100
        }
      }
    },
    "seq2vec_encoder": {
      "type": "lstm",
      "bidirectional": true,
      "input_size": 100,
      "hidden_size": 100,
      "num_layers": 1
    },
    "dropout": 0.5
  },
  "iterator": {
    "type": "bucket",
    "sorting_keys": [["tokens", "num_tokens"]],
    "batch_size": 4
  },

  "trainer": {
    "num_epochs": 10,
    "cuda_device": 0,
    "learning_rate_scheduler": {
      "type": "reduce_on_plateau",
      "factor": 0.5,
      "mode": "max",
      "patience": 2
    },
    "optimizer": {
      "type": "adam",
      "betas": [0.5, 0.9]
    }
  }
}