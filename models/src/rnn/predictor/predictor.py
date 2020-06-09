from copy import deepcopy
from typing import List, Dict
import json

from overrides import overrides
import numpy

from allennlp.common.util import JsonDict
from allennlp.data import Instance
from allennlp.predictors.predictor import Predictor
from allennlp.data.fields import LabelField


@Predictor.register("sentiment")
class TextClassifierPredictor(Predictor):
   
    def batch_json_to_labeled_instances(self, inputs: List[JsonDict]) -> List[Instance]:
        instances = self._batch_json_to_instances(inputs)
        outputs = self._model.forward_on_instances(instances)
        new_instances = self.predictions_to_labeled_instances(inputs, outputs)
        return new_instances

    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        """
        Expects JSON that looks like `{"text": "..."}`.
        Runs the underlying model, and adds the `"label"` to the output.
        """
        sentence = json_dict["text"]
        if not hasattr(self._dataset_reader, "tokenizer") and not hasattr(
            self._dataset_reader, "_tokenizer"
        ):
            tokenizer = SpacyTokenizer()
            sentence = [str(t) for t in tokenizer.tokenize(sentence)]
        return self._dataset_reader.text_to_instance(sentence)

    @overrides
    def predictions_to_labeled_instances(
        self, json_dicts: List[JsonDict], outputs: Dict[str, numpy.ndarray]
    ) -> List[JsonDict]:

        output_json_dicts = []
        for json, output in zip(json_dicts, outputs):

            new_json = deepcopy(json)

            label = numpy.argmax(output["probs"])
            new_json["label"] = str(label)
            output_json_dicts.append(new_json)
        return output_json_dicts

if __name__ == '__main__':

    import os
    import sys
    sys.path.append(os.getcwd())

    from models.src.rnn.data.dataset_readers.reader import TwiReader
    from models.src.rnn.model.model import RnnClassifier
    
    #from allennlp.models.archival import load_archive
    from allennlp.predictors import Predictor
    # https://github.com/allenai/allennlp/blob/master/allennlp/predictors/predictor.py
    # predictorはメンバ変数にmodelとreaderをもつ

    model_path = 'models/serials/xxlarge-bin/model.tar.gz'
    input_path = 'models/example/input/input.jsonl'
    output_path = 'models/example/output/output.jsonl'

    # これだと学習時と同じconfigのpredictorを作れるが、cuda_deviceもそのまま
    #archive = load_archive(model_path) 
    #predictor = Predictor.from_archive(archive=archive, predictor_name='sentiment-classifier')
    
    # こっちはcuda_deviceを指定できる
    predictor = Predictor.from_path(archive_path=model_path, predictor_name='sentiment', cuda_device=-1)

    with open(input_path, 'r') as f:
        json_lines = f.readlines()

    json_dicts = []
    for line in json_lines:
        json_dicts.append(predictor.load_line(line))

    output_dicts = predictor.batch_json_to_labeled_instances(json_dicts)
    outputs = [repr(json.dumps(d).encode().decode('unicode-escape')).strip('\'') + '\n' for d in output_dicts]

    with open(output_path, 'w') as f:
        f.writelines(outputs)

