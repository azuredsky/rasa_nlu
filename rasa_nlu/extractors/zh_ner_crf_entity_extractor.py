from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging
import os
import re
import typing
from builtins import str
from typing import Any, Dict, List, Optional, Text, Tuple

from rasa_nlu.config import RasaNLUModelConfig, InvalidConfigError
from rasa_nlu.extractors import EntityExtractor
from rasa_nlu.model import Metadata
from rasa_nlu.training_data import Message
from rasa_nlu.training_data import TrainingData

import tensorflow as tf
import numpy as np
import os, argparse, time, random
from rasa_nlu.utils.zh_ner_model import BiLSTM_CRF, tf_config
from rasa_nlu.utils.zh_ner_utils import str2bool, get_logger, get_entity
from rasa_nlu.utils.zh_ner_data import read_corpus, read_dictionary, tag2label, random_embedding

try:
    import tensorflow as tf
except ImportError:
    tf = None

logger = logging.getLogger(__name__)


class ZHNERCRFEntityExtractor(EntityExtractor):
    name = "zh_ner_crf_tf"

    provides = ["entities"]

    requires = ["tokens"]

    defaults = {
        # BILOU_flag determines whether to use BILOU tagging or not.
        # More rigorous however requires more examples per entity
        # rule of thumb: use only if more than 100 egs. per entity
        "BILOU_flag": True,
        "train_data_path": 'data_path',
        "test_data_path": 'data_path',
        # sample of each minibatch,
        "batch_size": 64,
        # epoch of training,
        "epoch": 40,
        # dim of hidden state
        "hidden_dim": 300,
        # 'Adam/Adadelta/Adagrad/RMSProp/Momentum/SGD'
        "optimizer": 'Adam',
        # use CRF at the top layer. if False, use Softmax
        "CRF": True,
        # learning rate
        "lr": 0.001,
        # gradient clipping
        "clip": 5.0,
        # dropout keep_prob
        "dropout": 0.5,
        # update embedding during training
        "update_embedding": True,
        # use pretrained char embedding or init it randomly
        "pretrain_embedding": 'random',
        # random init char embedding_dim
        "embedding_dim": 300,
        # shuffle training data before each epoch
        "shuffle": True
    }
    
    @classmethod
    def required_packages(cls):
        # type: () -> List[Text]
        return ["tensorflow"]

    @staticmethod
    def _check_tensorflow():
        if tf is None:
            raise ImportError(
                'Failed to import `tensorflow`. '
                'Please install `tensorflow`. '
                'For example with `pip install tensorflow`.')

    def __init__(self,
                 component_config=None,
                 ent_tagger=None,
                 session=None,  # type: Optional[tf.Session]
                 graph=None     # type: Optional[tf.Graph])
                 ):
        """Declare instant variables with default values"""
        self._check_tensorflow()
        super(ZHNERCRFEntityExtractor, self).__init__(component_config)

        self.component_config = component_config
        self.ent_tagger = ent_tagger
        self.session = session

    def train(self, training_data, config, **kwargs):
        # type: (TrainingData, RasaNLUModelConfig) -> None

        self.component_config = config.for_component(self.name, self.defaults)

        self.model = BiLSTM_CRF(self.component_config)
        self.model.build_graph()
        ## hyperparameters-tuning, split train/dev
        # dev_data = train_data[:5000]; dev_size = len(dev_data)
        # train_data = train_data[5000:]; train_size = len(train_data)
        # print("train data: {0}\ndev data: {1}".format(train_size, dev_size))
        # model.train(train=train_data, dev=dev_data)

        train_data_new = read_corpus(self.model.train_path)
        test_data = read_corpus(self.model.test_path);
        test_size = len(test_data)

        ## train model on the whole training data

        self.session = tf.Session(config=tf_config)
        self.model.train(self.session, train=train_data_new, dev=test_data)  # use test_data as the dev_data to see overfitting phenomena

    def process(self, message, **kwargs):
        # type: (Message, **Any) -> None

        extracted = self.add_extractor_name(self.extract_entities(message))
        message.set("entities", message.get("entities", []) + extracted,
                    add_to_output=True)

    def extract_entities(self, message):
        # type: (Message) -> List[Dict[Text, Any]]
        """Take a sentence and return entities in json format"""

        if self.ent_tagger is not None:
            demo_sent = list(message.text.strip())
            demo_data = [(demo_sent, ['O'] * len(demo_sent))]
            tag = self.ent_tagger.demo_one(self.session, demo_data)

            #PER, LOC, ORG = get_entity(tag, demo_sent)
            #print('PER: {}\nLOC: {}\nORG: {}'.format(PER, LOC, ORG))

            return self._from_crf_to_json(message, tag)
        else:
            return []

    def _from_crf_to_json(self, message, entities):
        # type: (Message, List[Any]) -> List[Dict[Text, Any]]

        tokens = list(message.text.strip())

        if len(tokens) != len(entities):
            raise Exception('Inconsistency in amount of tokens '
                            'between crfsuite and message')
        json_ents = []

        word_idx = 0
        start = 0
        end_idx=0
        finish = False
        mark = False
        entities.append('O')
        while word_idx < len(entities):
            entity_label = entities[word_idx]
            label = str(entity_label)

            if label.startswith('B-'):
                start = word_idx
                mark = True
                end_idx = start + 1

            elif label.startswith('I-'):
                finish= False
                end_idx += 1

            else:
                finish= mark
                end_idx -= 1

            if (mark and finish):
                value = ' '.join(t for t in tokens[start:end_idx + 1])
                finish = False
                mark = False

                ent = {
                        'start': start,
                        'end': end_idx+1,
                        'value': value,
                        'entity': str(entities[start])[2:],
                        'confidence': 1.0
                      }
                json_ents.append(ent)

            word_idx += 1

        return json_ents

    @classmethod
    def load(cls,
             model_dir=None,  # type: Text
             model_metadata=None,  # type: Metadata
             cached_component=None,  # type: Optional[Component]
             **kwargs  # type: **Any
             ):
        meta = model_metadata.for_component(cls.name)

        sess = tf.Session(config=tf_config)
        model = BiLSTM_CRF(meta)
        model.build_graph()

        if model_dir and meta.get("zh_ner_crf_model"):
            file_name = meta.get("zh_ner_crf_model")
            checkpoint = os.path.join(model_dir, file_name)
            model.saver.restore(sess, checkpoint)
        
            return ZHNERCRFEntityExtractor(
                    component_config=meta,
                    ent_tagger=model,
                    session=sess
            )

        else:
            logger.warning("Failed to load nlu model. Maybe path {} "
                           "doesn't exist"
                           "".format(os.path.abspath(model_dir)))
            return ZHNERCRFEntityExtractor(component_config=meta)

    def persist(self, model_dir):
        # type: (Text) -> Optional[Dict[Text, Any]]
        """Persist this model into the passed directory.
        Return the metadata necessary to load the model again."""
        if self.session is None:
            logger.warning("session is none")
            return {"zh_ner_crf_model": None}

        checkpoint = os.path.join(model_dir, self.name + ".ckpt")

        try:
            os.makedirs(os.path.dirname(checkpoint))
        except OSError as e:
            # be happy if someone already created the path
            import errno
            if e.errno != errno.EEXIST:
                raise

        self.model.save_global_model(self.session, checkpoint)

        return {"zh_ner_crf_model": self.name + ".ckpt"}
