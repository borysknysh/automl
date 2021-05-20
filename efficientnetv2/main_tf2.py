# Copyright 2021 Google Research. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""A simple script to train efficient net with tf2/keras."""

import copy
import os
import re

from absl import app
from absl import flags
from absl import logging
import tensorflow as tf

import cflags
import datasets
import effnetv2_configs
import effnetv2_model
import hparams
import utils
import numpy as np
FLAGS = flags.FLAGS

gpus = tf.config.experimental.list_physical_devices("GPU")

for gpu in gpus:
	tf.config.experimental.set_memory_growth(gpu, True)
if gpus:
	tf.config.experimental.set_visible_devices(gpus[0], "GPU")

def build_tf2_optimizer(learning_rate,
                        optimizer_name='rmsprop',
                        decay=0.9,
                        epsilon=0.001,
                        momentum=0.9):
  """Build optimizer."""
  if optimizer_name == 'sgd':
    logging.info('Using SGD optimizer')
    optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
  elif optimizer_name == 'momentum':
    logging.info('Using Momentum optimizer')
    optimizer = tf.keras.optimizers.SGD(
        learning_rate=learning_rate, momentum=momentum)
  elif optimizer_name == 'rmsprop':
    logging.info('Using RMSProp optimizer')
    optimizer = tf.keras.optimizers.RMSprop(learning_rate, decay, momentum,
                                            epsilon)
  elif optimizer_name == 'adam':
    logging.info('Using Adam optimizer')
    optimizer = tf.keras.optimizers.Adam(learning_rate)
  else:
    raise Exception('Unknown optimizer: %s' % optimizer_name)

  return optimizer


class TrainableModel(effnetv2_model.EffNetV2Model):
  """Wraps efficientnet to make a keras trainable model.

  Handles efficientnet's multiple outputs and adds weight decay.
  """

  def __init__(self,
               model_name='efficientnetv2-s',
               model_config=None,
               name=None,
               weight_decay=0.0):
    super().__init__(
        model_name=model_name,
        model_config=model_config,
        name=name or model_name)

    self.weight_decay = weight_decay

  def _reg_l2_loss(self, weight_decay, regex=r'.*(kernel|weight):0$'):
    """Return regularization l2 loss loss."""
    var_match = re.compile(regex)
    return weight_decay * tf.add_n([
        tf.nn.l2_loss(v)
        for v in self.trainable_variables
        if var_match.match(v.name)
    ])

  def train_step(self, data):
    features, labels = data
    images, labels = features['image'], labels['label']

    with tf.GradientTape() as tape:
      pred = self(images, training=True)[0]
      pred = tf.cast(pred, tf.float32)
      loss = self.compiled_loss(
          labels,
          pred,
          regularization_losses=[self._reg_l2_loss(self.weight_decay)])

    self.optimizer.minimize(loss, self.trainable_variables, tape=tape)
    self.compiled_metrics.update_state(labels, pred)
    return {m.name: m.result() for m in self.metrics}

  def test_step(self, data):
    features, labels = data
    images, labels = features['image'], labels['label']
    pred = self(images, training=False)[0]
    pred = tf.cast(pred, tf.float32)

    self.compiled_loss(
        labels,
        pred,
        regularization_losses=[self._reg_l2_loss(self.weight_decay)])

    self.compiled_metrics.update_state(labels, pred)
    return {m.name: m.result() for m in self.metrics}


def main(_) -> None:
  config = copy.deepcopy(hparams.base_config)
  config.override(effnetv2_configs.get_model_config('efficientnetv2-l'))
  #config.override(datasets.get_dataset_config(FLAGS.dataset_cfg))
  config.override(FLAGS.hparam_str)
  config.model.num_classes = config.data.num_classes
  
  model = TrainableModel(
        config.model.model_name,
        config.model,
        weight_decay=config.train.weight_decay)
  optimizer = build_tf2_optimizer(3e-4, optimizer_name=config.train.optimizer)
  
  model.compile(optimizer=optimizer,
                loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=config.train.label_smoothing, from_logits=True))
  
  #model.summary()
  inp = np.random.rand(1,224,224,3)
  res = model.predict(inp)
  print(len(res))
  model.save('saved_model')
  #x = tf.keras.Input(shape=(224,224,3))
  #modelka = tf.keras.Model(inputs=[x], outputs=model.call(x, training=True))
  
  #print(modelka.outputs)

if __name__ == '__main__':
  cflags.define_flags()
  app.run(main)
