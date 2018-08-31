# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
r"""Generate captions for images using default beam search parameters."""
# check this blog https://blog.csdn.net/heros_never_die/article/details/79763546
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os


import tensorflow as tf

from im2txt import configuration
from im2txt import inference_wrapper
from im2txt.inference_utils import caption_generator
from im2txt.inference_utils import vocabulary

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string("checkpoint_path", "",
                       "Model checkpoint file or directory containing a "
                       "model checkpoint file.")
tf.flags.DEFINE_string("vocab_file", "", "Text file containing the vocabulary.")
tf.flags.DEFINE_string("input_files", "",
                       "File pattern or comma-separated list of file patterns "
                       "of image files.")

tf.logging.set_verbosity(tf.logging.INFO)

def main(_):
  # Build the inference graph.
  g = tf.Graph()
  with g.as_default():
    model = inference_wrapper.InferenceWrapper()
    restore_fn = model.build_graph_from_config(configuration.ModelConfig(),
                                               FLAGS.checkpoint_path)
  g.finalize()

  # Create the vocabulary.
  vocab = vocabulary.Vocabulary(FLAGS.vocab_file)

  filenames = []
  for file_pattern in FLAGS.input_files.split(","):
    filenames.extend(tf.gfile.Glob(file_pattern))
  tf.logging.info("Running caption generation on %d files matching %s",
                  len(filenames), FLAGS.input_files)

  with tf.Session(graph=g) as sess:
    # Load the model from checkpoint.
    restore_fn(sess)

    # Prepare the caption generator. Here we are implicitly using the default
    # beam search parameters. See caption_generator.py for a description of the
    # available beam search parameters.
    generator = caption_generator.CaptionGenerator(model, vocab)

    for filename in filenames:
      with tf.gfile.GFile(filename, "r") as f:
        image = f.read()
      captions = generator.beam_search(sess, image)
      print("Captions for image %s:" % os.path.basename(filename))
      for i, caption in enumerate(captions):
        # Ignore begin and end words.
        sentence = [vocab.id_to_word(w) for w in caption.sentence[1:-1]]
        sentence = " ".join(sentence)
        print("  %d) %s (p=%f)" % (i, sentence, math.exp(caption.logprob)))

#old
#FLAGS.checkpoint_path = "D:/sthself/ml/RNN/im2txt/im2txt_2016_10_11.2000000/model.ckpt-2000000"
#new
FLAGS.checkpoint_path = "D:/sthself/ml/RNN/im2txt/im2txt_2016_10_11.2000000/newmodel.ckpt-2000000"

FLAGS.vocab_file = "./data/volab.txt"
FLAGS.input_files = "./data/bug.jpg,./data/bug2.jpg,./data/bug1.jpg,./data/COCO_val2014_000000224477.jpg,./data/ep271.jpg,./data/dog.jpg,./data/beauty.jpg,./data/bethan.jpg,./data/sharon.jpg,./data/games.jpg,./data/sharon1.jpg,./data/sharon3.jpg,./data/brotheryong1.jpg,./data/liangyishi.jpg,./data/desk.jpg"

# 由于版本不同，需要进行修改
def RenameCkpt():
    '''vars_to_rename = {
    "lstm/BasicLSTMCell/Linear/Matrix": "lstm/basic_lstm_cell/weights",
    "lstm/BasicLSTMCell/Linear/Bias": "lstm/basic_lstm_cell/biases",
    }'''
    # this change vars for some unknow things for website ,but looks it work ,good by ethan 2018-08-31 friday sunney with braveheart
    vars_to_rename = {
        "lstm/BasicLSTMCell/Linear/Matrix": "lstm/basic_lstm_cell/kernel",
        "lstm/BasicLSTMCell/Linear/Bias": "lstm/basic_lstm_cell/bias",
    }
    new_checkpoint_vars = {}
    print(FLAGS.checkpoint_path)
    reader = tf.train.NewCheckpointReader(FLAGS.checkpoint_path)
    for old_name in reader.get_variable_to_shape_map():
      if old_name in vars_to_rename:
        new_name = vars_to_rename[old_name]
      else:
        new_name = old_name
      new_checkpoint_vars[new_name] = tf.Variable(reader.get_tensor(old_name))
    
    init = tf.global_variables_initializer()
    saver = tf.train.Saver(new_checkpoint_vars)
    
    with tf.Session() as sess:
      sess.run(init)
      saver.save(sess, "D:/sthself/ml/RNN/im2txt/im2txt_2016_10_11.2000000/newmodel.ckpt-2000000")
    print("checkpoint file rename successful... ")
#FLAGS.checkpoint_path = "D:/sthself/ml/RNN/im2txt/im2txt_2016_10_11.2000000/newmodel1.ckpt-2000000"
# pic 2 txt entry 
def im2txt():
  # Build the inference graph.
  g = tf.Graph()
  with g.as_default():
    model = inference_wrapper.InferenceWrapper()
    restore_fn = model.build_graph_from_config(configuration.ModelConfig(),
                                               FLAGS.checkpoint_path)
  g.finalize()
  # 创建词汇表
  vocab = vocabulary.Vocabulary(FLAGS.vocab_file)

  filenames = []
  for file_pattern in FLAGS.input_files.split(","):
    filenames.extend(tf.gfile.Glob(file_pattern))
  tf.logging.info("Running caption generation on %d files matching %s",
                  len(filenames), FLAGS.input_files)
  with tf.Session(graph=g) as sess:
    # Load the model from checkpoint.
    restore_fn(sess)

    # Prepare the caption generator. Here we are implicitly using the default
    # beam search parameters. See caption_generator.py for a description of the
    # available beam search parameters.
    generator = caption_generator.CaptionGenerator(model, vocab)

    for filename in filenames:
      with tf.gfile.GFile(filename, "rb") as f:   # edit by Ethan on 2018-08-31 change from r to rb
          print(filename)
          image = f.read()
      captions = generator.beam_search(sess, image)
      print("图像 %s 标题是:" % os.path.basename(filename))
      for i, caption in enumerate(captions):
        # Ignore begin and end words.
        sentence = [vocab.id_to_word(w) for w in caption.sentence[1:-1]]
        sentence = " ".join(sentence)
        print("  %d) %s (概率=%f)" % (i, sentence, math.exp(caption.logprob)))


if __name__ == "__main__":
    #RenameCkpt()
   im2txt()
#   tf.app.run()
