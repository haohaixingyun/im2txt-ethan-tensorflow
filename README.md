# 谷歌图像叙事功能
基于tensorflow 1.0实现im2txt。也可见博客地址：[CSDN博客](http://blog.csdn.net/sparkexpert/article/details/70846094)

# 预训练模型下载
由于本人实验环境相对较差，没有GPU，所以没有测试训练过程。因此下载了个预训练模型。
下载地址如下所示：https://drive.google.com/file/d/0Bw6m_66JSYLlRFVKQ2tGcUJaWjA/view

### 运行环境介绍
* Python 3.6
* Tensorflow >= 1.0.1
* model/im2txt

# 测试过程中填“坑”
（1） word_counts.txt文件的处理，需要将文件中的 b' str'  ==>  str，即把字符串的引号等全部去掉。


（2）修改预训练模型中的名称，由于预训练模型的名称不一致的问题，所以需要进行修改。

在具体代码修改中，添加一个函数来进行模型的修改和重新保存

# 由于版本不同，需要进行修改
def RenameCkpt():
    vars_to_rename = {
    "lstm/BasicLSTMCell/Linear/Matrix": "lstm/basic_lstm_cell/weights",
    "lstm/BasicLSTMCell/Linear/Bias": "lstm/basic_lstm_cell/biases",
    }
    new_checkpoint_vars = {}
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
      saver.save(sess, "/home/ndscbigdata/work/change/tf/gan/im2txt/ckpt/newmodel.ckpt-2000000")
    print("checkpoint file rename successful... ")

# 训练结果：
	图片放在data目录下:
	![[image](./data/COCO_val2014_000000224477.jpg)]
图像 COCO_val2014_000000224477.jpg 标题是:
  0) a man riding a wave on top of a surfboard . (概率=0.035672)
  1) a person riding a surf board on a wave (概率=0.016238)
  2) a man on a surfboard riding a wave . (概率=0.010146)
  
	![[image](./data/ep271.jpg)]
图像 ep271.jpg 标题是:
  0) a woman is standing next to a horse . (概率=0.000759)
  1) a woman is standing next to a horse (概率=0.000647)
  2) a woman is standing next to a brown horse . (概率=0.000384)
  
	![[image](./data/dog.jpg)]
图像 dog.jpg 标题是:
  0) a dog is eating a slice of pizza . (概率=0.000138)
  1) a dog is eating a slice of pizza on a plate . (概率=0.000047)
  2) a dog is sitting at a table with a pizza on it . (概率=0.000039)


-----------------------------------------------
Tool version :
python 3.6.2 Anaconda
tf.__version__ 1.3.0

new dig hold:
1.UnicodeDecodeError: 'utf-8' codec can't decode byte 0xff in position 0: invalid start byte

2.tensorflow version 1.3  lstm 转换

图像 COCO_val2014_000000224477.jpg 标题是:
  0) a man riding a wave on top of a surfboard . (概率=0.035689)
  1) a person riding a surf board on a wave (概率=0.016247)
  2) a man on a surfboard riding a wave . (概率=0.010152)
.\data\ep271.jpg
图像 ep271.jpg 标题是:
  0) a woman is standing next to a horse . (概率=0.000733)
  1) a woman is standing next to a horse (概率=0.000642)
  2) a woman is standing next to a brown horse . (概率=0.000376)
.\data\dog.jpg
图像 dog.jpg 标题是:
  0) a dog is eating a slice of pizza . (概率=0.000138)
  1) a dog is eating a slice of pizza on a plate . (概率=0.000047)
  2) a dog is sitting at a table with a pizza on it . (概率=0.000039)
.\data\beauty.jpg
图像 beauty.jpg 标题是:
  0) a little girl wearing a hat and holding a pink umbrella . (概率=0.000001)
  1) a little girl wearing a pink hat holding a pink umbrella . (概率=0.000001)
  2) a little girl wearing a pink hat and a pink hat . (概率=0.000001)
.\data\ethan.jpg
图像 ethan.jpg 标题是:
  0) a man in a suit and tie standing in a room . (概率=0.000935)
  1) a man in a suit and tie standing in a kitchen . (概率=0.000526)
  2) a man in a suit and tie standing in a room (概率=0.000242)
.\data\sharon.jpg
图像 sharon.jpg 标题是:
  0) a little girl is holding a pink umbrella . (概率=0.000076)
  1) a little girl is holding a pink umbrella (概率=0.000065)
  2) a little girl holding a pink umbrella in her hands . (概率=0.000050)
.\data\games.jpg
图像 games.jpg 标题是:
  0) a woman sitting on a bench with her legs crossed . (概率=0.000223)
  1) a woman is sitting on a skateboard in the street . (概率=0.000204)
  2) a woman sitting on a bench with her legs crossed (概率=0.000086)
.\data\sharon1.jpg
图像 sharon1.jpg 标题是:
  0) a man and a woman standing next to a stop sign . (概率=0.000031)
  1) a man and a woman standing next to a fire hydrant . (概率=0.000030)
  2) a man and a woman standing next to a sign . (概率=0.000013)
.\data\sharon3.jpg
图像 sharon3.jpg 标题是:
  0) a group of young men playing a game of frisbee . (概率=0.004837)
  1) a group of young people playing a game of frisbee . (概率=0.002881)
  2) a group of young people playing a game of frisbee (概率=0.000175)
.\data\brotheryong1.jpg
图像 brotheryong1.jpg 标题是:
  0) a man sitting at a table with a laptop . (概率=0.000624)
  1) a man sitting at a table with a laptop computer . (概率=0.000398)
  2) a man sitting at a table with a laptop computer in front of him . (概率=0.000068)
.\data\brotheryong2.jpg
图像 brotheryong2.jpg 标题是:
  0) a man sitting at a table with a laptop . (概率=0.000382)
  1) a man sitting on a chair in a room . (概率=0.000313)
  2) a man sitting at a table with a laptop computer . (概率=0.000224)