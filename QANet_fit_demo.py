from tensorflow.keras.optimizers import *
from tensorflow.keras.initializers import *
from QANet_keras import QANet
import numpy as np
import tensorflow as tf

#关闭eager模式
tf.compat.v1.disable_eager_execution()

tf.keras.backend.set_learning_phase(1)  # training

#就代表生成10000行 300列的浮点数，浮点数都是从0-1中随机。
##模拟的是预训练的词向量，生成QAnet模型，然后训练时对于某个词，该词存在该向量中，就取出该词向量进行使用，不存在就使用随机的词向量
##所以模型中一定有embedding模块

embedding_matrix = np.random.random((10000, 300))
embedding_matrix_char = np.random.random((1233, 64))
config = {
    'word_dim': 300,
    'char_dim': 64,
    'cont_limit': 400,
    'ques_limit': 50,
    'char_limit': 16,
    'ans_limit': 30,
    'char_input_size': 1233,
    'filters': 128,
    'num_head': 8,
    'dropout': 0.5,
    'batch_size': 16,
    'epoch': 25,
    'ema_decay': 0.9999,
    'learning_rate': 1e-3,
    'path': 'QA001',
    'use_cove': True
}
model = QANet(config, word_mat=embedding_matrix, char_mat=embedding_matrix_char)
model.summary()

optimizer = Adam(lr=0.001, beta_1=0.8, beta_2=0.999, epsilon=1e-7)

##损失函数有4个应该是模型有4个输出，对应4个label,计算每个输出的损失函数，加权求和后最为最终的损失函数，权重即为loss_weights
model.compile(optimizer=optimizer, loss=['categorical_crossentropy', 'categorical_crossentropy', 'mae', 'mae'],
              loss_weights=[1, 1, 0, 0])

# load data
char_dim = 200
cont_limit = 400
ques_limit = 50
char_limit = 16
#生成维度为（300，cont_limit）,大小在0-10000之间的随机整数##上下文长度最大400个词，每个词的维度是300d（感觉不对，应该是有300个上下文）
context_word = np.random.randint(0, 10000, (300, cont_limit))
question_word = np.random.randint(0, 10000, (300, ques_limit))

##最多400个词，每个词最多16个字符，字符维度也是300维度
context_char = np.random.randint(0, 96, (300, cont_limit, char_limit))
question_char = np.random.randint(0, 96, (300, ques_limit, char_limit))

start_label = np.random.randint(0, 2, (300, cont_limit))
end_label = np.random.randint(0, 2, (300, cont_limit))
start_label_fin = np.argmax(start_label, axis=-1)
end_label_fin = np.argmax(end_label, axis=-1)
'''
fit(x=None, y=None, batch_size=None, epochs=1, verbose=1, callbacks=None, 
validation_split=0.0, validation_data=None, shuffle=True, class_weight=None, 
sample_weight=None, initial_epoch=0, steps_per_epoch=None, validation_steps=None, validation_freq=1)

即
x=[context_word, question_word, context_char, question_char],
y=[start_label, end_label, start_label_fin, end_label_fin]

Model(inputs=[contw_input_, quesw_input_, contc_input_, quesc_input_],
                 outputs=[x_start, x_end, x_start_fin, x_end_fin])
                 
Model根据输入经过网络得到输出，输出和对应的label求出损失函数，损失函数加权后作为最终的损失函数，优化器使得最终的损失函数最小
'''
model.fit([context_word, question_word, context_char, question_char],
          [start_label, end_label, start_label_fin, end_label_fin], batch_size=8)
