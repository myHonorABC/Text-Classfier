import tensorflow as tf
import numpy as np
import sys,time,datetime
import data_helpers
import text_cnn
from tensorflow.contrib import learn


#param

#Data loading params
tf.flags.DEFINE_float('dev_sample_percetage',1,'Percentage of the training data to use for vaidation')
tf.flags.DEFINE_string('positive_data_file','./data/rt-polarity.pos','Data source for the positive')
tf.flags.DEFINE_string('negative__data_file','./data/rt-polarity.neg','Data source for the negative')

#Model Hparams
tf.flags.DEFINE_integer('embedding',120,'Dimensionality of character embedding')
tf.flags.DEFINE_string('filter_size','3,4,5','filter size')
tf.flags.DEFINE_integer('num_filters',128,'Number of filters')
tf.flags.DEFINE_float('dropout',0.5,'Dropout')
tf.flags.DEFINE_float('L2_reg_lambda',0.0,'L2')

#Taining params
tf.flags.DEFINE_integer('batch_size',64,'Batch Size')     #训练集的大小
tf.flags.DEFINE_integer('num_epochs',200,'Number of epochs')   #迭代次数
tf.flags.DEFINE_integer('evaluate_every',100,'evaluate_every')
tf.flags.DEFINE_integer('checkpoint_every',100,'Saving......')
tf.flags.DEFINE_integer('num_checkpoints',5,'num_checkpoints')  #每训练五次保存一次模型
tf.flags.DEFINE_boolean('log_soft_placement',True,'log_soft_placement')   #是否打印日志
tf.flags.DEFINE_boolean('allow_soft_placement',True,'allow_soft_placement')   #当cpu或者gpu不存在时是否自动分配

FLAGS=tf.flags.FLAGS
FLAGS.flag_values_dict()                #将flags类型数据转化为字典类型

#打印事先定义的参数
print('\nParameters:')
for attr,value in sorted(FLAGS._flags.items()):
    print('{}={}'.format(attr.upper(),value))

#loading data
x_text,y=data_helpers.load_data_and_labels(FLAG.positive_data_file,FLAG.negative_data_file)

#使文件中的每个句子包含单词个数相同
max_document_length=max(len(x.split(' ')) for x in x_text)
vocab_prcocessor=learn.preprocessing.text.VocabularyProcessor(max_document_length)
x=np.array(list(vocab_processor.fit_transform(x_text)))

#打乱样本顺序
np.random.seed(10)
shuffle_indices=np.random.permutation(np.arrange(len(y)))
x_shuffled=x[shuffle_indices]
y_shuffled=y[shuffle_indices]

#从样本中选择训练集与测试集,并打印每个样本的大小，以及训练集与测试集的个数
dev_sample_index=-1*int(FLAGS.dev_sample_percetage*float(len(y)))
x_train,x_dev=x_shuffled[:dev_sample_index],x_shuffled[dev_sample_index:]
y_train,y_dev=y_shuffled[:dev_sample_index],y_shuffled[dev_sample_index:]
print('max_document_length:{}'.format(len(vocab_processor.vocabulary_)))
print('train/dev_split:{}/{}'.format(len(y_train),len(y_dev)))

#设置session的参数，初始化session,设置训练网络,参数为：训练集样本的个数，样本的类别数，中间层矩阵的维度,filter的大小，filter的个数,l2正则项
with tf.Graph().as_default():
    session_conf=tf.ConfigProto(              #session的配置
            allow_soft_placement=FLAGS_allow_soft_placement,    #如果指定的设备不存在(CPU或者GPU),是否自动分配设备
            log_soft_placement=FLAGS_log_soft_placement         #是否打印日志
            )
    sess=tf.Session(config=session_conf)       #将配置加入session
    with sess.as_default():
        cnn=text_cnn.TextCNN(                  #训练模型实例化
                sequence_length=x_train.shape[1],
                num_classes=y_train.shape[1],
                vocab_size=len(vocab_processor.vocabulary_),
                embedding_size=FLAGS.embedding,
                filter_sizes=list(map(int,FLAGS.filter_size.split(','))),
                num_filters=FLAGS.num_filters,
                l2_reg_lamdba=FLAGS.L2_reg_lambda
                )

saver=tf.train.Saver(tf.global_variables(),max_to_keep=FLAGS.num_checkpoints)     #设置保存模型参数的规则(保存模型中的所有变量；max_to_keep表示最大
checkpoint文件数,当一个新文件创建的时候,旧文件就会被删掉。
sess.run(tf.global_variables_initializer())    #session初始化

#用于将初始值赋给网络模型并开始训练
def train_step(x_batch,y_batch):
    feed_dict={           #设置初始值
            cnn.input_x:x_batch,
            cnn.input_y:y_batch,
            cnn.dropout_keep_prob:FLAGS.dropout
            }
    _,step,summaries,loss,accuracy=sess.run([train_op,global_step,train_summary_op,cnn.loss,cnn.accuracy],feed_dict)   #赋值并开始训练
    time_str=datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print('{}:step {},loss {},acc{}'.format(time_str,step,loss,accuracy))    #打印当前时间，步数，损失值，精确度

#用于对训练模型进行验证
def dev_step(x_batch,y_batch):
    feed_dict={           #设置初始值
            cnn.input_x:x_batch,
            cnn.input_y:y_batch,
            cnn.dropout_keep_prob:FLAGS.dropout
            }
    _,step,summaries,loss,accuracy=sess.run([train_op,global_step,train_summary_op,cnn.loss,cnn.accuracy],feed_dict)   #赋值并开始训练
    time_str=datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print('{}:step {},loss {},acc{}'.format(time_str,step,loss,accuracy))    #打印当前时间，步数，损失值，精确度

baches=data_helpers.batch_iter(list(zip(x_train,y_train),FLAGS.batch_size,FLAGS.num_epochs))    #将压缩的训练集分解成若干部分样本，每次选择这若干部
分样本进行训练

#开始训练，并进行验证与保存模型
for batch in batches:
    x_batch,y_batch=zip(*batch)   #将训练集解压缩
    train_step(x_batch,y_batch)   #指定训练集并开始训练
    current_step=tf.train.global_step(sess,global_step)   #获得当前训练执行到的步数
    if current_step %FLAGS.evaluate_every==0:   #每训练100步验证一次
        print('\n evaluate_every')
        dev_step(x_dev,y_dev)     #对训练结果进行验证
    if current_step%checkpoint_every==0:    #每训练100步保存一次模型
        path=saver.save(sess,'./',global_step=current_step)
        print('save sucessfully')





