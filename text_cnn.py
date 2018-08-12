class TextCNN():

    def __init__(self,sequence_length,num_classes,vocab_size,embedding_size,filter_sizes,num_filters,l2_reg_lamdba):
        self.input_x=tf.placeholder(tf.int32,[None,sequence_length],name='input_x')
        self.input_y=tf.placeholder(tf.float32,[None,num_classes],name='input_y')
        self.dropout_keep_prob=tf.placeholder(tf.float32,name='droup_keep_prob')
        l2_loss=tf.constant(0,0)

        #数据层
        with tf.device('/cpu:0'),tf.name_scope('embedding'):
            self.w=tf.Variable(tf.random_uniform([vocab_size,embedding_size],-1.0,1.0,name='w'))   #数据层的权重系数
            self.embedded_chars=tf.nn.embedding_look(self.w,self.input_x)   #取中间层
            self.embedded_chars_expanded=tf.expand_dims(self.embedded_chars,-1)  #增加中间层的维度

        pooled_outputs=[]   #池化层输出

        #每个filter的卷积层、激活层与池化层
        for i,filter_size in enumerate(filter_sizes):
            with tf.name.scope('conv-maxpool-%s'%filter_size):
                filter_shape=[filter_size,embedding,1,num_filters]
                w=tf.Variable(tf.truncated_normal(filter_shape,stddev=0.1),name='w')       #卷积层的权重项
                b=tf.Variable(t.constant(0,1),shape=[num_filters],name='b')                #卷积层的偏置项
                conv=tf.nn.conv2d(self.embedded_chars_expanded,w,strides=[1,1,1,1],padding='VALID',name='conv')    #卷积层
                h=tf.nn.relu(tf.nn.bias_add(conv,b),name='relu')   #激活层
                pooled=tf.nn.max_pool(h,ksize=[1,sequence_length-filter_sizes+1,1,1],stides=[1,1,1,1],padding='VALLD',name='pool')   #池化层
                pooled_outputs.append(pooled)

        num_filters_total=num_filters*len(filter_sizes)
        self.h_pool=tf.concat(3,pooled_outputs)    #将所有池化层的结果连接在一起
        self.h_pool_flat=tf.reshape(self.h_pool,[-1,num_filters_total])   #将矩阵拉平

        #dropout层
        with tf.name.scope('dropout'):
            self.h_drop=tf.nn.dropout(self.h_pool_flat,self.dropout_keep_prob)

        #全连接层与输出层
        with tf.name.scope('output'):
            w=tf.get_variable('w',shape=[num_filters_total,num_classes],initializer=tf.contrib.layers.Xavier_initializer())   #全连接层的权重项
            b=tf.Variable(tf.constant(0,1),shape=[num_classes],name='b')   #全连接层的偏移项
            l2_loss+=tf.nn.l2_loss(w)
            l2_loss+=tf.nn.l2_loss(b)
            self.scores=tf.nn.xw_plus_b(h_drop,w,b,name='score')  #z最终的得分
            self.predictions=tf.argmax(self.scores,1,name='predictions')   #返回最大值的索引

        #损失层
        with tf.name.scop('losss'):
            losses=tf.nn.softmax_cross_entropy(logits=self.scores,labels=self.input_y)   #计算交叉熵
            self.loss=tf.reduce_mean(losses)+l2_loss*l2_reg_lamdba  #计算损失值(交叉熵的平均值与修正项之和)

        #计算准确度层
        with tf.name_scope('accuracy'):
            correct_predictions=tf.equal(self.predictions,tf.argmax(self.input_y),1)       #预测正确返回True,预测错误返回False
            self.accuracy=tf.reduce_mean(tf.cast(correct_predictions,'float'),name='accuracy')   #计算预测准确率(tf.reduce_mean表示求平均值)


