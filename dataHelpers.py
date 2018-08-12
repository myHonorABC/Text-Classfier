import numpy as np
import tensorflow as tf

def load_data_and_labels(positive_data_file,negative_data_file):
    #读入正例样本与负例样本的文本
    positive=open(positive_data_file,'rb').read().decode('utf-8')
    negative=open(negative_data_file,'rb').read().decode('utf-8')
    
    #提取样本的文本中的每段句子
    positive_examples=positive.split('\n')[:-1]    
    negative_examples=negative.split('\n')[:-1]

    #对样本的文本中的每段句子进行strip操作
    positive_examples=[s.strip() for s in positive_examples]
    negative_examples=[s.strip() for s in negative_examples]

    x_text=positive_examples+negative_examples       #将正例样本与负例样本连在一起
    #x_text=[clean_str(sent) for sent in x_text]      #对数据进行清洗

    #设置正例样本与负例样本的标签
    positive_label=[[0,1] for _ in positive_examples]
    negative_label=[[1,0] for _ in negative_examples]
    y= np.concatenate([positive_label,negative_label],0)

    return [x_text,y]


#def clean_str(sent):                #用于清洗数据


def batch_iter(data,batch_size,num_epoch,shuffle=True):
    data=np.array(data)
    data_size=len(data)   #数据的长度（即样本的个数）
    num_batches_per_epoch=int(len(data)-1)/batch_size+1  #每一次训练样本的个数
    for epoch in range(num_epoch):
        if shuffle:    #如果需要打乱样本顺序
            shuffle_indices=np.random.permutation(np.arrange(data_size))      #将从0到data_size的索引打乱顺序
            shuffle_data=data[shuffle_indices]   #返回打乱后的数据
        else:     #如果不需要打乱样本顺序
            shuffle_data=data
        for batch_num in range(num_batches_per_epoch):
            start_index=batch_num*batch_size  #每一次的batch_size大小的样本的开始索引值
            end_index=min((batch_num+1)*batch_size,data_size)   #每一次选择的batch_size大小的样本的结束索引值
            yield shuffle_data[start_index:end_index]    #返回每次训练样本的索引


if __name__=='__main__':
    x_text,y=load_data_and_labels('./positiveData.txt','./negativeData.txt')
    #print(x_text,'\n',y)
    #x_split=[x.split(' ') for x in x_text]
    #print(x_split)
    max_length=max(len(x.split(' ')) for x in x_text)
    #print(max_length)
    vocab_processor=tf.contrib.learn.preprocessing.text.VocabularyProcessor(max_length)
    #print(vocab_processor)
    x=np.array(list(vocab_processor.fit_transform(x_text)))
    #print(x)
    np.random.seed(10)
    shuffle_indices=np.random.permutation(np.arange(len(y)))
    x_shuffled=x[shuffle_indices]
    y_shuffled=y[shuffle_indices]
    print(x_shuffled,'\n')
    print(x_shuffled.shape[1])


