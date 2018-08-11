# -*- coding:utf-8 -*-
#This file use to store some tool function in AutoEncoder
#Author: TomRen
#E-mail: jiejiren@sjtu.edu.cn



#write data into .tfrecords format
def getRecords(path,classes,record_name):
    """write imgs into .tfrecords"""
    #getRecords('path/your/store/imgs',{'class1','class2'},'path/your/output/xxx.tfrecords')
    
    '''params discribe:
    
    #path:  is the folder where you store your imgs, which may contain many folder with different classes
    e.g: path = 'surface/'    #where you store your data
    
    #clsses:    is a list, elements is string, which means differnet sub-folders' name with different classer
    e.g: classes = {'img_0'}    #setting the classes
    
    #record_name:   the out put records file,with it locations
    e.g: record_name = "yourfoler/path/myrecord.tfrecords" also could be record_name = path+"myrecords.tfrecords" 
    
    
    '''
    
    writer= tf.python_io.TFRecordWriter(record_name) #initialize a writer to record
     
    for index,name in enumerate(classes):
        #class_path=path+name+'/'
        class_path = name+'/'
        for img_name in os.listdir(class_path): 
            if img_name.endswith('.jpg'):
                #print(img_name)
                img_path=class_path+img_name #每一个图片的地址
                img=cv2.imread(img_path,0)
                #print(img.shape)
                #img= img.resize((28,28))
                #print(type(img))
                img_raw=img.tobytes()#将图片转化为二进制格式
                example = tf.train.Example(features=tf.train.Features(feature={
                "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[index])),
                'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
                })) #example对象对label和image数据进行封装
                writer.write(example.SerializeToString())  #序列化为字符串

    writer.close()
    
    #reference
    #https://www.2cto.com/kf/201702/604326.html


    
#process records data in read functions
def parser(record):
    features = tf.parse_single_example(record,
                                       features={
                                           'label': tf.FixedLenFeature([], tf.int64),
                                           'img_raw': tf.FixedLenFeature([], tf.string),
                                       })  # return image and label
                                       
    image = tf.decode_raw(features['img_raw'], tf.uint8)
    img = tf.reshape(image, [28, 28, 1])
    img = tf.cast(img, tf.float32) * (1. / 255)   #normalized
    #img = tf.image.per_image_standardization(img)  #standarize:(x - mean) / adjusted_stddev
    #this is not normalization, but standardazation.https://blog.csdn.net/sinat_21585785/article/details/74251563
    #normalization is very important for loss converge
    img = tf.cast(img, tf.float32)
    '''
    # random augmentation
    image_size = 28
    height = image_size
    width = image_size
    # Randomly crop a [height, width] section of the image.随机裁剪
    distorted_image = tf.random_crop(img, [height, width, 1])
    # Randomly flip the image horizontally.随机翻转
    distorted_image = tf.image.random_flip_left_right \
        (distorted_image)
    # 改变亮度
    distorted_image = tf.image.random_brightness(distorted_image,
                                                 max_delta=63)
    # 改变对比度
    distorted_image = tf.image.random_contrast(distorted_image,
                                               lower=0.2, upper=1.8)
    img = tf.image.per_image_standardization(distorted_image)
    '''
    label = tf.cast(features['label'], tf.int32)
    return img,label 



#read .tfrecords into system    
def read_and_decode(file_name,batch_size = 32,shuffle=True):
    '''read tfrecords file as iter into system'''
    
    '''
    params:
    file_name:  The tfecords path and name
    batch_size: Generate a train batch, with imgs num
    shuffer:    Means random shuffer data order
    '''
    
    dataset = tf.data.TFRecordDataset(file_name)
    if shuffle:
        dataset = dataset.map(parser).repeat().batch(batch_size).shuffle(buffer_size=1000)
    else:
        dataset = dataset.map(parser).repeat().batch(batch_size)

    iterator = dataset.make_one_shot_iterator() #iterator, next()

    img_input, label = iterator.get_next()
    return img_input,label
    
    #reference:
    #https://blog.csdn.net/qwe2508/article/details/80680640
    
    
    
# define neural network
def getNet():
    """define Net work, cost, optimizer"""  
    

    ######################Begin NET defination########################
    # define strctures and wright
    # 用字典的方式存储各隐藏层的参数
    n_input = 784       # input dimension
    n_hidden_1 = 256    # 第一编码层神经元个数
    n_hidden_2 = 64     # 第二编码层神经元个数
    n_hidden_3 = 32     # 第三编码层神经元个数

    # 权重和偏置的变化在编码层和解码层顺序是相逆的
    # 权重参数矩阵维度是每层的 输入*输出，偏置参数维度取决于输出层的单元数
    weights = {
        'encoder_h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
        'encoder_h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
        'encoder_h3': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_3])),
        'decoder_h1': tf.Variable(tf.random_normal([n_hidden_3, n_hidden_2])),
        'decoder_h2': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_1])),
        'decoder_h3': tf.Variable(tf.random_normal([n_hidden_1, n_input])),
    }
    biases = {
        'encoder_b1': tf.Variable(tf.random_normal([n_hidden_1])),
        'encoder_b2': tf.Variable(tf.random_normal([n_hidden_2])),
        'encoder_b3': tf.Variable(tf.random_normal([n_hidden_3])),
        'decoder_b1': tf.Variable(tf.random_normal([n_hidden_2])),
        'decoder_b2': tf.Variable(tf.random_normal([n_hidden_1])),
        'decoder_b3': tf.Variable(tf.random_normal([n_input])),
    }
    
    #reference：
    #https://blog.csdn.net/marsjhao/article/details/68950697
    #tf_book:https://blog.csdn.net/u012421976/article/details/72870512
    
    #define network graph
    # 每一层结构都是 xW + b
    # 构建编码器
    def encoder(x):
        layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['encoder_h1']),
                                       biases['encoder_b1']))
        layer_1 = tf.nn.dropout(layer_1,keep_prob=0.9)    #dropout = 0.9
        layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['encoder_h2']),
                                       biases['encoder_b2']))
        layer_3 = tf.nn.sigmoid(tf.add(tf.matmul(layer_2, weights['encoder_h3']),
                                       biases['encoder_b3']))
        return layer_3

    # 构建解码器
    def decoder(x):
        layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['decoder_h1']),
                                       biases['decoder_b1']))
        #print(layer_1.get_shape())
        layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['decoder_h2']),
                                       biases['decoder_b2']))
        #print(layer_2.get_shape())
        #print(weights["decoder_h3"].get_shape())
        layer_3 = tf.nn.sigmoid(tf.add(tf.matmul(layer_2, weights['decoder_h3']),
                                       biases['decoder_b3']))
        #print(layer_3.get_shape())
        return layer_3
    ####################finish NET defination######################
    
    # tf Graph input (only pictures)
    X = tf.placeholder("float", [None, n_input])    #设置占位符
    keep_prob = tf.placeholder(tf.float32)          #set dropout prob
    
    
    # set caculation graph
    # 定义计算图模型
    encoder_op = encoder(X)             #编码过程
    decoder_op = decoder(encoder_op)    #解码过程
     
    # Prediction
    # 预测
    y_pred = decoder_op                 #forward
    y_true = X                          #for cost caculation
    
    
    # Cost func and optimizer
    # 定义代价函数和优化器 
    current_iter = tf.Variable(0)    #设置优化器学习率衰减
    decay_steps = 30                 #每一步衰减一次,if too small to can't stable
    decay_rate = 0.997               #衰减率
    lr = tf.train.exponential_decay(learning_rate,current_iter,decay_steps,decay_rate) #for decay
    cost = tf.reduce_mean(tf.pow(y_true - y_pred, 2)) #损失函数  最小二乘法 y_pred内已经包含了模型
    optimizer = tf.train.AdamOptimizer(lr).minimize(cost,current_iter)
    
    #return the placeholder, opt and cost
    return X, optimizer, cost


#define training process
def trainAE(records_name, X, opt, cost):
    """define the training process"""


    #调用时需要设置设置网络训练参数
    #define tainning meta-parameters
    learning_rate = 0.001   #   learning rete
    training_epochs = 10001 # total train epochs
    batch_size = 256        # set in the reader_decoder
    display_step = 100      # display cost every num steps
    examples_to_show = 10   # predict results num
    total_cost = []         # store cost in eppchs


    #start a session to run process
    with tf.Session() as sess:
        #with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)) as sess:
        #with tf.device("/gpu:0"):  #GPU 没用？？？
        # tf.initialize_all_variables() no long valid from
        # 2017-03-02 if using tensorflow >= 0.12
        # 加上下面一行就可以使用 个gpu了
        #config = tf.ConfigProto(allow_soft_placement=True)
        # 这一行设置 gpu 随使用增长
        #config.gpu_options.allow_growth = True   #？数据上传GPU~~

        if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
            init = tf.initialize_all_variables()
        else:
            init = tf.global_variables_initializer()
        sess.run(init)

        # 首先计算总批数，保证每次循环训练集中的每个样本都参与训练，不同于批量训练
        total_batch = int(60000/256) #总批数
        images_train, labels_train = read_and_decode(records_name,batch_size)  #read .tfrecods into
        
        print('Training is begining~~~~~~~~~~')
        for epoch in range(training_epochs):  #training_epochs
            #current_iter +=1    #change into if
            #print("Current learning rate is",lr.eval(session=sess))
            for i in range(total_batch):     #total batch of dataset
                batch_xs = sess.run(images_train)
                batch_xs = batch_xs.reshape((-1,784))
                #print(batch_xs[0].reshape((28,28)).shape)
                #plt.imshow(np.reshape(batch_xs[0],(28,28)))
                #break
                #batch_xs, batch_ys = mnist.train.next_batch(batch_size)  # max(x) = 1, min(x) = 0
                # Run optimization op (backprop) and cost op (to get loss value)
                #_, c = sess.run([optimizer, cost], feed_dict={X: batch_xs,keep_prob:0.5})
                _, c = sess.run([optimizer, cost], feed_dict={X: batch_xs, current_iter: epoch})
            if epoch % display_step == 0:
                print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(c))
                #learning_rate *=.1 
            #current_iter +=1    #learning rate decay
            total_cost.append(c)    #log cost change
        plt.plot(total_cost)
        plt.show();plt.title('AutoEncoder Cost Change');plt.xlabel('Epochs');plt.ylabel("Cost")
        print("Optimization Finished!")


        '''
        ####################################################################################
        #测试和计算结果，这里馈入数据十个
        batch_xs = sess.run(images_train)
        batch_xs = batch_xs.reshape((-1,784))
        #encode_decode = sess.run(y_pred, feed_dict={X: batch_xs,keep_prob:1})
        encode_decode = sess.run(y_pred, feed_dict={X: batch_xs})    
        #print(encode_decode.shape)
        #计算均方根值
        ERR = 0    #图与图之间的整体误差
        for i in range(10):
            err_img = batch_xs[i]-encode_decode[i]
            #print('mnist',mnist.test.images[i].max())    #max is 1.0
            #print('endecoder',encode_decode[i].mean())
            #print('endecoder',encode_decode[i].max())
            #print('err_max',err_img.max())
            err_img = err_img**2
            err_sqrt = np.sqrt(err_img.sum()/n_input)
            #print("err is",err_sqrt)

            #计算总误差
            ERR = err_sqrt + ERR
            RMSE = ERR/(100*examples_to_show)

        print("RMSE is ",RMSE)    #计算平均RMES


        
        #选出十个指标用于显示
        f, a = plt.subplots(3, 10, figsize=(10, 3))
        f.suptitle('h1 = %d,h2=%d,h3=%d,epochs is %d, RMSE is %f'%(n_hidden_1,n_hidden_2,n_hidden_3,training_epochs,RMSE), fontsize=16)
        for i in range(examples_to_show):
            a[0][i].imshow(np.reshape(batch_xs[i], (28, 28)))
            a[1][i].imshow(np.reshape(encode_decode[i], (28, 28)))
            a[2][i].imshow(np.reshape(np.abs((batch_xs[i]-encode_decode[i])),(28,28)))
            print('PV values is',(batch_xs[i]-encode_decode[i]).max())
        plt.savefig("%d_%d_%d_001_%d_RMSE.png"%(n_hidden_1,n_hidden_2,n_hidden_3,training_epochs))    #要在show前调用
        plt.show()
        '''
        
        
        

        
        

