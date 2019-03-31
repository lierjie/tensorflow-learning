# 导入tensorflow库
import  tensorflow as tf
# 导入Numpy工具包,生成一个模拟数据集
from numpy.random import  RandomState

# 采用Mini-batch训练方法,速度快,准确率高
batch_size = 8

# 首先定义两个权重(变量),满足正太随机分布,维度([2,3]),标准差是1, seed使用随机种子,保证每次运行的结果一样
w1 = tf.Variable(tf.random_normal([2,3], stddev=1, seed=1))
# 与上类似,维度([3,1])
w2 = tf.Variable(tf.random_normal([3,1], stddev=1, seed=1))

# 设置x变量,使用placeholder,为变量提供一个位置.若使用variable等格式,每次赋值,都会在计算图中
# 增加一个结点,训练几百万轮,会使得计算图非常复杂,所以引入placeholder机制.
# 维度([None,2]),
x = tf.placeholder(tf.float32, shape=(None, 2), name='x-input')
y_ = tf.placeholder(tf.float32, shape=(None, 1), name='y-input')

# matmul实现矩阵相乘,这里tensorflow使用计算图方式,与numpy的矩阵相乘是不一样的.
a = tf.matmul(x, w1)
y = tf.matmul(a, w2)

# 通过极力函数sigmoid(),实现非线性化
y = tf.sigmoid(y)

# 定义损失函数,此处使用交叉熵,以后讨论,用来表征预测数据和真实数据的误差
cross_entropy = -tf.reduce_mean(y_ * tf.log(tf.clip_by_value(y, 1e-10, 1.0))
                                + (1 - y) * tf.log(tf.clip_by_value(1-y, 1e-10, 1.0)))

# 定义反向传播算法,来优化NN中的参数,此处使用AdamOptimizer
# 学习率:0.001,优化目标:cross_entropy
train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)

# 通过随机数生成模拟数据集
rdm = RandomState(1)

# 训练数据X:数据集大小[128,2]
# 128个样例,2个属性
dataSet_size = 128
X = rdm.rand(dataSet_size, 2)

# 标签数据Y:使用布尔变量,如果x1 + x2 < 1则认为是正样本,也就是为1
# 否则认为是负样本,也就是0.
Y = [[int(x1 + x2 < 1)] for (x1,x2) in X]

# 创建一个回话,来运行计算图
with tf.Session() as sess:
    # 对所有变量进行初始化
    init_op = tf.global_variables_initializer()
    sess.run(init_op)

    # 输出w1和w2值,在训练之前
    print(sess.run(w1))
    print(sess.run(w2))

    # 设置迭代次数为5000次
    Steps = 5000
    # 循环
    for i in range(Steps):
        # 选取batch_size个样本,取余
        start = (i*batch_size)%dataSet_size
        # 避免超出训练集的范围
        end = min(start + batch_size, dataSet_size)

        # 对选取的样本进行训练,因为是placeholder,使用字典对其赋值,并迭代更新参数
        sess.run(train_step, feed_dict={x : X[start:end], y_ : Y[start:end]})

        # 当迭代次数为100的倍数的时候,输出所有数据上的交叉熵
        if i%100 == 0:
            total_cross_entropy = sess.run(cross_entropy, feed_dict={x:X, y_ : Y})
            print("After %d training step(s), cross entropy on all data is %g" % (i, total_cross_entropy))

    # 输出训练过后的参数(权重)
    print(sess.run(w1))
    print(sess.run(w2))
