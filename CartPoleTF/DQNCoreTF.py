"""
This part of code is the DQN brain, which is a brain of the agent.
All decisions are made in here.
Using Tensorflow to build the neural network.
View more on tutorial page: https://morvanzhou.github.io/tutorials/
Using:
Tensorflow: 1.0
gym: 0.8.0
"""

import numpy as np
import tensorflow as tf


# Deep Q Network off-policy
class DeepQNetwork:
    def __init__(
            self,
            n_actions,  # 2
            n_features,  # 4
            learning_rate=0.01,
            reward_decay=0.9,
            e_greedy=0.9,
            replace_target_iter=300,
            memory_size=500,
            batch_size=32,
            e_greedy_increment=None,
            output_graph=True,
    ):
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max

        # total learning step
        self.learn_step_counter = 0

        # initialize zero memory [s, a, r, s_]
        self.memory = np.zeros((self.memory_size, n_features * 2 + 2))  # 2个s，1个a，1个r

        # consist of [target_net, evaluate_net]
        self._build_net()
        t_params = tf.get_collection('target_net_params')
        e_params = tf.get_collection('eval_net_params')
        # 以下两行是第一次运行时出现的结果
        # print('t_params', t_params)  # t_params [<tf.Variable 'target_net/l1/w1:0' shape=(4, 10) dtype=float32_ref>, <tf.Variable 'target_net/l1/b1:0' shape=(1, 10) dtype=float32_ref>, <tf.Variable 'target_net/l2/w2:0' shape=(10, 2) dtype=float32_ref>, <tf.Variable 'target_net/l2/b2:0' shape=(1, 2) dtype=float32_ref>]
        # t_params是一个长度为4的list，
        # ['target_net/l1/w1:0' shape=(4, 10),
        # 'target_net/l1/b1:0' shape=(1, 10),
        # 'target_net/l2/w2:0' shape=(10, 2),
        # 'target_net/l2/b2:0' shape=(1, 2)]
        self.replace_target_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]

        self.sess = tf.Session()

        if output_graph:
            # $ tensorboard --logdir=logs
            # tf.train.SummaryWriter soon be deprecated, use following
            tf.summary.FileWriter("logs/", self.sess.graph)

        self.sess.run(tf.global_variables_initializer())
        self.cost_his = []

    def _build_net(self):
        '''
        建立2个网络，
        :return:
        '''
        # ------------------ build evaluate_net ------------------
        self.s = tf.placeholder(tf.float32, [None, self.n_features], name='s')  # input
        self.q_target = tf.placeholder(tf.float32, [None, self.n_actions], name='Q_target')  # for calculating loss
        with tf.variable_scope('eval_net'):
            # c_names(collections_names) are the collections to store variables
            c_names, n_l1, w_initializer, b_initializer = \
                ['eval_net_params', tf.GraphKeys.GLOBAL_VARIABLES], 10, \
                tf.random_normal_initializer(0., 0.3), tf.constant_initializer(0.1)  # config of layers,只是一个自定义的列表而已

            # first layer. collections is used later when assign to target net
            with tf.variable_scope('l1'):
                w1 = tf.get_variable('w1', [self.n_features, n_l1], initializer=w_initializer, collections=c_names)
                b1 = tf.get_variable('b1', [1, n_l1], initializer=b_initializer, collections=c_names)
                l1 = tf.nn.relu(tf.matmul(self.s, w1) + b1)

            # second layer. collections is used later when assign to target net
            with tf.variable_scope('l2'):
                w2 = tf.get_variable('w2', [n_l1, self.n_actions], initializer=w_initializer, collections=c_names)
                b2 = tf.get_variable('b2', [1, self.n_actions], initializer=b_initializer, collections=c_names)
                self.q_eval = tf.matmul(l1, w2) + b2  # shape=[?,2]
            # q_eval是评估网络的输出，数据传输关系是q_eval→l1→s，所以feed_dict的第二个键名是s

        with tf.variable_scope('loss'):
            self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval))
        with tf.variable_scope('train'):
            self._train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)

        # ------------------ build target_net ------------------
        self.s_ = tf.placeholder(tf.float32, [None, self.n_features], name='s_')  # input
        with tf.variable_scope('target_net'):
            # c_names(collections_names) are the collections to store variables
            c_names = ['target_net_params', tf.GraphKeys.GLOBAL_VARIABLES]

            # first layer. collections is used later when assign to target net
            with tf.variable_scope('l1'):
                w1 = tf.get_variable('w1', [self.n_features, n_l1], initializer=w_initializer, collections=c_names)
                b1 = tf.get_variable('b1', [1, n_l1], initializer=b_initializer, collections=c_names)
                l1 = tf.nn.relu(tf.matmul(self.s_, w1) + b1)

            # second layer. collections is used later when assign to target net
            with tf.variable_scope('l2'):
                w2 = tf.get_variable('w2', [n_l1, self.n_actions], initializer=w_initializer, collections=c_names)
                b2 = tf.get_variable('b2', [1, self.n_actions], initializer=b_initializer, collections=c_names)
                self.q_next = tf.matmul(l1, w2) + b2  # 输出数据类型是Tensor
            # q_next是目标网络的输出，数据传输关系是q_next→l1→s_，所以feed_dict的第一个键名是s_

    def store_transition(self, s, a, r, s_):  # 每个episode的每个step，都会调用该函数
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0

        transition = np.hstack((s, [a, r], s_))

        # replace the old memory with new memory
        index = self.memory_counter % self.memory_size  # 取余
        self.memory[index,
        :] = transition  # self.memory = np.zeros((self.memory_size, n_features * 2 + 2))  # 2个s，1个a，1个r

        self.memory_counter += 1

    def choose_action(self, observation):  # observation.shape=(4,)
        # to have batch dimension when feed into tf placeholder
        observation = observation[np.newaxis, :]  # observation.shape=(1,4)

        if np.random.uniform() < self.epsilon:  # self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max
            # forward feed the observation and get q value for every actions
            actions_value = self.sess.run(self.q_eval, feed_dict={self.s: observation})  # [[2.1524513 2.2836583]]
            action = np.argmax(actions_value)  # 从形如[[2.1524513 2.2836583]]中选取大的那个数的索引,即[0,2)
        else:
            action = np.random.randint(0, self.n_actions)  # 2,即从[0,2)中随机选取整数
        # print(type(action))
        # action的数据类型，有时候是int，有时候是numpy.int64，程序还不够严谨，需要改善
        return action  # 0或者1

    def learn(self):  # if total_steps > START_POINT:就执行该函数
        # 检查是否需要更新目标网络参数
        # self.learn_step_counter在该函数末尾self.learn_step_counter += 1
        if self.learn_step_counter % self.replace_target_iter == 0:  # 取余，按self.replace_target_iter整数倍更换target net的参数
            self.sess.run(
                self.replace_target_op)  # self.replace_target_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]
            print('target_params_replaced，第', self.learn_step_counter / self.replace_target_iter, '次')

        # sample batch memory from all memory
        if self.memory_counter > self.memory_size:  # memory_counter在store_transition中，每个step都会增加1
            # 在[0, self.memory_size)数组中，抽取size为batch_size的数组作为索引
            sample_index = np.random.choice(self.memory_size,
                                            size=self.batch_size)  # 原文代码中是没有replace参数的，这是一种可重复抽样，个人感觉应该是不可重复抽样比较好些，为啥我也说不清，要实现不重复抽样，应该添加参数replace=False
        else:
            # 在[0, memory_counter)数组中，抽取size为batch_size的数组作为索引
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)  # 同上，
        # memory.shape=(xxx,yyy)，从中按索引sample_index抽取若干行，结果batch_memory的shape的维度不变，还是2
        # batch_memory.shape=(32,10)
        batch_memory = self.memory[sample_index,
                       :]  # self.memory = np.zeros((self.memory_size, n_features * 2 + 2))  # 2个s，1个a，1个r
        q_next, q_eval = self.sess.run(  # type(q_next)=ndarray, q_next.shape (32, 2), 32是batch_size
            # q_next是目标网络的输出，数据传输关系是q_next→l1→s_，所以feed_dict的第一个键名是s_
            # q_eval是评估网络的输出，数据传输关系是q_eval→l1→s，所以feed_dict的第二个键名是s
            [self.q_next, self.q_eval],  # 分别是2个网络输出，数据类型是Tensor
            feed_dict={
                # 记忆库的顺序是(s, a, r, s_)       store_transition(self, s, a, r, s_):
                # n_features=4。[:, -self.n_features:]意思是所有行的  [倒数第4个元素, 最后一个元素]
                # n_features=4。[:, :self.n_features]意思是所有行的  [第1个元素, 第4个元素]
                self.s_: batch_memory[:, -self.n_features:],
                # fixed params。  feed_dict键名是tf.placeholder:键值是numpy ndarray
                self.s: batch_memory[:, :self.n_features],  # newest params
            })

        # change q_target w.r.t q_eval's action     这部分代码解释见下面注释
        q_target = q_eval.copy()  # q_eval的数据类型是ndarray，该操作可以理解为深复制，也即c语言传统意义上的复制一个值
        batch_index = np.arange(self.batch_size, dtype=np.int32)  # [0, 1, 2, 3,...,32]的一维数组
        # [:, self.n_features]是所有行的第4个元素，也即action
        # 不加astype(int)时，输出的是浮点型数据
        # 以下两行代码，实际上是把batche_memory中的action和reward抽取了出来，形成一个一维数组（32，）
        eval_act_index = batch_memory[:, self.n_features].astype(
            int)  # batch_memory.shape=(32,10)  # eval_act_index.shape (32,)    由0和1组成
        reward = batch_memory[:, self.n_features + 1]
        # 这一句是算法的关键
        # q_target.shape=(32,2)
        # batch_index.shape=(32,)
        # eval_act_index.shape=(32,)
        # q_next.shape=(32,2)
        # np.max(q_next, axis=1).shape=(32,)
        q_target[batch_index, eval_act_index] = reward + self.gamma * np.max(q_next, axis=1)
        '''
        上面几句代码的原理，要从网络结构和DQN算法来解释。
        由简至繁解释一下
        ①最简单的情况下，网络输入为state和action，输出为Q(s,a)/Q(s_,a)
        如Q(s,a)=1.1，maxQ(s_,a)=3.2，r=0.9
        则根据DQN的伪代码，target是0.9+3.2=4.1
        网络更新，就是要更新Q(s,a)=1.1向4.1靠近
        ②更改网络结构，网络输入为state，输出为Q(s,a1)，Q(s,a2)等
        如q_eval输出为[1.2, 5,8]，并且记忆库告诉我们，选择的action是a2（若是e-greedy起作用，也可能选择a1)，q_next输出为[4.6, 1.9]，r=0.9
        则根据DQN的伪代码，target是0.9+4.6=5.5
        网络更新，就是要更新Q(s,a2)=5.8向5.5靠近，而Q(s,a1)是不！需！要！动的。所以，写成矩阵形式，target就是[1.2, 5.5]
        试想一下，如果上面的描述action换成是a1，其他不变，那么target就是[5.5, 5.8]

        这就是上边代码为何要如此操作的原因——仅仅改变需要改变的Q(s,a)，更换为target，其余的不变。    
        下面的是原始代码的解释：
        '''

        """
        For example in this batch I have 2 samples and 3 actions:
        q_eval =
        [[1, 2, 3],
         [4, 5, 6]]
        q_target = q_eval =
        [[1, 2, 3],
         [4, 5, 6]]
        Then change q_target with the real q_target value w.r.t the q_eval's action.
        For example in:
            sample 0, I took action 0, and the max q_target value is -1;
            sample 1, I took action 2, and the max q_target value is -2:
        q_target =
        [[-1, 2, 3],
         [4, 5, -2]]
        So the (q_target - q_eval) becomes:
        [[(-1)-(1), 0, 0],
         [0, 0, (-2)-(6)]]
        We then backpropagate this error w.r.t the corresponding action to network,
        leave other action as error=0 cause we didn't choose it.
        """

        # train eval network
        # _train_op数据传输关系是_train_op→loss→(q_target和q_eval)
        # q_eval是评估网络的输出，数据传输关系是q_eval→l1→s
        _, self.cost = self.sess.run([self._train_op, self.loss],
                                     feed_dict={self.s: batch_memory[:, :self.n_features],
                                                self.q_target: q_target})
        self.cost_his.append(self.cost)

        # increasing epsilon
        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1

    def plot_cost(self):
        import matplotlib.pyplot as plt
        plt.plot(np.arange(len(self.cost_his)), self.cost_his)
        plt.ylabel('Cost')
        plt.xlabel('training steps')
        plt.show()
