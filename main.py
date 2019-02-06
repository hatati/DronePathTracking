#import fire
import numpy as np
import data_import as di
import os
import tensorflow as tf


class Train:
    __x_ = []
    __y_ = []
    __logits = []
    __loss = []
    __train_step = []
    __merged_summary_op = []
    __saver = []
    __session = []
    __writer = []
    __is_training = []
    __loss_val = []
    __train_summary = []
    __val_summary = []

    def __init__(self):
        pass

    def build_graph(self):
        # First, we start the function by defining the placeholders for our inputs.
        self.__x_ = tf.placeholder("float", shape=[None, 60, 60, 3], name='X')
        self.__y_ = tf.placeholder("int32", shape=[None, 3], name='Y')
        self.__is_training = tf.placeholder(tf.bool)

        # We will define our network within the name_scope model. Name_scope returns a
        # context manager for use when defining TensorFlow ops. This context manager validates
        # that the variables are from the same graph, makes that graph the default graph, and pushes
        # a name scope in that graph.
        with tf.name_scope("model") as scope:
            conv1 = tf.layers.conv2d(inputs=self.__x_, filters=64,
                                     kernel_size=[5, 5],
                                     padding="same", activation=tf.nn.relu)
            pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

            conv2 = tf.layers.conv2d(inputs=pool1, filters=64, kernel_size=[5, 5], padding="same",
                                     activation=tf.nn.relu)

            pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

            conv3 = tf.layers.conv2d(inputs=pool2, filters=32, kernel_size=[5, 5], padding="same",
                                     activation=tf.nn.relu)

            pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[2, 2], strides=2)

            pool3_flat = tf.reshape(pool3, [-1, 7 * 7 * 32])

            # FC layers
            FC1 = tf.layers.dense(inputs=pool3_flat, units=128, activation=tf.nn.relu)
            FC2 = tf.layers.dense(inputs=FC1, units=64, activation=tf.nn.relu)
            self.__logits = tf.layers.dense(inputs=FC2, units=3)

            # The loss function within the name scope loss_func. The loss
            # function that is used here is the softmax cross entropy and, we
            # average the loss across the whole batch with tf.reduce_mean. We create variables to hold
            # training loss __loss and validation loss __loss_val, and add these scalars to the
            # TensorFlow summary data to display in TensorBoard later
            with tf.name_scope("loss_func") as scope:
                self.__loss = tf.reduce_mean(
                    tf.nn.softmax_cross_entropy_with_logits(logits=self.__logits, labels=self.__y_))
                self.__loss_val = tf.reduce_mean(
                    tf.nn.softmax_cross_entropy_with_logits(logits=self.__logits, labels=self.__y_))

                # Add loss to tensorboard
                self.__train_summary = tf.summary.scalar("loss_train", self.__loss)
                self.__val_summary = tf.summary.scalar("loss_val", self.__loss_val)

            # For this particular model, we will use the second approach of an exponentially decaying learning rate.
            # We use the TensorFlow operation—tf.train.exponential_decay. As input, it takes the current learning rate,
            # the global step, the amount of steps before decaying and a decay rate.
            #
            # At every iteration, the current learning rate is supplied to our Adam Optimizer, which uses
            # the minimize function that uses gradient descent to minimize the loss and increases the
            # global_step variable by one. Lastly, learning_rate and global_step are added to the
            # summary data to be displayed on TensorBoard during training:
            with tf.name_scope("optimizer") as scope:
                global_step = tf.Variable(0, trainable=False)
                starter_learning_rate = 1e-3
                # decay every 10000 steps with a base of 0.96 function
                learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step, 1000, 0.9,
                                                           staircase=True)
                self.__train_step = tf.train.AdamOptimizer(learning_rate).minimize(self.__loss, global_step=global_step)
                tf.summary.scalar("learning_rate", learning_rate)
                tf.summary.scalar("global_step", global_step)


            # Once all the components of the graph have been defined, all the summaries collected in the
            # graph are merged into __merged_summary_op, and all the variables of the graph are
            # initialized by tf.global_variables_initializer().
            #
            # Naturally, while training the model, we want to store the network weights as binary files so
            # that we can load them back to perform forward propagation. Those binary files in
            # TensorFlow are called checkpoints, and they map variable names to tensor values. To save
            # and restore variables to and from checkpoints, we use the Saver class. To avoid filling up
            # disks, savers manage checkpoint files automatically. For example, they can keep only the N
            # most recent files or one checkpoint for every N hours of training. In our case, we have set
            # max_to_keep to None, which means all checkpoint files are kept

            # Merge op for tensorboard
            self.__merged_summary_op = tf.summary.merge_all()
            # Build graph
            init = tf.global_variables_initializer()
            # Saver for checkpoints
            self.__saver = tf.train.Saver(max_to_keep=None)

            # After creating the FileWriter object to store the
            # summaries and events to a file, the __session.run(init) method runs one step of
            # TensorFlow computation, by running the necessary graph fragment to execute every
            # operation and evaluate every tensor that was initialized in init as parts of the graph:
            # Configure summary to output at given directory
            self.__session = tf.Session()
            self.__writer = tf.summary.FileWriter("./logs/flight_path", self.__session.graph)
            self.__session.run(init)



    def train(self, save_dir='./model_files', batch_size=20):
        # Now that we have a dataset set up, we can use tf.data.Iterators to iterate over and
        # extract elements from it. We use the one shot iterator. This iterator supports going
        # through a dataset just once, but it's super simple to set up. We create it by calling the
        # make_one_shot_iterator() method on our dataset and assigning the result to a
        # variable. We can then call get_next() on our created iterator and assign it to another
        # variable. Now, whenever this op is run in a session, we will iterate once through the dataset, and a
        # new batch will be extracted to use:

        #Load dataset and labels
        x = np.asarray(di.load_images())
        y = np.asarray(di.load_labels())

        #Shuffle dataset
        np.random.seed(0)
        shuffled_indeces = np.arange(len(y))
        np.random.shuffle(shuffled_indeces)
        shuffled_x = x[shuffled_indeces].tolist()
        shuffled_y = y[shuffled_indeces].tolist()
        shuffled_y = tf.keras.utils.to_categorical(shuffled_y, 3)

        dataset = (shuffled_x, shuffled_y)
        dataset = tf.data.Dataset.from_tensor_slices(dataset)
        #dataset = dataset.shuffle(buffer_size=300)

        # Using Tensorflow data Api to handle batches
        dataset_train = dataset.take(200)
        dataset_train = dataset_train.repeat()
        dataset_train = dataset_train.batch(batch_size)

        dataset_test = dataset.skip(200)
        dataset_test = dataset_test.repeat()
        dataset_test = dataset_test.batch(batch_size)

        # Create an iterator
        iter_train = dataset_train.make_one_shot_iterator()
        iter_train_op = iter_train.get_next()
        iter_test = dataset_test.make_one_shot_iterator()
        iter_test_op = iter_test.get_next()

        # Build model graph
        self.build_graph()

        # Once we have retrieved the data and built the graph, we can start our main training loop,
        # which will continue over 20,000 iterations. In every iteration, a batch of training data is
        # taken using the CPU device, and the __train_step.run method of the AdamOptimizer
        # object is called to run one forward and one backward pass. Every 100 iterations, we run a
        # forward pass over the current training and testing batch to collect training and validation
        # loss, and other summary data. Then, the add_summary method of the FileWriter object
        # wraps the provided TensorFlow summaries: summary_1 and summary_2 in an event
        # protocol buffer and adds it to the event file:

        # Train Loop
        for i in range(3000):
            batch_train = self.__session.run([iter_train_op])
            batch_x_train, batch_y_train = batch_train[0]
            # Print loss from time to time
            if i % 100 == 0:
                batch_test = self.__session.run([iter_test_op])
                batch_x_test, batch_y_test = batch_test[0]
                loss_train, summary_1 = self.__session.run([self.__loss,
                                                        self.__merged_summary_op],
                                                       feed_dict={self.__x_:
                                                                      batch_x_train,
                                                                  self.__y_:
                                                                      batch_y_train,
                                                                  self.__is_training: True})
                loss_val, summary_2 = self.__session.run([self.__loss_val,
                                                  self.__val_summary],
                                                 feed_dict={self.__x_: batch_x_test,
                                                            self.__y_: batch_y_test,
                                                            self.__is_training: False})
                print("Loss Train: {0} Loss Val: {1}".format(loss_train,
                                                     loss_val))
                # Write to tensorboard summary
                self.__writer.add_summary(summary_1, i)
                self.__writer.add_summary(summary_2, i)

            # Execute train op
            self.__train_step.run(session=self.__session, feed_dict={
                self.__x_: batch_x_train, self.__y_: batch_y_train,
                self.__is_training: True})
            print(i)


        # Once the training loop is over, we store the final model into a checkpoint file with op
        # __saver.save:

        # Save model
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            checkpoint_path = os.path.join(save_dir, "model.ckpt")
            filename = self.__saver.save(self.__session, checkpoint_path)
            tf.train.write_graph(self.__session.graph_def, save_dir, "save_graph.pbtxt")
            print("Model saved in file: %s" % filename)


if __name__ == '__main__':
    cnn = Train()
    cnn.train()
