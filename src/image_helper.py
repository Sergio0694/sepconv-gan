import tensorflow as tf

# reusable graph to compute the difference
diff_graph = None

def diff(path1, path2):
    '''Computes the mean squared error between a pair of images

    path1(str) -- the path of the first image to read
    path2(str) -- the path of the second image to read
    '''

    # initialize the graph if needed, then build it
    global diff_graph
    if not diff_graph:
        graph = tf.Graph()
        with graph.as_default():
            
            # load the input images
            source1 = tf.placeholder(tf.string)
            source2 = tf.placeholder(tf.string)
            contents1 = tf.read_file(source1)
            contents2 = tf.read_file(source2)    
            image1 = tf.image.decode_jpeg(contents1)
            image2 = tf.image.decode_jpeg(contents2)

            # convert to float and calculate the difference
            f1 = tf.image.convert_image_dtype(image1, tf.float32, True)
            f2 = tf.image.convert_image_dtype(image2, tf.float32, True)
            loss = tf.reduce_sum((tf.reshape(f1, [-1]) - tf.reshape(f2, [-1])) ** 2)

            # store the graph for later use
            diff_graph = (graph, source1, source2, loss)
        tf.reset_default_graph()

    # compute and return the difference
    with tf.Session(graph=diff_graph[0]) as sess:
        sess.run(tf.global_variables_initializer())
        return sess.run(diff_graph[3], feed_dict={diff_graph[1]: path1, diff_graph[2]: path2})
