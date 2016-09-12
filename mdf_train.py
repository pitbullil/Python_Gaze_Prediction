import sys
sys.path.insert(0,'./tensorflow-fcn')
import mdfgraph as mdf
import numpy as np
import tensorflow as tf
import mdf_preprocessing as mdp
from skimage import io as sio
import time
from six.moves import xrange # pylint: disable=redefined-builtin

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('max_steps', 10000,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_integer('image_per_batch', 5,
                            """Number of batches to run.""")

tf.app.flags.DEFINE_boolean('log_device_placement', False,
"""Whether to log device placement.""")
tf.app.flags.DEFINE_string('train_dir', '/tmp/s3cnn_train',
                           """Directory where to write event logs """
"""and checkpoint.""")

def do_eval(sess,
            eval_correct,
            mean,
            sp_in, nn_in,pic_in,
            labels_in,
            seg_dir,
            img_dir,
            data_set):
    """Runs one evaluation against the full epoch of data.
    Args:
        sess: The session in which the model has been trained.
        eval_correct: The Tensor that returns the number of correct predictions.
        images_placeholder: The images placeholder.
        labels_placeholder: The labels placeholder.
        data_set: The set of images and labels to evaluate, from
        input_data.read_data_sets().
    """
    # And run one epoch of eval.
    true_count = 0  # Counts the number of correct predictions.
    steps_per_epoch = data_set.__len__() // FLAGS.image_per_batch
    num_examples = 0
    for step in xrange(steps_per_epoch):
        for batch in range(0,data_set.__len__()/FLAGS.image_per_batch):
            for i in data_set[batch*FLAGS.image_per_batch:(batch+1)*FLAGS.image_per_batch]:
                segdata = mdp.trainable_segmentations_from_batch(np.load(seg_dir+i[:-3]+'npy').item())
                image = sio.imread(img_dir+i)
                for k in list(segdata.keys()):
                    seg = segdata[k]
                    num_examples = num_examples+seg['seglist'].__len__()
                    mdfin = mdp.im2mdfin2(image,mean,seg['segmap'],seg['seglist'])
                    sp = np.reshape(np.ravel(mdfin[0:mdfin.__len__():3]),[np.uint16(mdfin.__len__()/3),227,227,3])
                    nn = np.reshape(np.ravel(mdfin[1:mdfin.__len__():3]),[np.uint16(mdfin.__len__()/3),227,227,3])
                    pic = np.reshape(np.ravel(mdfin[2:mdfin.__len__():3]),[np.uint16(mdfin.__len__()/3),227,227,3])
                    labels = seg['labels']
                    for j in range(0,np.uint16(1+seg['seglist'].__len__()/mdf.MAX_BATCH_SIZE)):
                        feed_dict = {sp_in     :sp[j*mdf.MAX_BATCH_SIZE:(1+j)*mdf.MAX_BATCH_SIZE],
                                    nn_in     : nn[j*mdf.MAX_BATCH_SIZE:(1+j)*mdf.MAX_BATCH_SIZE],
                                    pic_in    : pic[j*mdf.MAX_BATCH_SIZE:(1+j)*mdf.MAX_BATCH_SIZE],
                                    labels_in : labels[j*mdf.MAX_BATCH_SIZE:(1+j)*mdf.MAX_BATCH_SIZE]}

                        with tf.device('/gpu:0'):
                            _, loss_value = sess.run([train_op, loss],feed_dict = feed_dict)

    true_count += sess.run(eval_correct, feed_dict=feed_dict)
    precision = true_count / num_examples
    print('  Num examples: %d  Num correct: %d  Precision @ 1: %0.04f' %
        (num_examples, true_count, precision))

def train():
    mean = np.load("/home/nyarbel/Python_Gaze_Prediction/mean.npy")
    img_dir = '/home/nyarbel/Python_Gaze_Prediction/MSRA10K_Imgs_GT/Imgs/'
    seg_dir = '/home/nyarbel/Python_Gaze_Prediction/f_Segs/'
    _ , images = mdp.dirtomdfbatchmsra(img_dir)
    train_data = images[0:80]
    val_data = images[80:90]
    test_data = images[90:100]
    del images
    #train_data = images[0:8000]
    #val_data = images[8000:9000]
    #test_data = images[9000:10000]
    with tf.Graph().as_default():
        global_step = tf.Variable(0, trainable=False)
        sess = tf.Session()
        xdim = (mdf.MDF_DIM,mdf.MDF_DIM,3)
        sp_in = tf.placeholder(tf.float32, (None,) + xdim)
        nn_in = tf.placeholder(tf.float32, (None,) + xdim)
        pic_in = tf.placeholder(tf.float32, (None,) + xdim)
        labels_in = tf.placeholder(tf.int32,(None,)+(2,))
        s3cnn = mdf.S3CNN()
        logits = s3cnn.inference(sp_in,nn_in,pic_in)
        loss = s3cnn.loss(logits,labels_in)
        train_op = s3cnn.train(loss,global_step)
        eval_correct = s3cnn.evaluation(logits, labels_in)

        saver = tf.train.Saver(tf.all_variables())
        # Build the summary operation based on the TF collection of Summaries.
        summary_op = tf.merge_all_summaries()
        # Build an initialization operation to run below.
        init = tf.initialize_all_variables()
        # Start running operations on the Graph.
        sess = tf.Session(config=tf.ConfigProto(
            log_device_placement=FLAGS.log_device_placement))
        summary_writer = tf.train.SummaryWriter(FLAGS.train_dir, sess.graph)
        sess.run(init)
        # Start the queue runners.
        tf.train.start_queue_runners(sess=sess)
        for step in xrange(FLAGS.max_steps):
            start_time = time.time()
            for batch in range(0,np.uint16((train_data.__len__()-1)/FLAGS.image_per_batch)):
                for i in train_data[batch*FLAGS.image_per_batch:(batch+1)*FLAGS.image_per_batch]:
                    segdata = mdp.trainable_segmentations_from_batch(np.load(seg_dir+i[:-3]+'npy').item())
                    image = sio.imread(img_dir+i)
                    for k in list(segdata.keys()):
                        seg = segdata[k]
                        mdfin = mdp.im2mdfin2(image,mean,seg['segmap'],seg['seglist'])
                        sp = np.reshape(np.ravel(mdfin[0:mdfin.__len__():3]),[np.uint16(mdfin.__len__()/3),227,227,3])
                        nn = np.reshape(np.ravel(mdfin[1:mdfin.__len__():3]),[np.uint16(mdfin.__len__()/3),227,227,3])
                        pic = np.reshape(np.ravel(mdfin[2:mdfin.__len__():3]),[np.uint16(mdfin.__len__()/3),227,227,3])
                        labels = seg['labels']
                        for j in range(0,np.uint16(1+(seg['seglist'].__len__()-1)/mdf.MAX_BATCH_SIZE)):
                            feed_dict = {sp_in     :sp[j*mdf.MAX_BATCH_SIZE:(1+j)*mdf.MAX_BATCH_SIZE],
                                         nn_in     : nn[j*mdf.MAX_BATCH_SIZE:(1+j)*mdf.MAX_BATCH_SIZE],
                                         pic_in    : pic[j*mdf.MAX_BATCH_SIZE:(1+j)*mdf.MAX_BATCH_SIZE],
                                         labels_in : labels[j*mdf.MAX_BATCH_SIZE:(1+j)*mdf.MAX_BATCH_SIZE]}
                            with tf.device('/gpu:0'):
                                _, loss_value = sess.run([train_op, loss],feed_dict = feed_dict)
            duration = time.time() - start_time
                  # Write the summaries and print an overview fairly often.
            duration = time.time() - start_time

            
            # Print status to stdout.
            print('Step %d: loss = %.2f (%.3f sec)' % (step, loss_value, duration))
            # Update the events file.
            summary_str = sess.run(summary_op, feed_dict=feed_dict)
            summary_writer.add_summary(summary_str, step)
            summary_writer.flush()

            # Save a checkpoint and evaluate the model periodically.
            if (step + 1) % 1000 == 0 or (step + 1) == FLAGS.max_steps:
                checkpoint_file = os.path.join(FLAGS.train_dir, 'checkpoint')
                saver.save(sess, checkpoint_file, global_step=step)
                # Evaluate against the training set.
                print('Training Data Eval:')
                do_eval(sess,
                        eval_correct,
                        mean,
                        sp_in, nn_in,pic_in,
                        labels_in,
                        seg_dir,
                        img_dir,
                        train_data)
                # Evaluate against the validation set.
                print('Validation Data Eval:')
                do_eval(sess,
                        eval_correct,
                        mean,
                        sp_in, nn_in,pic_in,
                        labels_in,
                        seg_dir,
                        img_dir,
                        val_data)

                # Evaluate against the test set.
                print('Test Data Eval:')
                do_eval(sess,
                        eval_correct,
                        mean,
                        sp_in, nn_in,pic_in,
                        labels_in,
                        seg_dir,
                        img_dir,
                        test_data)

def main(_):
  train()


if __name__ == '__main__':
    tf.app.run()
