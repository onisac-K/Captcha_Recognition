#-*- coding:utf-8 -*-
import tensorflow as tf
import numpy as np
import string
import Generate_captcha
import Captcha_model

log_dir = './Captcha_log'
save_dir = './Captcha_save'

if __name__ == '__main__':
    captcha = Generate_captcha.generateCaptcha()
    width,height,char_num,characters,classes = captcha.get_parameter()

    x = tf.placeholder(tf.float32, [None, height,width,1])
    y_ = tf.placeholder(tf.float32, [None, char_num*classes])
    keep_prob = tf.placeholder(tf.float32)

    model = Captcha_model.captchaModel(width,height,char_num,classes)
    y_conv = model.create_model(x,keep_prob)
    cross_entropy = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y_,logits=y_conv))
    tf.summary.scalar('loss', cross_entropy)
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

    predict = tf.reshape(y_conv, [-1,char_num, classes])
    real = tf.reshape(y_,[-1,char_num, classes])
    correct_prediction = tf.equal(tf.argmax(predict,2), tf.argmax(real,2))
    correct_prediction = tf.cast(correct_prediction, tf.float32)
    accuracy = tf.reduce_mean(correct_prediction)
    tf.summary.scalar('accuracy', accuracy)
    saver = tf.train.Saver()
    merged = tf.summary.merge_all()

    with tf.Session() as sess:
        summary_writer = tf.summary.FileWriter(log_dir, sess.graph)
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, "./model/capcha_model.ckpt")
        step = 69301
        while True:
            batch_x,batch_y = next(captcha.gen_captcha(64))
            summary,_,loss = sess.run([merged,train_step,cross_entropy],feed_dict={x: batch_x, y_: batch_y, keep_prob: 0.75})
            summary_writer.add_summary(summary,step)
            print ('step:%d,loss:%f' % (step,loss))
            if step % 100 == 0:
                batch_x_test,batch_y_test = next(captcha.gen_captcha(100))
                acc = sess.run(accuracy, feed_dict={x: batch_x_test, y_: batch_y_test, keep_prob: 1.})
                print ('###############################################step:%d,accuracy:%f' % (step,acc))
                saver.save(sess,"./capcha_model.ckpt")

                if acc > 0.99:
                    break
            step += 1
