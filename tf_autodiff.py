import tensorflow as tf
import numpy as np

from tensorflow.python.client import timeline


def run_with_timeline(sess, f,name='default'):
    options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    run_metadata = tf.RunMetadata()
    res = sess.run(f,  options=options, run_metadata=run_metadata)
    fetched_timeline = timeline.Timeline(run_metadata.step_stats)
    chrome_trace = fetched_timeline.generate_chrome_trace_format()
    with open('timeline_%s_%d.json' % (name, N), 'w') as f:
        f.write(chrome_trace)
    return res



def main():
    for N in (1e3,1e4):

        x = tf.Variable(np.random.randn(N,1), dtype=tf.float32)
        a = tf.constant(np.random.randint(0,10,N), shape=[N,1], dtype=tf.float32)

        # $f(x) =  \frac{1}{2}(x^{T}x + (a^{T}x)^2 )$
        f = tf.multiply(1./2, \
                        tf.matmul(x, x, transpose_a=True)  \
                        + tf.pow(tf.matmul(a, x, transpose_a=True), 2))
        
        df1 = tf.matmul(a, x, transpose_a=True)
        df2 = tf.multiply(df1, a)

        df = tf.multiply(1., df2) + x

        ddf1s = tf.eye(N)
        ddf =  tf.matmul(a, a, transpose_b=True) + ddf1s

        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init)
            print("f(x) =", run_with_timeline(sess, f, name='manual_f')[0][0])
            grads = run_with_timeline(sess, df, name='manual_grad')
            print("gradient f(x) =\n", grads)
            hess = run_with_timeline(sess, ddf, name='manual_hess')
            print("hessian f(x) =\n", hess)
        


        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init)
            print("f(x) =", run_with_timeline(sess, f, name='autodiff_f')[0][0])
            grads = tf.gradients(f, [x])
            grads_res = run_with_timeline(sess, grads[0], name='autodiff_grad')
            #print("gradient f(x) =\n", grads_res)

            # Get gradients of fx with repect to x
            dfx = grads[0]
            # Compute hessian
            hess = []
            for i in range(N):
                # Take the i th value of the gradient vector dfx 
                # tf.slice: https://www.tensorflow.org/versions/0.6.0/api_docs/python/array_ops.html#slice
                dfx_i = tf.slice(dfx, begin=[i,0] , size=[1,1])
                # Feed it to tf.gradients to compute the second derivative. 
                # Since x is a vector and dfx_i is a scalar, this will return a vector : [d(dfx_i) / dx_i , ... , d(dfx_n) / dx_n]
                ddfx_i = tf.gradients(dfx_i, x)[0] # whenever we use tf.gradients, make sure you get the actual tensors by putting [0] at the end
                hess.append(ddfx_i)

            hess = tf.squeeze(hess) 
                ## Instead of doing this, you can just append each element to a list, and then do tf.pack(list_object) to get the hessian matrix too.
                ## I'll use this alternative in the second example. 
            hess_res = run_with_timeline(sess, hess, name='autodiff_hess')
            #print("hessian f(x) =\n", hess_res)


if __name__ == '__main__':
    main()
