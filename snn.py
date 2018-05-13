'''
Simulates 1000 neurons for 1000 ms.
Each neuron receives (random) 10% of the 100 Poisson spike trains of rate f_rate = 2 Hz between time 200 ms and 700 ms. 
Neurons are not inter-connected.
'''


import tensorflow as tf
import numpy as np
import math

n = 1000 # Number of neurons
dt = 0.5
a = 0.02
b = 0.2
c = -65.0
d = 8.0
T = math.ceil(1000/dt)
v_init = -65
u_init = -14.0
n_in = 100
rate = 2*1e-3
tau_g = 10.0


with tf.Graph().as_default() as tf_graph:
    
    E_in = tf.constant(0.0, shape=[n_in, 1])
    conections = tf.greater_equal(tf.random_uniform([n, n_in]), 0.1)
    w_in = tf.where(conections, tf.constant(0.0, shape=[n, n_in]), tf.constant(0.07, shape=[n, n_in]))
    
    
    inh = tf.less_equal(tf.random_uniform([n, 1]), 0.2)
    exc = tf.logical_not(inh)
    inh_num = tf.cast(inh, tf.float32)
    exc_num = tf.cast(exc, tf.float32)
    d = (8.0 * exc_num) + (2.0 * inh_num)
    a = (0.02 * exc_num) + (0.1 * inh_num)

    v_shape = [n, 1]
    p_shape = [n_in, 1]
    g_in_shape = [n_in, 1]
    
    v = tf.Variable(tf.ones(shape=v_shape) * v_init, dtype=tf.float32, name='v')
    u = tf.Variable(tf.ones(shape=v_shape) * u_init, dtype=tf.float32, name='u')
    g_in = tf.Variable(tf.zeros(shape=g_in_shape), dtype=tf.float32, name='g_in')
    fired = tf.Variable(np.zeros(v_shape, dtype=bool), dtype=tf.bool, name='fired')
    
    
    p_in = tf.placeholder(tf.float32, shape=p_shape)
    g_inp = g_in + p_in
    iapp = tf.reshape(tf.matmul(w_in, np.multiply(g_inp, E_in)) - \
                    tf.multiply(tf.matmul(w_in, g_inp), v),
                    tf.shape(v))
    
    g_in_op = g_in.assign((1 - dt / tau_g) * g_inp)
    
    v_in = tf.where(fired, tf.ones(tf.shape(v))*c, v)
    u_in = tf.where(fired, tf.ones(tf.shape(u))*tf.add(u, d), u)

    '''
    ODEs to be updated

    dv =(0.04*v[:,t]+5)*v[:,t]+140−u[:,t]
    v(:,t+1) = v[:,t] + (dv+I_app)*dt
    du = a*(0.2*v[:,t]−u[:,t])
    u[:,t+1] = u[:,t] + dt*du

    Written below in TF
    '''

    dv = tf.subtract(tf.add(tf.multiply(
                tf.add(tf.multiply(0.04, v_in), 5.0), v_in), 140), u_in)
    v_updated = tf.add(v_in, tf.multiply(tf.add(dv, iapp), dt))
    du = tf.multiply(a, tf.subtract(tf.multiply(b, v_in), u_in))
    u_out = tf.add(u_in, tf.multiply(dt, du))
    
    fired_op = fired.assign(tf.greater_equal(v_updated, tf.ones(tf.shape(v)) * 35))
    v_out = tf.where(fired_op, tf.ones(tf.shape(v)) * 35, v_updated)
    
    p_in_mean = tf.reduce_mean(v_out)
    v_op = v.assign(v_out)
    u_op = u.assign(u_out)
    
vs = [np.ones([n, 1]) * v_init]
us = [np.ones([n, 1]) * u_init]
fires = [np.array(u_init).reshape(1)]
means = []
with tf.Session(graph=tf_graph) as sess:
    sess.run(tf.global_variables_initializer())
    
    for t in range(T):
        
        if t * dt > 200 and t*dt < 700:
            p = np.random.rand(n_in, 1) < rate*dt
        else:
            p = np.zeros([n_in,1])
            
        feed = {p_in: p}
        
        vo, uo, _, fire, meanv = sess.run(
                    [v_op, u_op, g_in_op, fired_op, p_in_mean],
                    feed_dict=feed)

        # Reset spikes
        vs.append(vo)
        us.append(uo)
        fires.append(fire)
        means.append(meanv)
    
    inh_logical, exc_logical = sess.run([inh, exc])