from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense,BatchNormalization,Activation
import tensorflow as tf
import numpy as np



def build_model():
    inputs = keras.Input(shape=(4,))
    x = layers.Dense(32,activation='tanh')(inputs)
    x = layers.Dense(64,activation='tanh')(x)

    outputs = layers.Dense(2, activation="linear")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="dqn_model")
    #lr stands for learning rate
    optimizer = keras.optimizers.Adam(lr=0.001)

    model.compile(
        optimizer =optimizer,loss='mse',
    )
    return model

#Get the predictd best action by our network
def deep_q_policy(state):
    
    #print("state: ", state)
    # Get the model's reward estimations of each Q(s,a)
    #action_q_values = model.predict([[state[0],state[1],state[2],state[3]]])
    #x = tf.random.uniform((1,4)) # working random - REPLACE (DEBUG)
    x=None
    try:
        x = tf.constant([[state[0], state[1], state[2], state[3]]])
    except:
        x = tf.constant([state[0]])
    #print(x)
    action_q_values = model.predict(x)
    
    # If exploiting, choose the model's estimated best action 
    return np.argmax(action_q_values)

steps_to_target = 20
target_update_steps = 20
#Update the target network
def update_target():
    global steps_to_target
    # Update target network if it is time 
    if steps_to_target<=0:
        target_network.set_weights(model.get_weights())
        steps_to_target = target_update_steps
    # Otherwise count down
    else:
        steps_to_target-=1

model = build_model()
target_network = build_model()