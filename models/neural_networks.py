# src/models/neural_networks.py
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Flatten, Conv2D, BatchNormalization, Input, Concatenate, Lambda
from tensorflow.keras.optimizers import Adam
import numpy as np

def create_dqn_model(input_shape, action_size, learning_rate=0.001):
    """
    Create a Deep Q-Network model
    
    Args:
        input_shape: Shape of the input state
        action_size: Number of possible actions
        learning_rate: Learning rate for the optimizer
        
    Returns:
        model: Compiled Keras model
    """
    # Convert to log scale to handle large numbers better
    model = Sequential([
        # Reshape and convert to log scale
        Lambda(
            lambda x: tf.math.log(tf.cast(x, tf.float32) + 1.0),
            input_shape=input_shape
        ),
        
        # Convolutional layers
        Conv2D(128, (2, 2), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(128, (2, 2), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(64, (2, 2), activation='relu', padding='same'),
        BatchNormalization(),
        
        # Flatten and Dense layers
        Flatten(),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dense(128, activation='relu'),
        Dense(action_size, activation='linear')
    ])
    
    model.compile(loss='mse', optimizer=Adam(learning_rate=learning_rate))
    return model

def create_actor_critic_model(input_shape, action_size, learning_rate=0.001):
    """
    Create an Actor-Critic model for policy-based methods
    
    Args:
        input_shape: Shape of the input state
        action_size: Number of possible actions
        learning_rate: Learning rate for the optimizer
        
    Returns:
        actor: Actor model (policy)
        critic: Critic model (value function)
    """
    # Shared feature extraction layers
    input_layer = Input(shape=input_shape)
    
    # Logarithmic transformation
    x = Lambda(lambda x: tf.math.log(tf.cast(x, tf.float32) + 1.0))(input_layer)
    
    # Convolutional layers
    x = Conv2D(128, (2, 2), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv2D(128, (2, 2), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv2D(64, (2, 2), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    
    # Flatten
    x = Flatten()(x)
    x = Dense(256, activation='relu')(x)
    x = BatchNormalization()(x)
    
    # Actor head (policy)
    actor_hidden = Dense(128, activation='relu')(x)
    actor_output = Dense(action_size, activation='softmax')(actor_hidden)
    
    # Critic head (value function)
    critic_hidden = Dense(128, activation='relu')(x)
    critic_output = Dense(1, activation='linear')(critic_hidden)
    
    # Create models
    actor = Model(inputs=input_layer, outputs=actor_output)
    critic = Model(inputs=input_layer, outputs=critic_output)
    
    # Compile models
    actor.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=learning_rate))
    critic.compile(loss='mse', optimizer=Adam(learning_rate=learning_rate))
    
    return actor, critic

def create_combined_model(input_shape, action_size, learning_rate=0.001):
    """
    Create a combined model for PPO or A2C implementations
    
    Args:
        input_shape: Shape of the input state
        action_size: Number of possible actions
        learning_rate: Learning rate for the optimizer
        
    Returns:
        combined_model: Combined actor-critic model
        actor: Actor sub-model
        critic: Critic sub-model
    """
    # Input layer
    input_layer = Input(shape=input_shape)
    
    # Logarithmic transformation
    x = Lambda(lambda x: tf.math.log(tf.cast(x, tf.float32) + 1.0))(input_layer)
    
    # Convolutional layers
    x = Conv2D(128, (2, 2), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv2D(128, (2, 2), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv2D(64, (2, 2), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    
    # Flatten
    x = Flatten()(x)
    x = Dense(256, activation='relu')(x)
    x = BatchNormalization()(x)
    
    # Actor head (policy distribution)
    actor_hidden = Dense(128, activation='relu')(x)
    actions_proba = Dense(action_size, activation='softmax', name='actor')(actor_hidden)
    
    # Critic head (state value)
    critic_hidden = Dense(128, activation='relu')(x)
    state_value = Dense(1, activation='linear', name='critic')(critic_hidden)
    
    # Create and compile models
    combined_model = Model(inputs=input_layer, outputs=[actions_proba, state_value])
    
    # Extract individual models for easier access
    actor = Model(inputs=input_layer, outputs=actions_proba)
    critic = Model(inputs=input_layer, outputs=state_value)
    
    # Custom loss functions for PPO will be defined in the agent class
    # Here we just use a placeholder loss to initialize the model
    losses = {
        'actor': 'categorical_crossentropy',
        'critic': 'mse'
    }
    loss_weights = {
        'actor': 1.0,
        'critic': 0.5
    }
    
    combined_model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss=losses,
        loss_weights=loss_weights
    )
    
    return combined_model, actor, critic