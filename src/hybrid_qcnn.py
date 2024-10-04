import os
import shutil
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
import pennylane as qml
from pennylane import numpy as qnp

# Set the random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

class QuantumLayer(layers.Layer):
    def __init__(self, n_qubits, n_layers, name='quantum_layer', **kwargs):
        super(QuantumLayer, self).__init__(name=name, **kwargs)
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self._weights_var = None
        self.dev = qml.device('default.qubit', wires=self.n_qubits)
        self.qnode = qml.QNode(self.quantum_circuit, self.dev, interface='tf')
    
    def build(self, input_shape):
        self._weights_var = self.add_weight(
            name='quantum_weights',
            shape=(self.n_layers, self.n_qubits),
            initializer='random_normal',
            trainable=True
        )
        self.built = True

    def quantum_circuit(self, inputs, weights):
        # Ensure inputs are properly scaled
        inputs = tf.cast(inputs, tf.float32)
        
        for i in range(self.n_qubits):
            qml.RY(inputs[i], wires=i)
        
        for layer in range(self.n_layers):
            for i in range(self.n_qubits):
                qml.RY(weights[layer, i], wires=i)
            for i in range(self.n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
        
        # Return measurements as a tensor instead of a list
        return qml.state()

    def call(self, inputs):
        inputs = tf.cast(inputs, tf.float32)
        batch_size = tf.shape(inputs)[0]
        
        # Ensure the inputs are reshaped to the expected shape
        if inputs.shape[-1] != self.n_qubits:
            raise ValueError(f"Expected input shape last dimension to be {self.n_qubits}, got {inputs.shape[-1]}.")

        # Process each input using the quantum circuit and ensure proper tensor shape
        quantum_output = tf.map_fn(
            lambda x: tf.cast(self.qnode(x, self._weights_var), tf.float32),
            inputs,
            fn_output_signature=tf.float32
        )
        
        # Reshape the output to match the expected shape
        quantum_output = tf.reshape(quantum_output, [batch_size, -1])
        # Take only the first n_qubits elements
        quantum_output = quantum_output[:, :self.n_qubits]
        
        return quantum_output

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.n_qubits)

    def get_config(self):
        config = super(QuantumLayer, self).get_config()
        config.update({
            'n_qubits': self.n_qubits,
            'n_layers': self.n_layers
        })
        return config


def prepare_dataset(base_dir, train_dir, val_dir, split_size=0.8):
    """
    Prepare the dataset by splitting images into train and validation sets
    """
    print(f"Preparing dataset from: {base_dir}")
    
    # Create directories if they don't exist
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    # Get all disease categories
    categories = [d for d in os.listdir(base_dir) 
                 if os.path.isdir(os.path.join(base_dir, d))]
    
    print(f"Found categories: {categories}")

    for category in categories:
        print(f"Processing category: {category}")
        
        # Create category directories
        train_category_dir = os.path.join(train_dir, category)
        val_category_dir = os.path.join(val_dir, category)
        os.makedirs(train_category_dir, exist_ok=True)
        os.makedirs(val_category_dir, exist_ok=True)
        
        # Get all images
        category_dir = os.path.join(base_dir, category)
        images = [f for f in os.listdir(category_dir) 
                 if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
        
        if not images:
            print(f"No images found in category: {category}")
            continue
            
        print(f"Found {len(images)} images in {category}")
        
        # Split images
        train_images, val_images = train_test_split(
            images, 
            train_size=split_size,
            random_state=42
        )
        
        # Copy images
        for img in train_images:
            src = os.path.join(category_dir, img)
            dst = os.path.join(train_category_dir, img)
            if not os.path.exists(dst):
                shutil.copy2(src, dst)
            
        for img in val_images:
            src = os.path.join(category_dir, img)
            dst = os.path.join(val_category_dir, img)
            if not os.path.exists(dst):
                shutil.copy2(src, dst)
        
        print(f"Split {category}: {len(train_images)} train, {len(val_images)} validation")

def create_hybrid_model(input_shape, n_classes, n_qubits=4):
    inputs = layers.Input(shape=input_shape)
    
    # Classical convolutional layers with batch normalization
    x = layers.Conv2D(32, (3, 3), padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    
    x = layers.Conv2D(64, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    
    x = layers.Conv2D(64, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Flatten()(x)
    
    # Dense layer to reduce dimensions
    x = layers.Dense(n_qubits)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    
    # Quantum layer
    x = QuantumLayer(n_qubits=n_qubits, n_layers=2)(x)
    
    # Output layers
    x = layers.Dense(64)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(n_classes, activation='softmax')(x)
    
    return models.Model(inputs=inputs, outputs=outputs)

def main():
    # Define directories
    current_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.join(current_dir, 'archive')
    train_dir = os.path.join(current_dir, 'dataset', 'train')
    val_dir = os.path.join(current_dir, 'dataset', 'validation')

    print("Starting data preparation...")
    prepare_dataset(base_dir, train_dir, val_dir)

    print("Setting up data generators...")
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    val_datagen = ImageDataGenerator(rescale=1./255)

    batch_size = 16  # Reduced batch size
    
    print("Creating data generators...")
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(128, 128),
        batch_size=batch_size,
        class_mode='categorical'
    )

    val_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size=(128, 128),
        batch_size=batch_size,
        class_mode='categorical'
    )

    print(f"Number of training samples: {train_generator.samples}")
    print(f"Number of validation samples: {val_generator.samples}")
    print(f"Number of classes: {train_generator.num_classes}")
    print(f"Class indices: {train_generator.class_indices}")

    # Create and compile model
    print("Creating model...")
    input_shape = (128, 128, 3)
    n_classes = train_generator.num_classes
    
    model = create_hybrid_model(input_shape=input_shape, n_classes=n_classes)
    
    # Use a lower learning rate
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
    
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # Model summary
    model.summary()

    # Create checkpoint directory
    checkpoint_dir = os.path.join(current_dir, 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Train the model
    print("Starting training...")
    try:
        history = model.fit(
            train_generator,
            epochs=10,
            validation_data=val_generator,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=3,
                    restore_best_weights=True
                ),
                tf.keras.callbacks.ModelCheckpoint(
                    filepath=os.path.join(checkpoint_dir, 'best_model.keras'),
                    save_best_only=True,
                    monitor='val_accuracy'
                ),
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=2,
                    min_lr=1e-6
                )
            ]
        )

        # Save the final model
        print("Saving final model...")
        model.save(os.path.join(current_dir, 'hybrid_qcnn_final_model.keras'))
        
        print("Training completed successfully!")
        
    except Exception as e:
        print(f"An error occurred during training: {str(e)}")
        raise

if __name__ == "__main__":
    main()