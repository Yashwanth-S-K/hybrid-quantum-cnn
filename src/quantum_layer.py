import tensorflow as tf
from tensorflow.keras import layers
import pennylane as qml

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
        inputs = tf.cast(inputs, tf.float32)
        
        for i in range(self.n_qubits):
            qml.RY(inputs[i], wires=i)
        
        for layer in range(self.n_layers):
            for i in range(self.n_qubits):
                qml.RY(weights[layer, i], wires=i)
            for i in range(self.n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
        
        return qml.state()

    def call(self, inputs):
        inputs = tf.cast(inputs, tf.float32)
        batch_size = tf.shape(inputs)[0]
        
        if inputs.shape[-1] != self.n_qubits:
            raise ValueError(f"Expected input shape last dimension to be {self.n_qubits}, got {inputs.shape[-1]}.")

        quantum_output = tf.map_fn(
            lambda x: tf.cast(self.qnode(x, self._weights_var), tf.float32),
            inputs,
            fn_output_signature=tf.float32
        )
        
        quantum_output = tf.reshape(quantum_output, [batch_size, -1])
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