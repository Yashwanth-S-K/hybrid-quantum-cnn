# Hybrid Quantum-Classical CNN

This project implements a hybrid quantum-classical convolutional neural network (QCNN) using TensorFlow and PennyLane. The model combines classical CNN layers with a quantum circuit layer for image classification tasks.

## Features

- Hybrid architecture combining classical CNN and quantum computing
- Custom quantum layer implementation using PennyLane
- Support for image classification tasks
- Easy-to-use training and testing scripts

## Prerequisites

- Python 3.8+
- TensorFlow 2.x
- PennyLane
- Other dependencies listed in `requirements.txt`

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Yashwanth-S-K/hybrid-quantum-cnn.git
cd hybrid-quantum-cnn
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Project Structure

- `src/`: Contains the main source code
  - `hybrid_qcnn.py`: Main implementation of the hybrid model
  - `quantum_layer.py`: Custom quantum layer implementation
  - `test.py`: Script for testing the trained model
- `data/`: Directory for dataset storage
- `models/`: Directory for saving trained models
- `examples/`: Contains example images for testing

## Usage

1. Training the model:
```bash
python src/hybrid_qcnn.py
```

2. Testing with a single image:
```bash
python src/test.py
```

## Model Architecture

The hybrid model consists of:
1. Classical convolutional layers for feature extraction
2. Custom quantum layer for quantum processing
3. Dense layers for classification

The quantum layer implements a variational quantum circuit with:
- Parameterized rotation gates (RY)
- Entangling operations (CNOT)
- Measurement in the computational basis

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- TensorFlow team for the deep learning framework
- PennyLane team for the quantum computing framework
- Anthropic for assistance in development

## Contact

Your Name - yashwanthskrishnamurthy@gmail.com
Project Link: https://github.com/Yashwanth-S-K/hybrid-quantum-cnn
