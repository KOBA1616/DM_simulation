import numpy as np
import sys

def main(path):
    d = np.load(path)
    print(list(d.keys()))
    if 'policies' in d:
        print('policies', d['policies'].shape)
    if 'states' in d:
        print('states', d['states'].shape)
    print('has legal_mask', 'legal_mask' in d)

if __name__ == '__main__':
    path = sys.argv[1] if len(sys.argv) > 1 else 'data/transformer_training_data_converted.npz'
    main(path)
