import argparse
my_parser = argparse.ArgumentParser(description='parse cmd')
my_parser.add_argument('--tflite_model',
                       metavar='tflite_model',
                       action='store',
                       type=str,
                       help='output tflite')


args = my_parser.parse_args()

import tensorflow as tf
import tflite_runtime.interpreter as tflite
import numpy as np
import platform

np.random.seed(1)
tf.random.set_seed(1)

#model_file, *device = "quantized/quantized_model_justopt.tflite".split('@')
#print(model_file, device)

interpreter = tflite.Interpreter(model_path=f"EXPERIMENTS/tflite/{args.tflite_model}/quantized_model_justopt.tflite", )

interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print(input_details)
print(output_details)
import time

start = time.time()
for i in range(10):
    print('Random data generation')
    input_data = [
        np.array(np.random.random_sample(input_details[0]['shape']), dtype=np.int8)
        ]
    
    interpreter.set_tensor(input_details[0]['index'], input_data[0])
    
    print('invoke')
    interpreter.invoke()
    print('out')
    output_data = interpreter.get_tensor(output_details[0]['index'])
print(time.time()-start)
