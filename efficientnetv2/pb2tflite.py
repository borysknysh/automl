import argparse
from argparse import RawTextHelpFormatter
my_parser = argparse.ArgumentParser(description='parse cmd', formatter_class=RawTextHelpFormatter)
my_parser.add_argument('--pb_model',
                       metavar='input_model',
                       action='store',
                       type=str,
                       help='input pb')
my_parser.add_argument('--tflite_model',
                       metavar='tflite_model',
                       action='store',
                       type=str,
                       help='output tflite')
my_parser.add_argument('--conv_mode',
                       metavar='conv_mode',
                       action='store',
                       type=int,
                       help='Conversion modes: \n0 --- simple tflite conversion;\n1 --- Dynamic range quantization;\n2 --- Full integer quantization with float fallback;\n3 --- Full integer quantization with Integer only;\n4 --- variation with optimization and supported ops;\n5 --- variation without optimization and with supported ops')


args = my_parser.parse_args()
print(f"EXPERIMENTS/pb/{args.pb_model}")

import tensorflow as tf
import numpy as np
import pathlib

def representative_dataset_gen():
  for _ in range(num_calibration_steps):
      input_data1 = np.array(np.random.random_sample((1,224,224,3)), dtype=np.float32)
      yield [input_data1]

converter = tf.lite.TFLiteConverter.from_saved_model(f"EXPERIMENTS/pb/{args.pb_model}")

converter.experimental_new_converter = True
num_calibration_steps = 100

if args.conv_mode == 1:
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
elif args.conv_mode == 2:
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset_gen
elif args.conv_mode == 3:
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8  # or tf.uint8
    converter.inference_output_type = tf.int8  # or tf.uint8
elif args.conv_mode == 4:
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
elif args.conv_mode == 5:
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
    
tflite_quant_model = converter.convert()

tflite_models_dir = pathlib.Path(f"EXPERIMENTS/tflite/{args.tflite_model}")
tflite_models_dir.mkdir(exist_ok=True, parents=True)

tflite_model_quant_file = tflite_models_dir/"quantized_model_justopt.tflite"
tflite_model_quant_file.write_bytes(tflite_quant_model)
