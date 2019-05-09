import tempfile
import os
import subprocess
import tensorflow as tf
from tensorflow import keras


def generate_serve_files(model):

	MODEL_DIR = tempfile.gettempdir()
	version = 2
	export_path = os.path.join(MODEL_DIR, str(version))
	print('export_path = {}\n'.format(export_path))
	if os.path.isdir(export_path):
	  print('\nAlready saved a model, cleaning up\n')

	tf.saved_model.simple_save(
	    keras.backend.get_session(),
	    export_path,
	    inputs={'input_image': model.input},
	    outputs={t.name:t for t in model.outputs})

	print('\nSaved model:')
