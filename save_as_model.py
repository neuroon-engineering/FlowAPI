
from new_train_model import returnCompiledModel as model
from new_train_model import returnTrainGenerator as train_generator
from new_train_model import returnValidationGenerator as validate_generator

nb_train_samples = 5005
nb_validation_samples = 218
epochs = 1
batch_size = 16


model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples,
    epochs=epochs,
    validation_data=validate_generator,
    validation_steps=nb_validation_samples)

model.save_weights('first_try.h5')
