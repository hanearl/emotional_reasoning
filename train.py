from model import create_model
from tensorflow import keras
from train_helper import create_learning_rate_scheduler
from datetime import datetime
import pickle
import os


def load_data(path):
    with open(path, 'rb') as f:
        inputs, labels = pickle.load(f)

    return [inputs['input_ids'], inputs['attention_mask']], labels


max_seq_len = 512
num_classes = 34
batch_size = 64
num_epochs = 10
warmup_epoch_count = 5
result_path = 'result'
tb_path = 'result/logs'

train_x, train_y = load_data('data/train_set.pkl')
val_x, val_y = load_data('data/val_set.pkl')
test_x, test_y = load_data('data/test_set.pkl')

model = create_model(max_seq_len, num_classes)

log_dir = os.path.join(tb_path, datetime.now().strftime("%Y%m%d-%H%M%s"))
tb_callback = keras.callbacks.TensorBoard(log_dir=log_dir)

hist = model.fit(
    x=train_x,
    y=train_y,
    validation_data=(val_x, val_y),
    validation_batch_size=batch_size,
    batch_size=batch_size,
    shuffle=True,
    epochs=num_epochs,
    callbacks=[create_learning_rate_scheduler(warmup_epoch_count=warmup_epoch_count,total_epoch_count=num_epochs),
               tb_callback]
)

model.save_weights(os.path.join(result_path, 'model.h5'), overwrite=True)

with open(os.path.join(result_path, 'result_train.pkl'), 'wb') as f:
    pickle.dump(hist.history, f)

result_eval = model.evaluate(x=test_x, y=test_y, batch_size=batch_size)
with open(os.path.join(result_path, 'result_eval.pkl'), 'wb') as f:
    pickle.dump(result_eval, f)
