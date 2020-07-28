import os
import pickle
import argparse
from datetime import datetime
import tensorflow_addons as tfa

from model import create_model
from tensorflow import keras
from train_helper import create_learning_rate_scheduler
from alarm_bot import ExamAlarmBot


parser = argparse.ArgumentParser()
parser.add_argument('--loss', type=str, default='bce')
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--num_epochs', type=int, default=10)
parser.add_argument('--alpha', type=float, default=0.0)
parser.add_argument('--gamma', type=float, default=0.0)

args = parser.parse_args()


def load_data(path):
    with open(path, 'rb') as f:
        inputs, labels = pickle.load(f)

    return [inputs['input_ids'], inputs['attention_mask']], labels


max_seq_len = 512
num_classes = 34
batch_size = args.batch_size
num_epochs = args.num_epochs
warmup_epoch_count = 5

train_name = "{}_{}_{}_{}".format(args.loss, int(args.alpha * 10), int(args.gamma * 10), datetime.now().strftime("%m%d-%H"))
result_path = os.path.join('result', train_name)
tb_path = os.path.join('result', train_name, 'logs')

train_x, train_y = load_data('data/train_set.pkl')
val_x, val_y = load_data('data/val_set.pkl')
test_x, test_y = load_data('data/test_set.pkl')

losses = {
    'bce': keras.losses.BinaryCrossentropy(),
    'focal': tfa.losses.SigmoidFocalCrossEntropy(alpha=args.alpha, gamma=args.gamma)
}
model = create_model(max_seq_len, num_classes, losses[args.loss])

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

bot = ExamAlarmBot()
bot.send_msg('{} train is done, result : {}'.format(train_name, result_eval))