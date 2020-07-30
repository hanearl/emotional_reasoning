import os
import pickle
import argparse
import json
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
parser.add_argument('--max_seq_len', type=int, default=512)
parser.add_argument('--num_classes', type=int, default=34)
parser.add_argument('--warmup_epoch_count', type=int, default=5)
parser.add_argument('--alpha', type=float, default=0.0)
parser.add_argument('--gamma', type=float, default=0.0)
parser.add_argument('--train_set', type=str, default='train_set.pkl')
parser.add_argument('--val_set', type=str, default='val_set.pkl')
parser.add_argument('--test_set', type=str, default='test_set.pkl')
parser.add_argument('--result_dir', type=str, default='result')
parser.add_argument('--data_dir', type=str, default='data')
args = parser.parse_args()


def load_data(path):
    with open(path, 'rb') as f:
        inputs, labels = pickle.load(f)

    return [inputs['input_ids'], inputs['attention_mask']], labels


# 학습 결과 저장 경로 설정
train_name = "{}_{}_{}_{}".format(args.loss, int(args.alpha * 10), int(args.gamma * 10), datetime.now().strftime("%m%d-%H%M"))
result_path = os.path.join(args.result_dir, train_name)
tb_path = os.path.join(args.result_dir, train_name, 'logs')

if not os.path.exists(result_path):
    os.mkdir(result_path)

# config 정보 저장
with open(os.path.join(result_path, 'config.json'), 'w') as f:
    json.dump(args.__dict__, f)
print(args.__dict__)

# train/val/test 데이터 load
train_x, train_y = load_data(os.path.join(args.data_dir, args.train_set))
val_x, val_y = load_data(os.path.join(args.data_dir, args.val_set))
test_x, test_y = load_data(os.path.join(args.data_dir, args.test_set))

# 모델 생성
losses = {
    'bce': keras.losses.BinaryCrossentropy(),
    'focal': tfa.losses.SigmoidFocalCrossEntropy(alpha=args.alpha, gamma=args.gamma)
}
model = create_model(losses[args.loss], args.max_seq_len, args.num_classes)

# callback 함수 정의
log_dir = os.path.join(tb_path, datetime.now().strftime("%Y%m%d-%H%M%s"))
tb_callback = keras.callbacks.TensorBoard(log_dir=log_dir)
lr_scheduler_callback = create_learning_rate_scheduler(
    warmup_epoch_count=args.warmup_epoch_count, total_epoch_count=args.num_epochs)

# 모델 학습
hist = model.fit(
    x=train_x,
    y=train_y,
    validation_data=(val_x, val_y),
    validation_batch_size=args.batch_size,
    batch_size=args.batch_size,
    shuffle=True,
    epochs=args.num_epochs,
    callbacks=[lr_scheduler_callback, tb_callback]
)

# 모델 학습 결과, eval 결과 저장
model.save_weights(os.path.join(result_path, 'model.h5'), overwrite=True)

with open(os.path.join(result_path, 'result_train.pkl'), 'wb') as f:
    pickle.dump(hist.history, f)

result_eval = model.evaluate(x=test_x, y=test_y, batch_size=args.batch_size)
with open(os.path.join(result_path, 'result_eval.pkl'), 'wb') as f:
    pickle.dump(result_eval, f)

# train 결과 telegram 전송
bot = ExamAlarmBot()
bot.send_msg('{} train is done, result : {}'.format(train_name, result_eval))