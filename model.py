from transformers import TFBertModel
import tensorflow as tf
from tensorflow import keras
from metircs import MultiLabelAccuracy, MultiLabelF1, MultiLabelPrecision, MultiLabelRecall


def create_model(loss, max_seq_len=512, num_classes=34):
    bert = TFBertModel.from_pretrained('bert-base-multilingual-cased')
    final_dense = tf.keras.layers.Dense(units=num_classes, activation='sigmoid')

    input_ids = keras.layers.Input(shape=(max_seq_len,), dtype='int32', name="input_ids")
    attention_mask = keras.layers.Input(shape=(max_seq_len,), dtype='int32', name="attention_mask")

    outputs, cls = bert(input_ids)
    logits = final_dense(cls)

    model = keras.Model(inputs=[input_ids, attention_mask], outputs=logits)
    model.build(input_shape=(None, max_seq_len))

    optimizer = keras.optimizers.Adam()

    model.compile(optimizer=optimizer,
                  loss=loss,
                  metrics=[MultiLabelAccuracy(),
                           MultiLabelPrecision(),
                           MultiLabelRecall(),
                           MultiLabelF1()])
    return model
