import os
import math
import numpy as np
import tensorflow as tf
from datetime import datetime


WORKSPACE = 'gs://ksalama-cloudml/text_workspace'
COOC_DIR = '{}/cooc'.format(WORKSPACE)
MODELS_DIR = '{}/models'.format(WORKSPACE)
SEED = 19831060


FEATURES_SCHEMA = {
    'item1': tf.FixedLenFeature(dtype=tf.string, shape=()),
    'item2': tf.FixedLenFeature(dtype=tf.string, shape=()),
    'score': tf.FixedLenFeature(dtype=tf.float32, shape=()),
    'weight': tf.FixedLenFeature(dtype=tf.float32, shape=()),
    'type': tf.FixedLenFeature(dtype=tf.string, shape=())
}

WEIGHT_FEATURE_NAME = 'weight'
TARGET_FEATURE_NAME = 'score'


def make_input_fn(file_pattern, 
                  batch_size=128, num_epochs=1, mode=tf.estimator.ModeKeys.EVAL):

    def _input_fn():
        dataset = tf.data.experimental.make_batched_features_dataset(
            file_pattern,
            batch_size,
            features=FEATURES_SCHEMA,
            label_key=TARGET_FEATURE_NAME,
            reader=tf.data.TFRecordDataset,
            shuffle_buffer_size=batch_size * 10,
            reader_num_threads=1,
            parser_num_threads=2,
            num_epochs=num_epochs,
            shuffle=(mode==tf.estimator.ModeKeys.TRAIN),
            sloppy_ordering=True,
            drop_final_batch=True
        )
        return dataset
    
    return _input_fn


def create_feature_columns(embedding_size, vocab1_file, vocab2_file):
    
    feature_columns = []

    feature_columns.append(
        tf.feature_column.embedding_column(
            tf.feature_column.categorical_column_with_vocabulary_file(
                key='item1', 
                vocabulary_file=vocab1_file
            ), 
            embedding_size
        )
    )

    feature_columns.append(
        tf.feature_column.embedding_column(
            tf.feature_column.categorical_column_with_vocabulary_file(
                key='item2', 
                vocabulary_file=vocab2_file
            ), 
            embedding_size
        )
    )
        
    return feature_columns


def compute_loss(labels, predictions, weights, types):
    
    def _positive_sample_cost(errors, weights):
        return 0.5 * weights * tf.math.square(errors)
    
    def _negative_sample_cost(errors, weights):
        return weights * tf.math.log(1 + tf.exp(errors))
    
    errors = predictions - labels
    
    p_loss = _positive_sample_cost(errors, weights)
    n_loss = _negative_sample_cost(errors, weights)
    loss = tf.where(tf.equal(types, 'P'), p_loss, n_loss)
    
    return tf.reduce_sum(loss)

def model_fn(features, labels, mode, params):
    
    items1 = features['item1']
    feature_columns = create_feature_columns(
        params.embedding_size, params.vocab1_file, params.vocab2_file)
    
    item1_layer = tf.feature_column.input_layer(
        features={'item1': items1}, feature_columns=[feature_columns[0]])
    
    if mode != tf.estimator.ModeKeys.PREDICT:
        items2 = features['item2']
        item2_layer = tf.feature_column.input_layer(
            features={'item2': items2}, feature_columns=[feature_columns[1]])
        
        dot_product = tf.keras.layers.Dot(axes=1)([item1_layer, item2_layer])
        logits = (params.max_value - params.min_value) * tf.sigmoid(dot_product) + params.min_value 

    predictions = None
    export_outputs = None
    loss = None
    train_op = None

    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions =  item1_layer
        export_outputs = {'predictions': tf.estimator.export.PredictOutput(predictions)}
    else:
        types = features['type']
        weights = features[WEIGHT_FEATURE_NAME]

        loss = compute_loss(
            labels=labels, 
            predictions=tf.squeeze(logits), 
            weights=weights, 
            types=types
        )
        
        train_op=tf.train.AdamOptimizer(params.learning_rate).minimize(
            loss=loss, global_step=tf.train.get_global_step())
        
    
    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=predictions,
        export_outputs=export_outputs,
        loss=loss,
        train_op=train_op
    )


def create_estimator(params, run_config):
    
    estimator = tf.estimator.Estimator(
        model_fn,
        params=params,
        config=run_config
    )
    
    return estimator


def run_experiment(params, run_config):
    
    train_data_files = params.train_data_files
    eval_data_files = params.eval_data_files
    
    # TrainSpec ####################################
    train_input_fn = make_input_fn(
        train_data_files,
        batch_size=params.batch_size,
        num_epochs=None,
        mode=tf.estimator.ModeKeys.TRAIN
    )
    
    train_spec = tf.estimator.TrainSpec(
        input_fn = train_input_fn,
        max_steps=params.traning_steps
    )
    ###############################################    
    
    # EvalSpec ####################################
    eval_input_fn = make_input_fn(
        eval_data_files,
        num_epochs=None,
        batch_size=params.batch_size,
    )

    eval_spec = tf.estimator.EvalSpec(
        name=datetime.utcnow().strftime("%H%M%S"),
        input_fn = eval_input_fn,
        steps=params.eval_steps,
        start_delay_secs=0,
        throttle_secs=params.eval_throttle_secs
    )
    ###############################################

    tf.logging.set_verbosity(tf.logging.INFO)
    
    if tf.gfile.Exists(run_config.model_dir):
        print("Removing previous artefacts...")
        tf.gfile.DeleteRecursively(run_config.model_dir)
            
    print("")
    estimator = create_estimator(params, run_config)
    print("")
    
    time_start = datetime.utcnow() 
    print("Experiment started at {}".format(time_start.strftime("%H:%M:%S")))
    print(".......................................") 

    tf.estimator.train_and_evaluate(
        estimator=estimator,
        train_spec=train_spec, 
        eval_spec=eval_spec
    )

    time_end = datetime.utcnow() 
    print(".......................................")
    print("Experiment finished at {}".format(time_end.strftime("%H:%M:%S")))
    print("")
    time_elapsed = time_end - time_start
    print("Experiment elapsed time: {} seconds".format(time_elapsed.total_seconds()))
    
    return estimator


MODEL_NAME = 'cooc2emb-01'
model_dir = os.path.join(MODELS_DIR, MODEL_NAME)
info_file = os.path.join(COOC_DIR, 'info.log')
min_value = 15
max_value = -5

info_map = {}

if os.path.exists(info_file):
    try:
        with open(info_file) as f:
            for line in f.readlines():
                key, value = line.split(":")
                info_map[key] = float(value)
        min_value = math.floor(info_map['min'])
        max_value = math.ceil(info_map['max'])
    except: pass
    
class HParams():
    pass

params  = HParams()
params.train_data_files = "{}/cooc-00000.tfrecords".format(COOC_DIR)
params.eval_data_files = "{}/cooc-000000.tfrecords".format(COOC_DIR)
params.vocab1_file = os.path.join(COOC_DIR,'vocab.txt')
params.vocab2_file = os.path.join(COOC_DIR,'vocab.txt')
params.embedding_size = 128
params.min_value = min_value
params.max_value = max_value
params.batch_size = 265
params.traning_steps = 30000
params.learning_rate = 0.001
params.eval_steps = 1
params.eval_throttle_secs = 0

print(vars(params))

run_config = tf.estimator.RunConfig(
    tf_random_seed=SEED,
    save_checkpoints_steps=1000,
    keep_checkpoint_max=3,
    model_dir=model_dir,
)


estimator = run_experiment(params, run_config)

