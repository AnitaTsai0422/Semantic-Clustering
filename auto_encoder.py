import pandas as pd
import tensorflow as tf
import numpy as np
from ast import literal_eval
import datetime, uuid

class AutoEnoder(tf.keras.Model):
  def __init__(self):
    super(AutoEnoder,self).__init__()
    
    #encoder
    self.model_encoder = tf.keras.Sequential([
        tf.keras.layers.Dense(1536,activation="relu"),
        tf.keras.layers.Dense(512,activation="relu"),
        tf.keras.layers.Dense(256,activation="relu"),
        tf.keras.layers.Dense(128,activation="relu"),
        tf.keras.layers.Dense(64,name="bottleneck",activation="relu") ])
    #decoder
    self.model_decoder = tf.keras.Sequential([
        tf.keras.layers.Dense(128,activation="relu"),
        tf.keras.layers.Dense(256,activation="relu"),
        tf.keras.layers.Dense(512,activation="relu"),
        tf.keras.layers.Dense(1536,activation="relu")])

  def call(self, inputs, training=None):
    encoder = self.model_encoder(inputs)
    decoder = self.model_decoder(encoder)
    return decoder
  
url = 'XXX.csv'
df = pd.read_csv(url, converters={'ada_v2': literal_eval})
train_df = np.stack(df['ada_v2'].values)

# Define a unique directory in DBFS
try:
  username = dbutils.notebook.entry_point.getDbutils().notebook().getContext().tags().apply('user')
except:
  username = str(uuid.uuid1()).replace("-", "")
experiment_log_dir = "XXX/user/{}/tensorboard_log_dir/".format(username)

def scheduler(epoch, lr):
  if epoch < 10:
    return lr
  else:
    return lr * tf.math.exp(-0.1)
  
autoencoder = AutoEnoder()
autoencoder.compile(optimizer=tf.keras.optimizers.Adam(lr=0.01), loss=tf.keras.losses.MeanSquaredError())

run_log_dir = experiment_log_dir + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=run_log_dir, histogram_freq=1)
callback = tf.keras.callbacks.LearningRateScheduler(scheduler)

autoencoder.fit(train_df, train_df,batch_size=128,
                epochs=50, callbacks=[callback,tensorboard_callback])

%load_ext tensorboard

%tensorboard --logdir $run_log_dir

dbutils.fs.rm(experiment_log_dir.replace("/dbfs",""), recurse=True)