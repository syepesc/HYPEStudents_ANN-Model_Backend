import pandas as pd, numpy as np, tensorflow as tf
from imblearn.over_sampling import ADASYN
from sklearn.metrics import classification_report

data = pd.read_csv('data')
train_x, train_y = data[data['INTAKE TERM CODE'] < 2020].drop(columns='failure'), data[data['INTAKE TERM CODE'] < 2020]['failure']
test_x, test_y = data[data['INTAKE TERM CODE'] >= 2020].drop(columns='failure'), data[data['INTAKE TERM CODE'] >= 2020]['failure']
test_x, test_y = ADASYN().fit_resample(test_x, test_y)  # test set is unbalanced
# one hot encoding
train_y, test_y = tf.keras.utils.to_categorical(train_y, dtype=int), tf.keras.utils.to_categorical(test_y, dtype=int)

hidden_units = [64, 64]
def create_deep_and_cross_model():
	input_layer = tf.keras.Input(shape=train_x.shape[1])

	cross = input_layer
	for _ in hidden_units:
		units = input_layer.shape[-1]
		x = tf.keras.layers.Dense(units)(input_layer)
		cross = input_layer * x + cross
	cross = tf.keras.layers.BatchNormalization()(cross)

	deep = input_layer
	for units in hidden_units:
		deep = tf.keras.layers.Dense(units)(deep)
		deep = tf.keras.layers.BatchNormalization()(deep)
		deep = tf.keras.layers.ReLU()(deep)
		deep = tf.keras.layers.Dropout(.5)(deep)

	merged = tf.keras.layers.concatenate([cross, deep])
	output_layer = tf.keras.layers.Dense(units=2, activation="sigmoid")(merged)
	model = tf.keras.Model(inputs=input_layer, outputs=output_layer)
	model.compile(loss='binary_crossentropy', optimizer='Adamax', metrics=['accuracy'])
	return model

model = create_deep_and_cross_model()
model.fit(train_x, train_y, validation_data=(test_x, test_y), epochs=100)
predict = model.predict(test_x)
predict, test_y = [np.argmax(one_hot) for one_hot in predict], [np.argmax(one_hot) for one_hot in test_y]
print(classification_report(test_y, predict))
model.save('deep and cross network')