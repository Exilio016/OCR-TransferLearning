---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.1'
      jupytext_version: 1.2.4
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

<!-- #region {"id": "cUvUJrvm_uZG", "colab_type": "text"} -->
# Trabalho 1
## OCR - Optical Character Recognizer

Aluno: Bruno Flávio Ferreira - 9791330

Na primeira parte do trabalho, foi utilizada a técnica de Transfer Learning, para treinar uma rede neural capaz de reconhecer caracteres. 
<!-- #endregion -->

```python id="Vt0fHJRht9hz" colab_type="code" colab={}
from __future__ import absolute_import, division, print_function, unicode_literals
import matplotlib.pylab as plt
import tensorflow as tf
import scipy.io as io
import tensorflow_hub as hub
from shutil import copyfile
from tensorflow.keras import layers
import pandas as pd
import PIL.Image as Image
import os
import numpy as np

```

<!-- #region {"id": "71ED6EzBGSrf", "colab_type": "text"} -->
Primeiro, carregamos o dataset Chars74K, 75% é carregado como dataset de treinamento, 25% como validação
<!-- #endregion -->

```python id="yQq6kzY8BnFK" colab_type="code" outputId="be24d06f-517b-4d91-c5df-1a0b1d2fc5e7" colab={"base_uri": "https://localhost:8080/", "height": 51}
IMAGE_SHAPE = (224, 224)
dataset_folder = './TrainSet'
image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1/255, validation_split=0.25)
image_data = image_generator.flow_from_directory(dataset_folder, target_size=IMAGE_SHAPE, subset='training')
image_validation = image_generator.flow_from_directory(dataset_folder  , target_size=IMAGE_SHAPE, subset='validation')
```

<!-- #region {"id": "GJg92ltXBZcc", "colab_type": "text"} -->
A Rede que será utilizada para o Transfer Learning é a Inception V3 da Google.

Primeiramente, foi carregado o Inception V3 e executado sobre algumas imagens do dataset para vermos as predições antes de realizarmos a técnica do Transfer Learning
<!-- #endregion -->

```python id="a6N3v6pht9h3" colab_type="code" colab={}
classifier_url ="https://tfhub.dev/google/tf2-preview/inception_v3/classification/4" #@param {type:"string"}
labels_path = tf.keras.utils.get_file('ImageNetLabels.txt','https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt')
imagenet_labels = np.array(open(labels_path).read().splitlines())
```

```python id="cx5QWgmmt9h7" colab_type="code" outputId="8aa80a9a-d79a-4ab1-b540-d85b40ab9164" colab={"base_uri": "https://localhost:8080/", "height": 775}
classifier = tf.keras.Sequential([
    hub.KerasLayer(classifier_url, input_shape=IMAGE_SHAPE+(3,))
])

for image_batch, label_batch in image_data:
  print("Image batch shape: ", image_batch.shape)
  print("Label batch shape: ", label_batch.shape)
  break

result_batch = classifier.predict(image_batch)
print(result_batch.shape)

predicted_class_names = imagenet_labels[np.argmax(result_batch, axis=-1)]

plt.figure(figsize=(10,9))
plt.subplots_adjust(hspace=0.5)
for n in range(30):
  plt.subplot(6,5,n+1)
  plt.imshow(image_batch[n])
  plt.title(predicted_class_names[n])
  plt.axis('off')
_ = plt.suptitle("ImageNet predictions")

plt.show()
```

<!-- #region {"id": "B3cWc3oqDS9d", "colab_type": "text"} -->
Agora, carregamos uma versão do Inception V3 sem a camada final de predição, para podermos treinar no novo dataset e obtermos a classificação desejada
<!-- #endregion -->

```python id="5CJ1LTI6t9h_" colab_type="code" outputId="6d3fb302-81a4-4834-c20a-efc01a581dea" colab={"base_uri": "https://localhost:8080/", "height": 238}
feature_extractor_url = "https://tfhub.dev/google/tf2-preview/inception_v3/feature_vector/4" #@param {type:"string"}
feature_extractor_layer = hub.KerasLayer(feature_extractor_url,
                                         input_shape=(224,224,3))
feature_batch = feature_extractor_layer(image_batch)
print(feature_batch.shape)
feature_extractor_layer.trainable = False
model = tf.keras.Sequential([
  feature_extractor_layer,
  layers.Dense(image_data.num_classes, activation='softmax')
])
model.summary()

```

<!-- #region {"id": "SNNR2ad8Fp5d", "colab_type": "text"} -->
Agora, nós treinamos o modelo com o dataset Chars74K
<!-- #endregion -->

```python id="3JXHspabt9iD" colab_type="code" outputId="4e1bcb93-000c-44c7-dd3a-e9ea27d7b5c2" colab={"base_uri": "https://localhost:8080/", "height": 782}
predictions = model(image_batch)
model.compile(
  optimizer=tf.keras.optimizers.Adam(),
  loss='categorical_crossentropy',
  metrics=['acc'])

class CollectBatchStats(tf.keras.callbacks.Callback):
  def __init__(self):
    self.batch_losses = []
    self.batch_acc = []

  def on_train_batch_end(self, batch, logs=None):
    self.batch_losses.append(logs['loss'])
    self.batch_acc.append(logs['acc'])
    self.model.reset_metrics()

steps_per_epoch = np.ceil(image_data.samples/image_data.batch_size)

batch_stats_callback = CollectBatchStats()

history = model.fit_generator(image_data, epochs=15,
                              steps_per_epoch=steps_per_epoch,
                              validation_data=image_validation,
                              callbacks = [batch_stats_callback])

```

<!-- #region {"id": "8cFrC8EvWbcB", "colab_type": "text"} -->
Com o modelo treinado, podemos visualizar os gráficos de como as medidas de Loss e Accuracy se modificaram com os passos do treinamento
<!-- #endregion -->

```python id="zY4nda22WY5F" colab_type="code" colab={"base_uri": "https://localhost:8080/", "height": 566} outputId="7fd25ccd-7ffe-4b6e-e3f2-de9115f40e07"
plt.figure()
plt.ylabel("Loss")
plt.xlabel("Training Steps")
plt.ylim([0,2])
plt.plot(batch_stats_callback.batch_losses)

plt.figure()
plt.ylabel("Accuracy")
plt.xlabel("Training Steps")
plt.ylim([0,1])
plt.plot(batch_stats_callback.batch_acc)

```

<!-- #region {"id": "AXrjFQhDXq4X", "colab_type": "text"} -->
Após todos os passos completos, podemos executar a predição sobre as mesmas imagens que foram classificadas pelo ImageNet anteriormente e vermos os resultados:
<!-- #endregion -->

```python id="LKVjCDFkt9iG" colab_type="code" colab={"base_uri": "https://localhost:8080/", "height": 602} outputId="be150df2-e63d-44cd-f9d9-eda73466393f"
class_names = sorted(image_data.class_indices.items(), key=lambda pair:pair[1])
class_names = np.array([key.title() for key, value in class_names])
class_names
predicted_batch = model.predict(image_batch)
predicted_id = np.argmax(predicted_batch, axis=-1)
predicted_label_batch = class_names[predicted_id]
label_id = np.argmax(label_batch, axis=-1)

plt.figure(figsize=(10,9))
plt.subplots_adjust(hspace=0.5)
for n in range(30):
  plt.subplot(6,5,n+1)
  plt.imshow(image_batch[n])
  color = "green" if predicted_id[n] == label_id[n] else "red"
  plt.title(predicted_label_batch[n].title(), color=color)
  plt.axis('off')
_ = plt.suptitle("Model predictions (green: correct, red: incorrect)")
```

<!-- #region {"id": "xbyHRZHvapsb", "colab_type": "text"} -->
Também, podemos executar a predição em imagens do dataset de validação:
<!-- #endregion -->

```python id="aSCxD6lNt9iK" colab_type="code" colab={"base_uri": "https://localhost:8080/", "height": 636} outputId="a6fd3e45-1e7b-4a5f-cadc-8836b30f62ec"
for validation_batch, label_batch in image_validation:
  print("Validation batch shape: ", validation_batch.shape)
  print("Label batch shape: ", label_batch.shape)
  break  

predicted_batch = model.predict(validation_batch)
predicted_id = np.argmax(predicted_batch, axis=-1)
predicted_label_batch = class_names[predicted_id]
label_id = np.argmax(label_batch, axis=-1)

plt.figure(figsize=(10,9))
plt.subplots_adjust(hspace=0.5)
for n in range(30):
  plt.subplot(6,5,n+1)
  plt.imshow(validation_batch[n])
  color = "green" if predicted_id[n] == label_id[n] else "red"
  plt.title(predicted_label_batch[n].title(), color=color)
  plt.axis('off')
_ = plt.suptitle("Model predictions - validation set (green: correct, red: incorrect)")


```
