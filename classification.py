{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Using TensorFlow backend.\n"
     ]
    }
   ],
   "source": [
    "from keras.preprocessing.image import ImageDataGenerator\n",
    "from keras.models import Sequential\n",
    "from keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense\n",
    "from keras import backend as k\n",
    "import numpy as np\n",
    "from keras.preprocessing import image"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [],
   "source": [
    "img_width, img_height = 150, 150"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [],
   "source": [
    "train_data_dir = r'D:\\\\AIEngineer\\\\DL\\\\data\\\\train'\n",
    "test_data_dir = r'D:\\\\AIEngineer\\\\DL\\\\data\\\\test'"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [],
   "source": [
    "np_train_samples = 1000\n",
    "np_test_samples = 100\n",
    "epocs = 50\n",
    "batch_size = 20"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [],
   "source": [
    "if k.image_data_format() == 'channels_first':\n",
    "    input_shape = (3, img_width, img_height)\n",
    "else:\n",
    "    input_shape = (img_width, img_height, 3)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [],
   "source": [
    "train_datagen = ImageDataGenerator(\n",
    "rescale = 1. / 255,\n",
    "shear_range = 0.2,\n",
    "zoom_range = 0.2,\n",
    "horizontal_flip = True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [],
   "source": [
    "test_datagen = ImageDataGenerator(rescale=1. /255)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Found 1050 images belonging to 2 classes.\n"
     ]
    }
   ],
   "source": [
    "train_generator = train_datagen.flow_from_directory(\n",
    "train_data_dir,\n",
    "target_size = (img_width, img_height),\n",
    "batch_size = batch_size,\n",
    "class_mode = 'binary')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Found 20 images belonging to 2 classes.\n"
     ]
    }
   ],
   "source": [
    "test_generator = test_datagen.flow_from_directory(\n",
    "test_data_dir,\n",
    "target_size = (img_width, img_height),\n",
    "batch_size = batch_size,\n",
    "class_mode = 'binary')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {},
   "outputs": [],
   "source": [
    "model = Sequential()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {},
   "outputs": [],
   "source": [
    "model.add(Conv2D(32, (5,5), input_shape=input_shape))\n",
    "model.add(Activation('relu'))\n",
    "model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {},
   "outputs": [],
   "source": [
    "model.add(Conv2D(64, (5,5), input_shape=input_shape))\n",
    "model.add(Activation('relu'))\n",
    "model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "metadata": {},
   "outputs": [],
   "source": [
    "model.add(Flatten())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "metadata": {},
   "outputs": [],
   "source": [
    "model.add(Dense(64))\n",
    "model.add(Activation('relu'))\n",
    "model.add(Dropout(0.4))\n",
    "model.add(Dense(1))\n",
    "model.add(Activation('sigmoid'))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Model: \"sequential_1\"\n",
      "_________________________________________________________________\n",
      "Layer (type)                 Output Shape              Param #   \n",
      "=================================================================\n",
      "conv2d_1 (Conv2D)            (None, 146, 146, 32)      2432      \n",
      "_________________________________________________________________\n",
      "activation_1 (Activation)    (None, 146, 146, 32)      0         \n",
      "_________________________________________________________________\n",
      "max_pooling2d_1 (MaxPooling2 (None, 73, 73, 32)        0         \n",
      "_________________________________________________________________\n",
      "conv2d_2 (Conv2D)            (None, 69, 69, 64)        51264     \n",
      "_________________________________________________________________\n",
      "activation_2 (Activation)    (None, 69, 69, 64)        0         \n",
      "_________________________________________________________________\n",
      "max_pooling2d_2 (MaxPooling2 (None, 34, 34, 64)        0         \n",
      "_________________________________________________________________\n",
      "flatten_1 (Flatten)          (None, 73984)             0         \n",
      "_________________________________________________________________\n",
      "dense_1 (Dense)              (None, 64)                4735040   \n",
      "_________________________________________________________________\n",
      "activation_3 (Activation)    (None, 64)                0         \n",
      "_________________________________________________________________\n",
      "dropout_1 (Dropout)          (None, 64)                0         \n",
      "_________________________________________________________________\n",
      "dense_2 (Dense)              (None, 1)                 65        \n",
      "_________________________________________________________________\n",
      "activation_4 (Activation)    (None, 1)                 0         \n",
      "=================================================================\n",
      "Total params: 4,788,801\n",
      "Trainable params: 4,788,801\n",
      "Non-trainable params: 0\n",
      "_________________________________________________________________\n"
     ]
    }
   ],
   "source": [
    "model.summary()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "metadata": {},
   "outputs": [],
   "source": [
    "model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 1/50\n",
      "50/50 [==============================] - 54s 1s/step - loss: 0.9180 - accuracy: 0.5566 - val_loss: 0.6594 - val_accuracy: 0.6000\n",
      "Epoch 2/50\n",
      "50/50 [==============================] - 56s 1s/step - loss: 0.6779 - accuracy: 0.6263 - val_loss: 0.6481 - val_accuracy: 0.6500\n",
      "Epoch 3/50\n",
      "25/50 [==============>...............] - ETA: 26s - loss: 0.6466 - accuracy: 0.6840"
     ]
    },
    {
     "ename": "KeyboardInterrupt",
     "evalue": "",
     "output_type": "error",
     "traceback": [
      "\u001b[1;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[1;31mKeyboardInterrupt\u001b[0m                         Traceback (most recent call last)",
      "\u001b[1;32m<ipython-input-17-9f36c2a27021>\u001b[0m in \u001b[0;36m<module>\u001b[1;34m\u001b[0m\n\u001b[0;32m      4\u001b[0m \u001b[0mepochs\u001b[0m\u001b[1;33m=\u001b[0m\u001b[0mepocs\u001b[0m\u001b[1;33m,\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m      5\u001b[0m \u001b[0mvalidation_data\u001b[0m\u001b[1;33m=\u001b[0m\u001b[0mtest_generator\u001b[0m\u001b[1;33m,\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m----> 6\u001b[1;33m validation_steps=np_test_samples // batch_size)\n\u001b[0m",
      "\u001b[1;32mD:\\anaconda\\envs\\tensorflow\\lib\\site-packages\\keras\\legacy\\interfaces.py\u001b[0m in \u001b[0;36mwrapper\u001b[1;34m(*args, **kwargs)\u001b[0m\n\u001b[0;32m     89\u001b[0m                 warnings.warn('Update your `' + object_name + '` call to the ' +\n\u001b[0;32m     90\u001b[0m                               'Keras 2 API: ' + signature, stacklevel=2)\n\u001b[1;32m---> 91\u001b[1;33m             \u001b[1;32mreturn\u001b[0m \u001b[0mfunc\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;33m*\u001b[0m\u001b[0margs\u001b[0m\u001b[1;33m,\u001b[0m \u001b[1;33m**\u001b[0m\u001b[0mkwargs\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m     92\u001b[0m         \u001b[0mwrapper\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0m_original_function\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mfunc\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m     93\u001b[0m         \u001b[1;32mreturn\u001b[0m \u001b[0mwrapper\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;32mD:\\anaconda\\envs\\tensorflow\\lib\\site-packages\\keras\\engine\\training.py\u001b[0m in \u001b[0;36mfit_generator\u001b[1;34m(self, generator, steps_per_epoch, epochs, verbose, callbacks, validation_data, validation_steps, validation_freq, class_weight, max_queue_size, workers, use_multiprocessing, shuffle, initial_epoch)\u001b[0m\n\u001b[0;32m   1730\u001b[0m             \u001b[0muse_multiprocessing\u001b[0m\u001b[1;33m=\u001b[0m\u001b[0muse_multiprocessing\u001b[0m\u001b[1;33m,\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m   1731\u001b[0m             \u001b[0mshuffle\u001b[0m\u001b[1;33m=\u001b[0m\u001b[0mshuffle\u001b[0m\u001b[1;33m,\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m-> 1732\u001b[1;33m             initial_epoch=initial_epoch)\n\u001b[0m\u001b[0;32m   1733\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m   1734\u001b[0m     \u001b[1;33m@\u001b[0m\u001b[0minterfaces\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mlegacy_generator_methods_support\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;32mD:\\anaconda\\envs\\tensorflow\\lib\\site-packages\\keras\\engine\\training_generator.py\u001b[0m in \u001b[0;36mfit_generator\u001b[1;34m(model, generator, steps_per_epoch, epochs, verbose, callbacks, validation_data, validation_steps, validation_freq, class_weight, max_queue_size, workers, use_multiprocessing, shuffle, initial_epoch)\u001b[0m\n\u001b[0;32m    218\u001b[0m                                             \u001b[0msample_weight\u001b[0m\u001b[1;33m=\u001b[0m\u001b[0msample_weight\u001b[0m\u001b[1;33m,\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m    219\u001b[0m                                             \u001b[0mclass_weight\u001b[0m\u001b[1;33m=\u001b[0m\u001b[0mclass_weight\u001b[0m\u001b[1;33m,\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m--> 220\u001b[1;33m                                             reset_metrics=False)\n\u001b[0m\u001b[0;32m    221\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m    222\u001b[0m                 \u001b[0mouts\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mto_list\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mouts\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;32mD:\\anaconda\\envs\\tensorflow\\lib\\site-packages\\keras\\engine\\training.py\u001b[0m in \u001b[0;36mtrain_on_batch\u001b[1;34m(self, x, y, sample_weight, class_weight, reset_metrics)\u001b[0m\n\u001b[0;32m   1512\u001b[0m             \u001b[0mins\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mx\u001b[0m \u001b[1;33m+\u001b[0m \u001b[0my\u001b[0m \u001b[1;33m+\u001b[0m \u001b[0msample_weights\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m   1513\u001b[0m         \u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0m_make_train_function\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m-> 1514\u001b[1;33m         \u001b[0moutputs\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mtrain_function\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mins\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m   1515\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m   1516\u001b[0m         \u001b[1;32mif\u001b[0m \u001b[0mreset_metrics\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;32mD:\\anaconda\\envs\\tensorflow\\lib\\site-packages\\tensorflow_core\\python\\keras\\backend.py\u001b[0m in \u001b[0;36m__call__\u001b[1;34m(self, inputs)\u001b[0m\n\u001b[0;32m   3725\u001b[0m         \u001b[0mvalue\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mmath_ops\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mcast\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mvalue\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mtensor\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mdtype\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m   3726\u001b[0m       \u001b[0mconverted_inputs\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mappend\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mvalue\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m-> 3727\u001b[1;33m     \u001b[0moutputs\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0m_graph_fn\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;33m*\u001b[0m\u001b[0mconverted_inputs\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m   3728\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m   3729\u001b[0m     \u001b[1;31m# EagerTensor.numpy() will often make a copy to ensure memory safety.\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;32mD:\\anaconda\\envs\\tensorflow\\lib\\site-packages\\tensorflow_core\\python\\eager\\function.py\u001b[0m in \u001b[0;36m__call__\u001b[1;34m(self, *args, **kwargs)\u001b[0m\n\u001b[0;32m   1549\u001b[0m       \u001b[0mTypeError\u001b[0m\u001b[1;33m:\u001b[0m \u001b[0mFor\u001b[0m \u001b[0minvalid\u001b[0m \u001b[0mpositional\u001b[0m\u001b[1;33m/\u001b[0m\u001b[0mkeyword\u001b[0m \u001b[0margument\u001b[0m \u001b[0mcombinations\u001b[0m\u001b[1;33m.\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m   1550\u001b[0m     \"\"\"\n\u001b[1;32m-> 1551\u001b[1;33m     \u001b[1;32mreturn\u001b[0m \u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0m_call_impl\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0margs\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mkwargs\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m   1552\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m   1553\u001b[0m   \u001b[1;32mdef\u001b[0m \u001b[0m_call_impl\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mself\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0margs\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mkwargs\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mcancellation_manager\u001b[0m\u001b[1;33m=\u001b[0m\u001b[1;32mNone\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;32mD:\\anaconda\\envs\\tensorflow\\lib\\site-packages\\tensorflow_core\\python\\eager\\function.py\u001b[0m in \u001b[0;36m_call_impl\u001b[1;34m(self, args, kwargs, cancellation_manager)\u001b[0m\n\u001b[0;32m   1589\u001b[0m       raise TypeError(\"Keyword arguments {} unknown. Expected {}.\".format(\n\u001b[0;32m   1590\u001b[0m           list(kwargs.keys()), list(self._arg_keywords)))\n\u001b[1;32m-> 1591\u001b[1;33m     \u001b[1;32mreturn\u001b[0m \u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0m_call_flat\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0margs\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mcaptured_inputs\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mcancellation_manager\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m   1592\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m   1593\u001b[0m   \u001b[1;32mdef\u001b[0m \u001b[0m_filtered_call\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mself\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0margs\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mkwargs\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;32mD:\\anaconda\\envs\\tensorflow\\lib\\site-packages\\tensorflow_core\\python\\eager\\function.py\u001b[0m in \u001b[0;36m_call_flat\u001b[1;34m(self, args, captured_inputs, cancellation_manager)\u001b[0m\n\u001b[0;32m   1690\u001b[0m       \u001b[1;31m# No tape is watching; skip to running the function.\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m   1691\u001b[0m       return self._build_call_outputs(self._inference_function.call(\n\u001b[1;32m-> 1692\u001b[1;33m           ctx, args, cancellation_manager=cancellation_manager))\n\u001b[0m\u001b[0;32m   1693\u001b[0m     forward_backward = self._select_forward_and_backward_functions(\n\u001b[0;32m   1694\u001b[0m         \u001b[0margs\u001b[0m\u001b[1;33m,\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;32mD:\\anaconda\\envs\\tensorflow\\lib\\site-packages\\tensorflow_core\\python\\eager\\function.py\u001b[0m in \u001b[0;36mcall\u001b[1;34m(self, ctx, args, cancellation_manager)\u001b[0m\n\u001b[0;32m    543\u001b[0m               \u001b[0minputs\u001b[0m\u001b[1;33m=\u001b[0m\u001b[0margs\u001b[0m\u001b[1;33m,\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m    544\u001b[0m               \u001b[0mattrs\u001b[0m\u001b[1;33m=\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;34m\"executor_type\"\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mexecutor_type\u001b[0m\u001b[1;33m,\u001b[0m \u001b[1;34m\"config_proto\"\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mconfig\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m,\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m--> 545\u001b[1;33m               ctx=ctx)\n\u001b[0m\u001b[0;32m    546\u001b[0m         \u001b[1;32melse\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m    547\u001b[0m           outputs = execute.execute_with_cancellation(\n",
      "\u001b[1;32mD:\\anaconda\\envs\\tensorflow\\lib\\site-packages\\tensorflow_core\\python\\eager\\execute.py\u001b[0m in \u001b[0;36mquick_execute\u001b[1;34m(op_name, num_outputs, inputs, attrs, ctx, name)\u001b[0m\n\u001b[0;32m     59\u001b[0m     tensors = pywrap_tensorflow.TFE_Py_Execute(ctx._handle, device_name,\n\u001b[0;32m     60\u001b[0m                                                \u001b[0mop_name\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0minputs\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mattrs\u001b[0m\u001b[1;33m,\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m---> 61\u001b[1;33m                                                num_outputs)\n\u001b[0m\u001b[0;32m     62\u001b[0m   \u001b[1;32mexcept\u001b[0m \u001b[0mcore\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0m_NotOkStatusException\u001b[0m \u001b[1;32mas\u001b[0m \u001b[0me\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m     63\u001b[0m     \u001b[1;32mif\u001b[0m \u001b[0mname\u001b[0m \u001b[1;32mis\u001b[0m \u001b[1;32mnot\u001b[0m \u001b[1;32mNone\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;31mKeyboardInterrupt\u001b[0m: "
     ]
    }
   ],
   "source": [
    "model.fit_generator(\n",
    "train_generator,\n",
    "steps_per_epoch=np_train_samples // batch_size,\n",
    "epochs=epocs,\n",
    "validation_data=test_generator,\n",
    "validation_steps=np_test_samples // batch_size)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "model.save_weights('img_pred.h5')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "img_pred = image.load_img('D:\\\\AIEngineer\\\\DL\\\\data\\\\test\\\\dogs\\\\110.jpg', target_size=(150, 150))\n",
    "img_pred = image.img_to_array(img_pred)\n",
    "img_pred = np.expand_dims(img_pred, axis=0)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "rslt = model.predict(img_pred)\n",
    "print(rslt)\n",
    "\n",
    "if rslt[0][0] == 1:\n",
    "    prediction = \"dog\"\n",
    "else:\n",
    "    prediction = \"cat\"\n",
    "\n",
    "print(prediction)"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.9"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
