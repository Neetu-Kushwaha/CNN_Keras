import numpy as np
import tensorflow as tf
import pandas as pd
import pickle
from sklearn import preprocessing


def extract_activations(model, input_test, target_test, index_layer):
    """Extracts and stores activation values of intermediate layers for test images."""
    activation_model = tf.keras.models.Model(inputs=model.input,
                                             outputs=[layer.output for layer in model.layers[:-1]])

    activation_test_images = []
    class_neuron = []
    predicted_class = []
    index_right = []
    index_wrong = []

    for i in range(len(input_test)):
        x = input_test[i].reshape((1,) + input_test[i].shape)
        pred_class = np.argmax(model.predict(x), axis=-1)
        predicted_class.append(pred_class)

        if target_test[i] == pred_class:
            class_neuron.append("Right")
            index_right.append(i)
        else:
            class_neuron.append("Wrong")
            index_wrong.append(i)

        activations = activation_model.predict(x)
        activation_test_images.append(activations)

    activation_layer = activation_test_images

    dd = []
    for i in range(len(input_test)):
        array_acti = []
        for j in index_layer:
            a = activation_layer[i][j].flatten()
            array_acti.append(a)
        dd.append(np.hstack(array_acti))

    activation_layer_array = np.array(dd)
    n_neuron = len(activation_layer_array[1])
    print("Total number of neurons :", n_neuron)

    def Neuron_Name_Transaction(n_neuron):
        neuron_name = []
        for j in range(n_neuron):
            neuron_name.append(str(j + 1))
        return neuron_name

    neuron_names = Neuron_Name_Transaction(n_neuron)
    dfObj_Neuron_activation = pd.DataFrame(activation_layer_array, columns=neuron_names)
    print("Activation Dataset Shape :", dfObj_Neuron_activation.shape)

    lb = preprocessing.LabelBinarizer()
    class_label = lb.fit_transform(class_neuron)

    pickle.dump(dfObj_Neuron_activation, open("Activation_dataset.p", "wb"))
    pickle.dump([class_neuron, class_label], open("classes_activation.p", "wb"))

    return activation_test_images, class_neuron, predicted_class, index_right, index_wrong
