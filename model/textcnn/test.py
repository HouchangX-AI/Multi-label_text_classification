import argparse
from data_helper import preprocess,get_data
from tensorflow.keras.models import load_model
import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.utils.multiclass import unique_labels
import matplotlib.pyplot as plt
import os
from keras.metrics import categorical_accuracy
from pprint import pprint


def f1_np(y_true, y_pred):
    """F1 metric.

    Computes the micro_f1 and macro_f1, metrics for multi-label classification of
    how many relevant items are selected.
    """
    true_positives = np.sum(np.round(np.clip(y_true * y_pred, 0, 1)), axis=0)
    predicted_positives = np.sum(np.round(np.clip(y_pred, 0, 1)), axis=0)
    possible_positives = np.sum(np.round(np.clip(y_true, 0, 1)), axis=0)

    """Macro_F1 metric.
    """
    precision = true_positives / (predicted_positives + 1e-8)
    recall = true_positives / (possible_positives + 1e-8)
    macro_f1 = np.mean(2 * precision * recall / (precision + recall + 1e-8))

    """Micro_F1 metric.
    """
    precision = np.sum(true_positives) / np.sum(predicted_positives)
    recall = np.sum(true_positives) / np.sum(possible_positives)
    micro_f1 = 2 * precision * recall / (precision + recall + 1e-8)
    return micro_f1, macro_f1


def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax


def test(model, x_test, y_test):
    print("Test...")
    """
    y_pred_one_hot = model.predict(x=x_test, batch_size=1, verbose=1)
    y_pred = tf.math.argmax(y_pred_one_hot, axis=1)

    plot_confusion_matrix(y_test, y_pred, np.arange(args.num_classes))
    plt.savefig(os.path.join(args.results_dir, "confusion_matrix.pdf"))

    print('\nTest accuracy: {}\n'.format(accuracy_score(y_test, y_pred)))
    print('Classification report:')
    target_names = ['class {:d}'.format(i) for i in np.arange(args.num_classes)]
    print(classification_report(y_test, y_pred, target_names=target_names, digits=4))
    """
    y_pred = model.predict(x=x_test, batch_size=1, verbose=1)
    metrics = [f1_np]
    result = {}
    for func in metrics:
        result[func.__name__] = func(y_test, y_pred)
    pprint(result)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='This is the TextCNN test project.')
    parser.add_argument('results_dir', type=str, help='The results dir including log, model, vocabulary and some images.')
    parser.add_argument('-p', '--padding_size', default=200, type=int, help='Padding size of sentences.(default=200)')
    parser.add_argument('-c', '--num_classes', default=95, type=int, help='Number of target classes.(default=95)')
    args = parser.parse_args()
    print('Parameters:', args)

    # x_test, y_test = preprocess("./data/test_data.csv", os.path.join(args.results_dir, "vocab.json"),
    #                             args.padding_size, test=True)

    x_test, y_test = get_data('./data/baidu_95_test_x.npy', './data/baidu_95_test_y.npy')
    print("Loading model...")
    model = load_model(os.path.join(args.results_dir, 'TextCNN.h5'))
    test(model, x_test, y_test)
