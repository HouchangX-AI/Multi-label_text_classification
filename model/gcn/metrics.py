import tensorflow as tf

def masked_softmax_cross_entropy(preds, labels, mask, multi_label=False):
    """Softmax cross-entropy loss with masking."""
    print(preds)
    if multi_label:
        loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=preds, labels=labels)
        loss = tf.reduce_mean(loss, axis=1)

        mask = tf.cast(mask, dtype=tf.float32)
        mask /= tf.reduce_mean(mask)
        loss *= mask
        return tf.reduce_mean(loss)
    else:
        loss = tf.nn.softmax_cross_entropy_with_logits(logits=preds, labels=labels)
        mask = tf.cast(mask, dtype=tf.float32)
        mask /= tf.reduce_mean(mask)
        loss *= mask
        return tf.reduce_mean(loss)



def masked_accuracy(preds, labels, mask, multi_label=False):
    """Accuracy with masking."""
    if multi_label:
        predictions = tf.where(tf.nn.sigmoid(preds) > 0.5, tf.ones(tf.shape(preds)), tf.zeros(tf.shape(preds)))
        t = tf.multiply(predictions, labels)
        true_positives = tf.reduce_sum(t, axis=0)
        predicted_positives = tf.reduce_sum(predictions, axis=0)
        possible_positives = tf.reduce_sum(labels, axis=0)

        # Macro_F1 metric.
        precision = true_positives / (predicted_positives + 1e-8)
        recall = true_positives / (possible_positives + 1e-8)
        macro_f1 = tf.reduce_mean(2 * precision * recall / (precision + recall + 1e-8))

        # Micro_F1 metric.
        prec = tf.reduce_sum(true_positives) / tf.reduce_sum(predicted_positives)
        reca = tf.reduce_sum(true_positives) / tf.reduce_sum(possible_positives)
        micro_f1 = 2 * prec * reca / (prec + reca + 1e-8)
        return micro_f1 #, macro_f1

    else:
        correct_prediction = tf.equal(tf.argmax(preds, 1), tf.argmax(labels, 1))
        accuracy_all = tf.cast(correct_prediction, tf.float32)
        mask = tf.cast(mask, dtype=tf.float32)
        mask /= tf.reduce_mean(mask)
        accuracy_all *= mask
        return tf.reduce_mean(accuracy_all)