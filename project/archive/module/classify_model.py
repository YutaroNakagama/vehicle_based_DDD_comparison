
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Bidirectional, LSTM, Dense, Attention, Concatenate, Flatten
from sklearn.model_selection import train_test_split

def get_result_svm(X,Y): 
    X_train, X_test, y_train, y_test = train_test_split(X, Y, stratify=Y, train_size=0.3, random_state=0)

#    print("training",len(y_train))
#    print("True" ,(y_train==True).sum().sum())
#    print("False",(y_train==False).sum().sum())
#    print("test",len(y_test))
#    print("True" ,(y_test==True).sum().sum())
#    print("False",(y_test==False).sum().sum())

    from sklearn import svm
    clf = svm.SVC(kernel='rbf')
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    from sklearn import metrics
    acc    = metrics.accuracy_score(y_test, y_pred)
    prec   = metrics.precision_score(y_test, y_pred)
    recall = metrics.recall_score(y_test, y_pred)
#    print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
#    print("Precision:",metrics.precision_score(y_test, y_pred))
#    print("Recall:",metrics.recall_score(y_test, y_pred))
#    print("confusion matrix:\n",metrics.confusion_matrix(y_test, y_pred))

    from sklearn.metrics import RocCurveDisplay, roc_curve
    y_score = clf.decision_function(X_test)

    auc = metrics.roc_auc_score(y_test, y_score)
#    print("AUC:",metrics.roc_auc_score(y_test, y_score))
    fpr, tpr, _ = roc_curve(y_test, y_score, pos_label=clf.classes_[1])
    #roc_display = RocCurveDisplay(fpr=fpr, tpr=tpr).plot()
    #plt.show()

    from sklearn.metrics import PrecisionRecallDisplay, precision_recall_curve
    prec, recall, _ = precision_recall_curve(y_test, y_score, pos_label=clf.classes_[1])
    #pr_display = PrecisionRecallDisplay(precision=prec, recall=recall).plot()
    #plt.show()

    return acc, prec, recall, auc

class AttentionLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # Trainable weights for attention mechanism
        self.W = self.add_weight(name="att_weight", shape=(input_shape[-1], input_shape[-1]),
                                 initializer="glorot_uniform", trainable=True)
        self.b = self.add_weight(name="att_bias", shape=(input_shape[-1],),
                                 initializer="zeros", trainable=True)
        self.u = self.add_weight(name="att_u", shape=(input_shape[-1],),
                                 initializer="glorot_uniform", trainable=True)

        super(AttentionLayer, self).build(input_shape)

    def call(self, inputs):
        # Score computation
        v = tf.tanh(tf.tensordot(inputs, self.W, axes=1) + self.b)
        vu = tf.tensordot(v, self.u, axes=1)
        alphas = tf.nn.softmax(vu)

        # Weighted sum of input
        output = tf.reduce_sum(inputs * tf.expand_dims(alphas, -1), axis=1)
        return output, alphas

# Sample Bi-LSTM model with Attention
def create_model(input_shape):
    inputs = Input(shape=input_shape)
    lstm_out = Bidirectional(LSTM(64, return_sequences=True))(inputs)
    #flatten_out = Flatten()(lstm_out)
    attention_out, attention_weights = AttentionLayer()(lstm_out)
    dense_out = Dense(16)(attention_out)
    outputs = Dense(1, activation='sigmoid')(dense_out)
    model = Model(inputs, outputs)
    return model

def model_Wang2022(feats):
    # Set input shape and compile the model
    input_shape = (1, len(feats))  # For example, sequence length = 100, features per step = 50
    model = create_model(input_shape)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()
    return model
