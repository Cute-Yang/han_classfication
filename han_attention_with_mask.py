import imp
from sys import set_asyncgen_hooks
from tensorflow import keras
import tensorflow as tf
import numpy as np
from tensorflow.keras import backend as K
from tensorflow.python.keras.backend import dtype


class WordAttention(keras.layers.Layer):
    def __init__(self,maxlen_sentence,maxlen_word,atten_size,**kwargs):
        self.init=keras.initializers.get("truncated_normal")
        self.maxlen_word=maxlen_word
        self.maxlen_sentence=maxlen_sentence
        self.atten_size=atten_size
        super(WordAttention,self).__init__()

    
    def build(self,input_shapes):
        input_shape=input_shapes[0]
        assert len(input_shape)==3,"we expected 3-D tensor"
        _,_,feature_dim=input_shape
        self.feature_dim=feature_dim

        self.kernel=tf.Variable(
            tf.random.truncated_normal(
                shape=(feature_dim,self.atten_size),
                mean=0,
                stddev=0.1,
                dtype=tf.float32
            ),
            name="word_attention_weight"
        )

        self.bias=tf.Variable(
            tf.random.truncated_normal(
                mean=0.0,
                stddev=0.1,
                shape=(self.atten_size,),
                dtype=tf.float32
            ),
            name="word_attention_bias"
        )

        self.u_context_variable=tf.Variable(
            tf.random.truncated_normal(
                mean=0.0,
                stddev=0.1,
                shape=(self.atten_size,1),
                dtype=tf.float32
            ),
            name="word_attention_u_context"
        )
    
        self.built=True
    
    
    def compute_mask(self, inputs, mask=None):
        pass

        
    def call(self,inputs_tensors):
        inputs,mask=inputs_tensors
        word_matmul=tf.matmul(inputs,self.kernel)
        word_bias_add=tf.nn.bias_add(word_matmul,self.bias)
        word_uit=tf.tanh(word_bias_add,name="word_attention_tanh")
        word_kv=tf.matmul(word_uit,self.u_context_variable)
        max_value=tf.reduce_max(word_kv)
        word_kv_exp=tf.exp(word_kv-max_value)+1e-7
        if mask is not None:
            word_kv_exp=word_kv_exp*mask
    
        word_kv_sum=tf.reduce_sum(word_kv_exp,axis=1,keepdims=True)
        word_prob=tf.divide(word_kv_exp,(word_kv_sum+1e-7),name="word_attention_alpha")
        word_prob_reshape=tf.reshape(word_prob,shape=(-1,self.maxlen_sentence,self.maxlen_word))
        output_wisemul=word_prob*inputs
        #along the last axis to sum
        output_reduce=tf.reduce_sum(output_wisemul,axis=1)
        sentence_vector=tf.reshape(output_reduce,shape=(-1,self.maxlen_sentence,self.feature_dim))
        return sentence_vector,word_prob_reshape
    

    def compute_output_shape(self, input_shape):
        batch_size=input_shape[0]/self.maxlen_sentence
        return (batch_size,self.maxlen_sentence,self.feature_dim)
    

class SentenceAttention(keras.layers.Layer):
    def __init__(self,atten_size:int,**kwargs):
        self.init=keras.initializers.get("truncated_normal")
        self.atten_size=atten_size
        super(SentenceAttention,self).__init__()
    

    def build(self, input_shapes):
        input_shape=input_shapes[0]
        assert len(input_shape)==3,"we expected 3-D tensor"
        _,_,feature_dim=input_shape
        self.feature_dim=feature_dim
        
        self.kernel=tf.Variable(
            tf.random.truncated_normal(
                shape=(feature_dim,self.atten_size),
                mean=0,
                stddev=0.1,
                dtype=tf.float32
            ),
            name="sentence_attention_weight"
        )

        self.bias=tf.Variable(
            tf.random.truncated_normal(
                mean=0.0,
                stddev=0.1,
                shape=(self.atten_size,),
                dtype=tf.float32
            ),
            name="sentence_attention_bias"
        )

        self.u_context_variable=tf.Variable(
            tf.random.truncated_normal(
                mean=0.0,
                stddev=0.1,
                shape=(self.atten_size,1),
                dtype=tf.float32
            ),
            name="sentence_attention_u_context"
        )
    
        self.built=True
    
    def compute_mask(self, inputs, mask=None):
        pass

    def call(self, inputs_tensors):
        inputs,mask=inputs_tensors
        sentence_matmul=tf.matmul(inputs,self.kernel,name="sentence_matmul")
        sentence_bias_add=tf.nn.bias_add(sentence_matmul,self.bias,name="sentence_uit")
        sentence_uit=tf.tanh(sentence_bias_add,name="sentence_uit")
        sentence_kv=tf.matmul(sentence_uit,self.u_context_variable)
        max_value=tf.reduce_sum(sentence_kv)
        sentence_kv_exp=tf.exp(sentence_kv-max_value)+1e-7
        if mask is not None:
            sentence_kv_exp=sentence_kv_exp*mask
    
        sentence_kv_sum=tf.reduce_sum(sentence_kv_exp,axis=1,keepdims=True)
        sentence_prob=tf.divide(sentence_kv_exp,(sentence_kv_sum+1e-7),name="sentence_attention_alpha")
        output_wisemul=inputs*sentence_prob
        sentence_prob_squeeze=tf.squeeze(sentence_prob,axis=-1)
        output_reduce=tf.reduce_sum(output_wisemul,axis=1)
        return output_reduce,sentence_prob_squeeze
    
    def compute_output_shape(self, input_shape):
        batch_size=input_shape[0]
        return (batch_size,self.feature_dim)


def self_loss(y_true,y_pred):
    #for numerical statility
    loss=tf.reduce_mean(
        tf.multiply(
            y_true,-tf.math.log(y_pred+1e-8)
        )
    )
    return loss

def create_HAN(maxlen_word,maxlen_sentence,vocab_size,embed_size,hidden_size,atten_size,num_classes,pretrian_embedding=None):
    doc_input=keras.layers.Input(shape=(maxlen_sentence,maxlen_word),name="doc_input",dtype=tf.int32)
    if pretrian_embedding is None:
        pretrian_embedding=np.random.randn(vocab_size,embed_size)

    word_mask_temp=tf.cast(tf.greater(doc_input,0,name="word_mask_greater"),tf.float32,name="cast_word_mask")
    word_mask=tf.reshape(word_mask_temp,shape=(-1,maxlen_word,1),name="word_mask")

    #if the last axis sum->0,means that this sentence is paded...hah
    sentence_mask_temp=tf.reduce_sum(word_mask_temp,axis=-1,name="sentence_mask_sum")
    sentence_mask=tf.cast(tf.greater(sentence_mask_temp,0,name="sentence_mask_greater"),tf.float32,name="cast_sentence_mask")
    sentence_mask=tf.reshape(sentence_mask,shape=(-1,maxlen_sentence,1),name="sentence_mask")
    
    

    #fine tune
    embedding_layer=keras.layers.Embedding(vocab_size,embed_size,weights=[pretrian_embedding],trainable=True)
    doc_embeded=embedding_layer(doc_input)
    
    #for word attention
    doc_embeded_reshape=tf.reshape(doc_embeded,shape=(-1,maxlen_word,embed_size),name="reshape_3-D_Tensor")
    word_gru_layer=keras.layers.Bidirectional(keras.layers.GRU(hidden_size,activation="tanh",return_sequences=True),name="word_bigru")
    word_gru=word_gru_layer(doc_embeded_reshape)

    word_attention_layer=WordAttention(
        maxlen_sentence=maxlen_sentence,
        maxlen_word=maxlen_word,
        atten_size=atten_size,
        name="word_attention"
    )

    sentence_matrix,word_prob=word_attention_layer([word_gru,word_mask])

    sentence_gru_layer=keras.layers.Bidirectional(keras.layers.GRU(hidden_size,activation="tanh",return_sequences=True),name="sentence_gru")
    sentence_gru=sentence_gru_layer(sentence_matrix)
    sentence_attention_layer=SentenceAttention(
        atten_size=atten_size,
        name="sentence_attention"
    )
    doc_matrix,sentence_prob=sentence_attention_layer([sentence_gru,sentence_mask])

    softmax_layer=keras.layers.Dense(units=num_classes,name="multi_class",activation="softmax")
    softmax_output=softmax_layer(doc_matrix)    
    
    model=keras.models.Model(doc_input,softmax_output,name="HAN-calssfication")

    model_with_prob=keras.models.Model([doc_input],[softmax_output,word_prob,sentence_prob],name="HAN-classfication_with_prob")
    
    adam=keras.optimizers.Adam(learning_rate=1e-3,decay=1e-2)
    model.compile(
        optimizer=adam,
        loss=self_loss,
        metrics=["accuracy"]
    )
    keras.utils.plot_model(model)

    return model,model_with_prob



if __name__=="__main__":
    model,model_with_prob=create_HAN(
        maxlen_word=10,
        maxlen_sentence=6,
        pretrian_embedding=None,
        vocab_size=12000,
        embed_size=200,
        hidden_size=256,
        atten_size=128,
        num_classes=18
    )
    model.summary()

    import numpy as np
    v1=np.random.randint(0,1,size=(100,6,10))
    print(v1[0])
    sb=model_with_prob.predict(v1)
    _,word_prob,sentence_prob=sb
    print(word_prob[0])
    print(sentence_prob[0])
    

        
            
        
