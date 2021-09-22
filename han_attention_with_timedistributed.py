from tensorflow import keras
import tensorflow as tf
from tensorflow.keras import backend as K
import numpy as np

class Attention(keras.layers.Layer):
    def __init__(self,attention_size,use_bias:bool=True):
        self.attention_size=attention_size
        self.use_bias=use_bias
        super(Attention,self).__init__()

    
    def build(self,input_shape):
        print(input_shape)
        assert len(input_shape)==3,"we expected 3-D tensor,but you give {}-D".format(len(input_shape))

        _,time_step,feature_dims=input_shape
        #define our weight
        self.weight=self.add_weight(
            name="attention_weight",
            shape=(feature_dims,self.attention_size),
            dtype=tf.float32
        )

        self.bias=self.add_weight(
            name="attention_bias",
            shape=(self.attention_size,),
            dtype=tf.float32
        )

        self.u_context_vector=self.add_weight(
            name="attention_u_context_vector",
            shape=(self.attention_size,1)
        )

        self.built=True
        # self.trainable_weights=[self.weight,self.bias,self.u_context_vector]
    

    
    def compute_mask(self,inputs,mask=None):
        return mask

    def call(self,x,mask=None):
        uit=K.tanh(K.bias_add(K.dot(x,self.weight),self.bias))
        ait=K.dot(uit,self.u_context_vector)
        ait=K.squeeze(ait,-1)

        ait=K.exp(ait)
        ait=ait-K.max(ait)
        
        if mask is not None:
            ait=ait*K.cast(mask,K.floatx())
        
        ait=ait/K.cast(K.sum(ait,axis=1,keepdims=True)+K.epsilon(),K.floatx())
        ait=K.expand_dims(ait)
        
        #can broacast
        weighted_x=x*ait
        output=K.sum(weighted_x,axis=1)
        return output
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0],self.attention_size)


def create_han_model(max_word_size:int,max_sentence_size:int,pretrain_weights,vocab_size,embed_size,hidden_size,attention_size,num_classes):
    word_input=keras.layers.Input(shape=(max_word_size,),dtype=tf.int32)
    if pretrain_weights is None:
        pretrain_weights=np.random.random(
            size=(vocab_size,embed_size)
        )
    embedding_layer=keras.layers.Embedding(
        input_dim=vocab_size,
        output_dim=embed_size,
        weights=[pretrain_weights],
        trainable=True
    )

    embedded_sentence=embedding_layer(word_input)
    word_gru_layer=keras.layers.Bidirectional(
        keras.layers.GRU(
            hidden_size,return_sequences=True
        )
    )
    word_gru=word_gru_layer(embedded_sentence)
    word_atten_layer=Attention(
        attention_size=attention_size
    )
    word_atten=word_atten_layer(word_gru)
    word_encoder=keras.models.Model(word_input,word_atten)


    review_input=keras.layers.Input(shape=(max_sentence_size,max_word_size),dtype=tf.int32)
    review_encoder=keras.layers.TimeDistributed(word_encoder)(review_input)
    
    sentence_gru_layer=keras.layers.Bidirectional(
        keras.layers.GRU(
            hidden_size,
            return_sequences=True
        )
    )
    sentence_gru=sentence_gru_layer(review_encoder)
    sentence_atten_layer=Attention(
        attention_size=attention_size
    )
    sentence_atten=sentence_atten_layer(sentence_gru)


    logits=keras.layers.Dense(
        num_classes,
        activation="softmax"
    )(sentence_atten)
    model=keras.models.Model(review_input,logits)

    model.compile(
        loss="categorical_crossentropy",
        optimizer="rmsprop",
        metrics=["accuracy"]
    )

    return model


if __name__=="__main__":
    model=create_han_model(
        max_word_size=10,
        max_sentence_size=6,
        pretrain_weights=None,
        vocab_size=2000,
        embed_size=200,
        hidden_size=256,
        attention_size=128,
        num_classes=20
    )
    
    model.summary()

        


    

    

