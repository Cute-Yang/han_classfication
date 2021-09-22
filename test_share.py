from tensorflow import keras

import numpy as np

def make_simple():
    sb=keras.layers.Input(shape=(128,))
    fw=keras.layers.Dense(64)(sb)
    output=keras.layers.Dense(3,activation="softmax")(fw)

    m1=keras.models.Model(sb,output)
    
    m2=keras.models.Model(sb,output)

    m1.compile(optimizer="adam",loss="categorical_crossentropy",metrics=["accuracy"])
    m1.summary()
    return m1,m2


x=np.random.random(size=(100,128))
y=np.zeros(shape=(100,3),dtype=np.float32)

for i in range(100):
    s=np.random.randint(0,3)
    y[i,s]=1.0

m1,m2=make_simple()

m1.fit(x,y,batch_size=20,epochs=100)

print(m1.predict(x)[0])
print(m2.predict(x)[0])

m2.save("sb.h5")

m3=keras.models.load_model("sb.h5")
print(m3.predict(x)[0])
