from perceptron import Perceptron

w=[0.2,0.5,-0.3,0.1,0.4]
b=0.5

p=Perceptron(w,b)

z=p.pre_activation([7,2.4,5,81,3])
y=p.activation(z)
print(y)