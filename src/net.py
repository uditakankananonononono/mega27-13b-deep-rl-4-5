"""Small two-hidden-layer Q network with explicit Adam and Huber regression."""
import numpy as np
class QNet:
    def __init__(self, n_in, n_out, width=64, seed=0):
        g=np.random.default_rng(seed); self.w=[g.normal(0,np.sqrt(2/a),(a,b)) for a,b in ((n_in,width),(width,width),(width,n_out))]; self.b=[np.zeros(b) for b in (width,width,n_out)]
        self.m=[np.zeros_like(p) for p in self.w+self.b];self.v=[np.zeros_like(p) for p in self.w+self.b];self.t=0
    def predict(self,x):
        x=np.asarray(x,dtype=float);h=np.maximum(x@self.w[0]+self.b[0],0);h=np.maximum(h@self.w[1]+self.b[1],0);return h@self.w[2]+self.b[2]
    def fit(self,x,y,epochs=80,batch=128,lr=.002,seed=0):
        x=np.asarray(x,dtype=float);y=np.asarray(y,dtype=float);rng=np.random.default_rng(seed);loss=[]
        for epoch in range(epochs):
            for idx in np.array_split(rng.permutation(len(x)),max(1,int(np.ceil(len(x)/batch)))):
                a=x[idx]; target=y[idx]
                z1=a@self.w[0]+self.b[0];h1=np.maximum(z1,0);z2=h1@self.w[1]+self.b[1];h2=np.maximum(z2,0);p=h2@self.w[2]+self.b[2]
                e=p-target;d=np.clip(e,-1,1)/np.prod(e.shape);dh2=d@self.w[2].T;dz2=dh2*(z2>0);dh1=dz2@self.w[1].T;dz1=dh1*(z1>0)
                grads=[a.T@dz1,h1.T@dz2,h2.T@d,dz1.sum(0),dz2.sum(0),d.sum(0)]
                self.t+=1
                for i,(param,g) in enumerate(zip(self.w+self.b,grads)):
                    self.m[i]=.9*self.m[i]+.1*g;self.v[i]=.999*self.v[i]+.001*g*g
                    param-=lr*(self.m[i]/(1-.9**self.t))/(np.sqrt(self.v[i]/(1-.999**self.t))+1e-8)
            loss.append(float(np.mean(np.where(np.abs(self.predict(x)-y)<=1,.5*(self.predict(x)-y)**2,np.abs(self.predict(x)-y)-.5))))
        return loss
