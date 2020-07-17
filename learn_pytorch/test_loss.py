import torch
import numpy as np
from torch import optim
# reduction = 'sum'  --> reduce=True,size_average=False
# reduction = 'mean' --> reduce=True,size_average=True
# reduction = ''     --> reduce=False

#loss_fn = torch.nn.MSELoss(reduce=False, size_average=False)
#loss_fn = torch.nn.MSELoss(reduce=False, size_average=True) 
#loss_fn = torch.nn.MSELoss(reduce=True, size_average=False) 
#loss_fn = torch.nn.MSELoss(reduce=True, size_average=True) 
loss_fn = torch.nn.MSELoss(reduction='sum')
a=np.array([[1,2],[3,8]],dtype='float32')
b=np.array([[2,4],[6,4]],dtype='float32')
input = torch.autograd.Variable(torch.from_numpy(a),requires_grad=True)
print("inpute shape is ,",input.shape)
target = torch.autograd.Variable(torch.from_numpy(b))
weight = torch.Tensor([[1,1],[0,0]])
output = torch.sigmoid(input * weight)
loss = loss_fn(output, target)
#print("input requires_grad ? ",input.requires_grad)
#print("weight is leaf?",weight.is_leaf)
#print("output is leaf?",output.is_leaf)
#print("output grad_fn ",output.grad_fn)
#print("loss = \n",loss,"\n loss is leaf? ",loss.is_leaf,"\nloss requires_grad ? ",loss.requires_grad)

#output.backward()
#print("loss grad: %g, output grad: %g, input grad: %g"%(1.0,1.0,input.grad))
input_grad = 2*(output-target)*weight*output*(torch.full_like(output,1.0)-output)
disturb = input * 0.3
disturb = disturb.sum().detach()
total_loss = disturb + loss #a detach tensor input_grad don't disturb total_loss
#print(input_grad.is_leaf)
#print("loss = %g, disturb = %g, total_loss = %g"%(loss,disturb,total_loss))
#print("compute grad: \n",input_grad,"\n True grad: \n",input.grad)



optimizer = optim.SGD(params=[{'params':input}], lr=0.01)
def mytrain(loss = total_loss,opt = optimizer):
    opt.zero_grad()
    loss.backward()
    opt.step()
    print("loss = %g, disturb = %g, total_loss = %g"%(loss,disturb,total_loss))
    print("compute grad: \n",input_grad,"\n True grad: \n",input.grad)


if __name__ == "__main__":
    print("################ training ################")
    print("input, \n",input,"\ngrad, \n",input.grad)
    mytrain(loss = total_loss,opt = optimizer)
    print("input after training: \n",input)