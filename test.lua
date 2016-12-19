
require 'optim'
require 'seboost_parallel_simulation'
require 'torch'
require 'image'
require 'nn'
require 'cunn'
require 'cudnn'

trainData = torch.randn(1000)
trainLabels = torch.randn(1000)

optimState = {}
sesopConfig = {
  --optMethod=optim.adadelta, 
  optMethod=optim.adagrad, 
  histSize=10,
  sesopData=trainData,
  sesopLabels=trainLabels,
  isCuda=true,
  optConfig={}, --state for the inner optimization function. We need one for each solver!!
  sesopUpdate=400, --sesop iteration every 400 'optMethod' iterations.
  sesopBatchSize=100,
  nodeIters=2
}

function feval(x, inputs, targets)
  return x*x, 2*x
end

x = torch.randn(1)
x[1] = 1
x = x:cuda()

for i=1,6 do
  optim.seboost(feval, x, sesopConfig, optimState)
end




