
local function copy2(obj)
  if type(obj) ~= 'table' then return obj end
  local res = setmetatable({}, getmetatable(obj))
  for k, v in pairs(obj) do res[copy2(k)] = copy2(v) end
  return res
end



local ipc = require 'libipc'
local sys = require 'sys'
require 'cunn'

do --define server
  local Master = torch.class('Master')

  --n is the number of workers.
  function Master:__init(x, n)
    self.server = ipc.server('127.0.0.1', 8080)
    self.remote_models = {}
    self.next_free = 0
    self.n = n
    
    for i = 0, n - 1 do
      self.remote_models[i] = x:clone()
    end
    
  end
  
  --wait to get results from n workers.
  function Master:block_on_workers()
    self.next_free = 0
    self.server:clients(self.n, function(client)
      --will this run on paralel?? BUG!!!
      local msg = client:recv(self.remote_models[self.next_free])
      self.next_free = self.next_free + 1
    end)
  
  end
  
  function Master:broadcast_to_workers(x)
    self.server:clients(self.n, function(client)
      client:send(x)
    end)
  end
  
  function Master:close()
    self.server:close()
  end
  
  
  
  
  
  --define worker
  local Worker = torch.class('Worker')
  function Worker:__init(id)
    self.client = ipc.client('127.0.0.1', 8080)
    self.id = id
  end
  
  function Worker:send_to_master(x)
    self.client:send(x)
  end
  
  --blocks
  function Worker:recv_from_master(x)
    return self.client:recv(x)
  end
  
  function Worker:close()
    self.client:close()
  end
  
end


function optim.seboost(opfunc, x, config, state)

  -- get/update state
  local state = state or config
  local isCuda = config.isCuda or false
  local sesopData = config.sesopData
  local sesopLabels = config.sesopLabels
  local sesopBatchSize = config.sesopBatchSize or 100
  config.nodeIters = config.nodeIters or 100
  
  state.itr = state.itr or 0
	config.numNodes = config.numNodes or 2  
  state.sesopIteration = state.sesopIteration or 0
  state.itr = state.itr + 1
  
  if (state.itr % config.nodeIters ~= 0) then
    x,fx = config.optMethod(opfunc, x, config.optConfig)
    return x,fx
  end
    
  if (config.master == nil) then
    --WORKER--
    --print ('WORKER SESOP begin')
    config.worker:send_to_master(x)
    config.worker:recv_from_master(x)
    local fHist = {}
    fHist = config.worker:recv_from_master(fHist)
    return x, fHist
  end

  if (config.worker == nil) then
    --MASTER--
    --print ('MASTER SESOP begin')
    state.splitPoint = state.splitPoint or x:clone() --the first split point is the first point

    config.master:block_on_workers()
    --Do SESOP on master.remote_models:
    
    if (state.dirs == nil) then
      --if it is the first time
      state.dirs = torch.zeros(x:size(1), config.numNodes)
      state.aOpt = torch.zeros(config.numNodes)
      --state.aOpt[1] = 1 --we start from taking the first node direction (maybe start from avrage?).
        
      if (isCuda) then
        state.dirs = state.dirs:cuda()
        state.aOpt = state.aOpt:cuda()
      end
    end
    
    state.aOpt:copy(torch.ones(config.numNodes)*(1/config.numNodes)) --avrage
    state.dirs[{ {}, 1 }]:copy(x - state.splitPoint)
    --SV, build directions matrix
    for i = 1, config.numNodes - 1 do   
      --[{ {}, i }] means: all of the first dim, slice in the second dim at i = get i col.
      state.dirs[{ {}, i + 1 }]:copy(config.master.remote_models[i - 1] - state.splitPoint)
    end


    --now optimize!
    local xInit = state.splitPoint
      -- create mini batch
    local subT = (state.sesopIteration) * sesopBatchSize + 1
    subT = subT % (sesopData:size(1) - sesopBatchSize) --Calculate the next batch index
    local sesopInputs = sesopData:narrow(1, subT, sesopBatchSize)
    local sesopTargets = sesopLabels:narrow(1, subT, sesopBatchSize)
    
    --print(sesopInputs:size())
    if isCuda then
    --  sesopInputs = sesopInputs:cuda()
    --  sesopTargets = sesopTargets:cuda()
    end

    -- Create inner opfunc for finding a*
    local feval = function(a)
      --A function of the coefficients
      local dirMat = state.dirs
      --Note that opfunc also gets the batch
      local afx, adfdx = opfunc(xInit + dirMat*a, sesopInputs, sesopTargets)
      return afx, (dirMat:t()*adfdx)
    end
    --x,f(x)
    config.maxIter = config.numNodes
    local _, fHist = optim.cg(feval, state.aOpt, config, state) --Apply optimization using inner function
       
    --updating model weights!
    x:copy(xInit)
    local sesopDir = state.dirs*state.aOpt 
    x:add(sesopDir)
      
    --the new split point is 'x'.
    --The next time this function is called will be with 'x'.
    --The next time we will change a node, it will get this 'x'.
    state.splitPoint:copy(x)
      
    state.sesopIteration = state.sesopIteration + 1
      
    config.master:broadcast_to_workers(x)
    config.master:broadcast_to_workers(fHist)
    return x,fHist
  end  
end
  

