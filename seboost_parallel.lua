
local function copy2(obj)
  if type(obj) ~= 'table' then return obj end
  local res = setmetatable({}, getmetatable(obj))
  for k, v in pairs(obj) do res[copy2(k)] = copy2(v) end
  return res
end


local unistd = require "posix.unistd"
local p = require "posix"

--[[
- config.save : path for a directory to save model.nodeId.txt. Every 100 iterations, we save model.nodeId.txt.
                Then we wait for all other nodes to do the same and touch nodeId.ready.
                When all nodes are ready, node 0 read the models and perform sesop.
                Then it writes the result into sesop.out.
                Rest of nodes poll on sesop.out, when it is ready, they continue. 
                Node 0 remove *.ready and continue.
]]


local function lock_file(filename) 
  local fd = p.creat(filename, "rw-r--r--")
    -- Set lock on file
  local lock = {
      l_type = p.F_WRLCK;     -- Exclusive lock
      l_whence = p.SEEK_SET;  -- Relative to beginning of file
      l_start = 0;            -- Start from 1st byte
      l_len = 0;              -- Lock whole file
  }
  
  local result = -1
  while true do
    local result = p.fcntl(fd, p.F_SETLK, lock)
    if result ~= -1 then
      break
    end
    
    print("file locked by another process, try again... filename = "..filename)
    print('result = '..result)
  end
  return fd
end

local function unlock_file(fd) 
    -- Set lock on file
  local lock = {
      l_type = p.F_WRLCK;     -- Exclusive lock
      l_whence = p.SEEK_SET;  -- Relative to beginning of file
      l_start = 0;            -- Start from 1st byte
      l_len = 0;              -- Lock whole file
  }
  
  -- Release the lock
  lock.l_type = p.F_UNLCK
  p.fcntl(fd, p.F_SETLK, lock)
end


--lock is actually the fd.
--Every process must have its model file locked untill broadcast_model is called.
--broadcast_model then release the lock
--only then the other node will be able to complete poll_model.
local function broadcast_model(x, nodeId, save, lock)
  --assert that we have the lock!
  torch.save(save..'/model.'..nodeId..'.txt', x)
  unlock_file(lock)
end

--block untill model from nodeId is ready
local function poll_model(x, nodeId, save)
  local lock = lock_file(save..'/model.'..nodeId..'.txt')
  local res = torch.load(save..'/model.'..nodeId..'.txt', x)
  unlock_file(lock)
  return res
end

--assume model and done locks are taken for self.
local function sync(x, config, lock, lockDone)
  print (config.nodeId..' Sync')
  if config.nodeId == 0 then --MASTER
    local remote_models = {}
    
    print (config.nodeId..' Waiting for worker nodes to finish their iterations')
    --poll on all other nodes
    for i = 1, config.numNodes do
      remote_models[i] = poll_model(x, i, config.save)
      print (config.nodeId..' Recived model from node '..i)
    end
    
    print (config.nodeId..' Worker nodes finished, starting merging')
    x = merge_models(x, config, remote_models)
    print (config.nodeId..' Done merging, sending updated model')
    broadcast_model(x, 0, config.save, lock)
    print (config.nodeId..' Updated model sent, waiting for workers to consume it')
    --now we need to wait untill all models 
    --consume this updated model before we continue
    for i = 1, config.numNodes do
      --block untill node i is done.
      unlock_file(lock_file(config.save..'/done.'..i..'.txt'))
    end
    
  else --WORKERS
    print (config.nodeId..' Worker node sending model')
    broadcast_model(x, config.nodeId, config.save, lock)
    x = poll_model(x, 0, config.save)
    print (config.nodeId.. ' Worker node got merged model back!')
  end
  
  print (config.nodeId..' Done, releasing done lock')
  unlock_file(lockDone) --set that we are done
end

local function merge_models(x, config, remote_models)
  print (config.nodeId.. ' Merging models...')
  return x
end

local function start_divergence(x, config)
  local lock = lock_file(config.save..'/model.'..config.nodeId..'.txt')
  --as long as we didnt release lockDone, other nodes will not think we are done.
  --in other words, this is "set_not_done".
  local lockDone = lock_file(config.save..'/done.'..config.nodeId..'.txt')
  
  --Do 100 iterations
  for i = 1, config.numNodes do
    print (config.nodeId.. ' Is running iteration '..i)
  end
  
  
  sync(x, config, lock, lockDone)
end

require 'xlua'

opt = lapp[[
   -n,--num_of_nodes          (default 1)             Number of nodes to simulate.
   -i,--id          (default 1)             Number of nodes to simulate.
]]


  
local config = {
  save='./tmp',
  nodeId=opt.id,
  numNodes=opt.num_of_nodes
}
local x = 1

print (config)

start_divergence(x, config)
