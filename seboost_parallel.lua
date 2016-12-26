
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
  
  function Worker:recv_from_master(x)
    self.client:recv(x)
  end
  
  function Worker:close()
    self.client:close()
  end
  
end

require 'xlua'

opt = lapp[[
   -n,--num_of_nodes          (default 1)             Number of nodes to simulate.
   -i,--id          (default 1)             Number of nodes to simulate.
]]


  

if (opt.id == 0) then
  local x = torch.randn(3,3):float():cuda()
  
  print('master x = ')
  print(x)
  
  master = Master(x, opt.num_of_nodes - 1)
  
  master:block_on_workers()
  
  print('master remote_models after blocking = ')
  print(master.remote_models[0])
  print(master.remote_models[1])
  
  master:broadcast_to_workers(x)
  
  master:close()
  
end

if (opt.id > 0) then
  local x = torch.randn(3,3):float():cuda()
  
  print('worker x = ')
  print(x)
  
  worker = Worker(opt.id)
  
  worker:send_to_master(x)
  
  worker:recv_from_master(x)
  print('worker x after broadcast = ')
  print(x)
  
  worker:close()
end

