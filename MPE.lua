-- require 'cunn'
-- require 'cudnn'
-- require 'cutorch'
-- cutorch.setDevice(1)
-- require 'simul-funcs'
-- require 'optim'

-- function feval(w_,input_data,output_label)
--         		if w~=w_ then
--         			w:copy(w_)
--         		end
--         		local data = input_data or x
--         		label = output_label or yt
--         		y_out = model:forward(data)
--                 err = criterion:forward(y_out,label)       		
--         		model:zeroGradParameters()
--         		local dE_dy = criterion:backward(y_out,label)
--         		model:backward(data,dE_dy)
--         		return err,dE_dw
-- end

-- model,criterion = create_model()
-- w,dw = model:getParameters()

-- MPE_config = {
-- 	interval_iter = 100,
-- 	K = 5,
--     num_w = w:nElement(),
--     iter = 1,
--     count = 0,
--     optMethod = optim.sgd,
--     optConfig = {
--     learningRate = 1e-3
-- }
-- }


-- MPE Vector Extrapolation
function MPE_Acel(opfunc,x,config)
local isCuda = config.isCuda or false
local interval_iter = config.interval_iter or 100
local K = config.K or 5
config.Space_x = config.Space_x or torch.zeros(torch.LongStorage{config.num_w,K+2})
config.Iter_pre = config.Iter_pre or 0 -- precondition technique, i.e. the number of initial iterations
if isCuda then 
	config.Space_x:cuda()
end

config.iter = config.iter or 1
config.count = config.count or 0
--print('MPE is called\n' .. config.count)

config.iter = config.iter + 1
x,fx = config.optMethod(opfunc, x, config.optConfig)
if config.iter%interval_iter==0 and config.iter >= config.Iter_pre then
	config.count = config.count+1
  --print(config.count)
	config.Space_x[{{},{config.count}}]:copy(x)
end
if config.count%(K+2)~=0 or config.count==0 then
	return x,fx
end
print('Vector Extrapolation MPE is called\n')
local isCG = config.isCG
local space_u = config.Space_x:narrow(2,2,K+1) - config.Space_x:narrow(2,1,K+1)
local U = space_u:narrow(2,1,K)
local u_k = space_u[{{},{K+1}}]
local c

if isCG then
c = torch.ones(K)
local config_CG = {
	maxIter = 100--1.5*(K+2)
} -- it should be defined in config later
function feval_cg()
	local temp = torch.add(torch.mv(U,c),u_k)
	local f_CG = 0.5*torch.dot(temp,temp)
    local grad_CG = torch.mv(U:t(),temp)
    --print(grad_CG:norm())
return  f_CG,grad_CG
end
optim.cg(feval_cg,c,config_CG)
c = torch.cat(c,torch.ones(1))
else
c = torch.mm(torch.inverse(torch.mm(U:t(),U)),U:t())
c = -torch.mv(c,torch.squeeze(u_k))
c = torch.cat(c,torch.ones(1))
end

local gamma =torch.div(c,torch.sum(c))
--gamma = torch.div(torch.ones(K+1),K+1)
--print(torch.mv(space_u,gamma):norm())
-- combine the last point with the obtained point as the initial point -- like moment
config.lambda = config.lambda or 1
x:copy((1-config.lambda)*config.Space_x[{{},{K+2}}]+config.lambda*torch.mv(config.Space_x:narrow(2,1,K+1),gamma)) -- it is very important that how to combine the previous iteration

config.Space_x:zero()
config.count = 0

-- it may be a good idea to keep the previous points in the next iteration
--local point_drop = K -- the maximal value is K+1, i.e., we only keep the last point
--if point_drop > K+1 then
--  error("The point drop is too large\n")
--end
--config.count = config.count-point_drop
--config.Space_x[{{},{1,K+2-point_drop}}] = config.Space_x:narrow(2,point_drop+1,K+2-point_drop)--order keep the point, randomly choose may also improve

--config.Space_x[{{},{config.count}}]:copy(x)
return x,fx

end



-- RRE Vector Extrapolation
function RRE_Acel(opfunc,x,config)
local isCuda = config.isCuda or false
local interval_iter = config.interval_iter or 100
local K = config.K or 5
config.Space_x = config.Space_x or torch.zeros(torch.LongStorage{config.num_w,K+2})
config.Iter_pre = config.Iter_pre or 0
if isCuda then 
	config.Space_x:cuda()
end

config.iter = config.iter or 1
config.count = config.count or 0
--print('RRE is called\n' .. config.count)

config.iter = config.iter + 1

x,fx = config.optMethod(opfunc, x, config.optConfig)

if config.iter%interval_iter==0 and config.iter >= config.Iter_pre then
	config.count = config.count+1
	config.Space_x[{{},{config.count}}]:copy(x)
end
if config.count%(K+2)~=0 or config.count==0 then
	return x,fx
end

print('Vector Extrapolation RRE is called\n')

local space_u = config.Space_x:narrow(2,2,K+1) - config.Space_x:narrow(2,1,K+1)
local temp_u = torch.inverse(torch.mm(space_u:t(),space_u))
local vector_one = torch.ones(K+1)
local gamma =torch.div(torch.mv(temp_u,vector_one),torch.sum(temp_u))
--print('The sum of gamma is: ' .. torch.sum(gamma))
config.lambda = config.lambda or 1
x:copy((1-config.lambda)*config.Space_x[{{},{K+2}}]+config.lambda*torch.mv(config.Space_x:narrow(2,1,K+1),gamma)) -- it is very important that how to combine the previous iteration
config.count = 0
config.Space_x:zero()
-- it may be a good idea to keep the previous points in the next iteration
--local point_drop = 2 -- the maximal value is K+1, i.e., we only keep the last point
--if point_drop > K+1 then
--  error("The point drop is too large\n")
--end
--config.count = config.count-point_drop
--config.Space_x[{{},{1,K+2-point_drop}}]=config.Space_x:narrow(2,point_drop+1,K+2-point_drop)
--config.Space_x[{{},{config.count}}]:copy(x)
return x,fx
end

-- for i=1,700 do
-- 	w:copy(torch.randn(MPE_config.num_w))
-- 	MEP_Acel(feval,w,MPE_config)
-- end
