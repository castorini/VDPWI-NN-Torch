--[[ Author: Hua
--]]

local NTM, parent = torch.class('ntm.NTM', 'nn.Module')

function NTM:__init(config)
  self.input_dim   = config.input_dim   or error('config.input_dim must be specified')
  self.mem_cols    = config.mem_cols    or 300
  self.cont_dim    = config.cont_dim    or 200
  self.cont_layers = config.cont_layers or 1
  self.shift_range = config.shift_range or 1
  self.write_heads = config.write_heads or 1
  self.read_heads  = config.read_heads  or 1
  self.task        = config.task        or 'seme' --'twitter'  
  self.sim_nhidden = config.sim_nhidden or 150
  self.reg         = config.reg         or 1e-5

  self.read_heads = 1
  self.yo_dim = 13
  self.batch_size = 1
  self.depth = 0
  self.cells = {}
  self.sparseTrain = 0
  self.stopWord = 0
  
  self.master_cell = self:new_cell()

  -- word embedding
  self.emb_vecs = config.emb_vecs
  self.emb_dim = config.emb_vecs:size(2)
  self.in_dim = self.emb_dim
  self.mem_dim = self.cont_dim

  -- number of similarity rating classes
  if self.task == 'sic' then
    self.num_classes = 5
  elseif self.task == 'vid' or self.task == 'seme' or self.task ==  'smteur' or self.task == 'msrpar'
          or self.task == 'sts2013' or self.task == 'sts2014' or self.task == 'sts2012' then
    self.num_classes = 6
  elseif self.task == 'twitter' or self.task == 'mspr' or self.task == 'qa'  or self.task == 'wikiqa' then
    self.num_classes = 2
    local claa = {}
    for iii = 1, 2 do
      claa[iii]=iii
    end
    self.confusion = optim.ConfusionMatrix(claa)
  else
    error("not possible task!")
  end

  self.learningRate =1e-4
  self.limit = 32
  if self.task == 'mspr' or self.task == 'msrpar' or self.task == 'twitter'
          or self.task == 'sts2013' or self.task == 'sts2014' then
    self.limit = 48
  elseif self.task == 'sic' then
    self.limit = 32
  elseif self.task == 'qa' then
    self.limit = 64
  elseif self.task == 'wikiqa' then
    self.limit = 48
  else
    error("No such task!")
  end
  self.outputAllSimi = torch.zeros(self.limit, self.limit, self.yo_dim)
	
  -- Objective
  if self.task == 'mspr' or self.task == 'twitter' or self.task == 'qa' or self.task == 'wikiqa' then
    self.criterion = nn.MultiMarginCriterion()
  else
    self.criterion = nn.DistKLDivCriterion()
  end

  --self.criterion = nn.ClassNLLCriterion()
  print('Task is: ' .. self.task .. ' | class:' .. self.num_classes)
  print(self.criterion)

  self.softMaxC = self:VggConv()
  --print(self.softMaxC)

  self.yesc = false
  if self.yesc then
    self.softMaxC = self.softMaxC:cuda()
    self.criterion = self.criterion:cuda()
  end

  -------------------------------------------------------
  -------------------------------------------------------

  local lstm_config = {
    in_dim = self.emb_dim,
    mem_dim = self.cont_dim,
    num_layers = 1,
    gate_output = true,
  }
  --print('LSTM config:')
  --print(lstm_config)
  
  ---BiLSTM---
  self.llstm = ntm.LSTM(lstm_config) -- "left" LSTM
  self.rlstm = ntm.LSTM(lstm_config) -- "left" LSTM
  self.llstm_b = ntm.LSTM(lstm_config) -- backward "left" LSTM
  self.rlstm_b = ntm.LSTM(lstm_config) -- backward "right" LSTM
  
  ----------------------------------------
  local modules = nn.Parallel()
    :add(self.master_cell)
    :add(self.llstm)
    :add(self.softMaxC) 
  
  self.params, self.grad_params = modules:getParameters()

  self.rlstm:share(self.llstm, 'weight', 'bias', 'gradWeight', 'gradBias')
  self.rlstm_b:share(self.llstm, 'weight', 'bias', 'gradWeight', 'gradBias')
  self.llstm_b:share(self.llstm, 'weight', 'bias', 'gradWeight', 'gradBias')
end

function NTM:VggConv()
  include('./models/AlignMaxDropLeakyMulti.lua')
  dofile('./models/very_deep.lua')
  local convM = createModel(self.yo_dim, self.num_classes, self.limit)
  return convM
end

function NTM:ClassifierOOne()
  local maxMinMean = 3
  local separator = (maxMinMean+1)*self.cont_dim
  
  modelQ1 = nn.Sequential()
  local paraQuery=nn.ParallelTable()
  paraQuery:add(nn.Identity())
  paraQuery:add(nn.Identity())	
  modelQ1:add(paraQuery)
  modelQ1:add(nn.JoinTable(1))

  modelQ1:add(nn.Linear(3*self.yo_dim+ separator + (maxMinMean+1)*2, self.sim_nhidden))
  --modelQ1:add(nn.Linear(3*self.cont_dim+ separator + (maxMinMean+1)*2, self.sim_nhidden))
  modelQ1:add(nn.Tanh())	
  modelQ1:add(nn.Linear(self.sim_nhidden, self.num_classes))
  modelQ1:add(nn.LogSoftMax())	
  return modelQ1
end

function NTM:new_cell()
  -- previous memory state and read/write weights
  local M_p = nn.Identity()()
  local M_p2 = nn.Identity()()

  local ConstantIn = nn.Identity()()
 
  -- LSTM controller output
  local mtable = nn.Identity()()
  local mtable2 = nn.Identity()()
  
  -- output and hidden states of the controller module
  --local mtable, ctable = self:new_controller_module(input, mtable_p, ctable_p)
  local m = (self.cont_layers == 1) and mtable 
    or nn.SelectTable(self.cont_layers)(mtable)
  local m2 = (self.cont_layers == 1) and mtable2 
    or nn.SelectTable(self.cont_layers)(mtable2)

  local r = self:new_mem_module(M_p, m, M_p2, m2, ConstantIn) 

  local inputs = {mtable, M_p, mtable2, M_p2, ConstantIn}
  
  local outputs = nn.Identity()(r)

  local cell = nn.gModule(inputs, {outputs})
  if self.master_cell ~= nil then
    share_params(cell, self.master_cell, 'weight', 'bias', 'gradWeight', 'gradBias')
  end
  return cell
end

function NTM:new_mem_module(M_p, m, M_p2, m2, ConstantIn) -- note nere
  -- read heads
  local wr, r
  if self.read_heads == 1 then
    r = self:new_head(M_p, m, M_p2, m2, ConstantIn, true)
  else
    local r1 = {}
    for i = 1, self.read_heads do
      r1[i] = self:new_read_head(nn.SelectTable(i)(M_p), m)
    end     
    r = nn.Identity()(nn.JoinTable(1)(r1))
  end
  return r
end

-- Create a new head
function NTM:new_head(M_p, m, M_p2, m2, ConstantIn, is_read) 
  ------------------Forward
  local sim1 = nn.CsDis(){M_p, m}
  --local sim2 = nn.Abs()(nn.CSubTable(){M_p2, k2})
  local sim3 = nn.MulConstant(-1)(nn.PairwiseDistance(2){M_p, m})
  local sim4 = nn.DotProduct(){M_p, m}
  --local sim5 = nn.CMulTable(){M_p2, k2}

  ------------------Backward
  local sim1_r = nn.CsDis(){M_p2, m2}
  local sim3_r = nn.MulConstant(-1)(nn.PairwiseDistance(2){M_p2, m2})
  local sim4_r = nn.DotProduct(){M_p2, m2}

  ------------------Forward and Backward Both
  local M_pAll = nn.CAddTable(){M_p, M_p2}
  local mAll = nn.CAddTable(){m, m2}

  local sim1_a = nn.CsDis(){M_pAll, mAll}
  local sim3_a = nn.MulConstant(-1)(nn.PairwiseDistance(2){M_pAll, mAll})
  local sim4_a = nn.DotProduct(){M_pAll, mAll}

  local M_pAll2 = nn.JoinTable(1){M_p, M_p2}
  local mAll2 = nn.JoinTable(1){m, m2}

  local sim1_a2 = nn.CsDis(){M_pAll2, mAll2}
  local sim3_a2 = nn.MulConstant(-1)(nn.PairwiseDistance(2){M_pAll2, mAll2})
  local sim4_a2 = nn.DotProduct(){M_pAll2, mAll2}

  local simi = nn.JoinTable(1){sim1, sim3, sim4, sim1_r, sim3_r, sim4_r, sim1_a, sim3_a, sim4_a, sim1_a2, sim3_a2, sim4_a2, ConstantIn}
  return simi
end

function NTM:forward2(rdoc, lquery, rdoc_b, lquery_b, rsize, lsize, reverse)
  self.rdoc_size = rdoc:size(1) -- docs
  self.lquery_size = lquery:size(1) -- query
  local constantOne = torch.Tensor(1):fill(1)
  self.depth = 0

  self.outputAllSimi:zero()

  --print("l:" .. self.lquery_size)
  --print("r:" .. self.rdoc_size)
  if self.lquery_size > self.limit then
    self.lquery_size = self.limit
    --print("l out")
  end
  if self.rdoc_size > self.limit then
    self.rdoc_size = self.limit
    --print("r out")
  end

  for tq = 1, self.lquery_size do
    local linput = lquery[tq]
    local linput_b = lquery_b[tq] --reverse

    for td = 1, self.rdoc_size do
      local rinput = rdoc[td]
      local rinput_b = rdoc_b[td] --reverse

      self.depth = self.depth + 1
      local cell = self.cells[self.depth]
      if cell == nil then
        cell = self:new_cell()
        self.cells[self.depth] = cell
      end

      local prev_outputs
      local inputs = {linput, rinput, linput_b, rinput_b, constantOne}
      self.output = cell:forward(inputs)
      self.outputAllSimi[tq][td] = self.output:clone()
    end
  end
  return self.outputAllSimi:permute(3,1,2)
end

function NTM:backward3(rdoc, lquery, rdoc_b, lquery_b, grad_outputs_in, reverse)
  self.rdoc_size = rdoc:size(1) -- docs
  self.lquery_size = lquery:size(1) -- query

  local constantOne = torch.Tensor(1):fill(1)

  local feasible = grad_outputs_in:permute(2,3,1)
  local rgrad = torch.zeros(self.rdoc_size, self.cont_dim)
  local lgrad = torch.zeros(self.lquery_size, self.cont_dim)

  local rgrad_b = torch.zeros(self.rdoc_size, self.cont_dim)
  local lgrad_b = torch.zeros(self.lquery_size, self.cont_dim)

  if self.lquery_size > self.limit then
    self.lquery_size = self.limit
  end
  if self.rdoc_size > self.limit then
    self.rdoc_size = self.limit
  end

  if self.depth == 0 or self.depth ~= self.rdoc_size*self.lquery_size then
    error("No cells to backpropagate through or memory words are wrong")
  end
  
  for tq = self.lquery_size, 1, -1 do
    local linput = lquery[tq]
    local linput_b = lquery_b[tq]

    for td = self.rdoc_size, 1, -1 do
      local rinput = rdoc[td]
      local rinput_b = rdoc_b[td]

      local grad_output = feasible[tq][td]
      local cell = self.cells[self.depth]
      if not cell or self.depth ~= (tq-1)*self.rdoc_size + td then
        print(self.depth .. " check:" .. ((tq-1)*self.rdoc_size + td)  ..  " td:" .. td .. "  tq:" .. tq .. " rsize:" .. self.rdoc_size .. " lsize:" .. self.lquery_size)
        error("not possible!")
      end
      -- get inputs
      local inputs = {linput, rinput, linput_b, rinput_b, constantOne} 
      ---
      self.gradInput = cell:backward(inputs, grad_output)
      self.depth = self.depth - 1
      lgrad[tq]:add(self.gradInput[1])
      rgrad[td]:add(self.gradInput[2])
      lgrad_b[tq]:add(self.gradInput[3])
      rgrad_b[td]:add(self.gradInput[4])
    end
  end
  self:forget() 
  return rgrad, lgrad, rgrad_b, lgrad_b
end

function NTM:trainCombineSeme(dataset)
  self.optim_state = {
    learningRate = self.learningRate,
    momentum = 0.9,
    decay = 0.95
  }
  self.softMaxC:training()
  self.llstm:training()
  self.rlstm:training()
  self.rlstm_b:training()
  self.llstm_b:training()
  self.master_cell:training()

  local train_looss = 0.0   
  local indices = torch.randperm(dataset.size)
  self.lmaxsize = dataset.lmaxsize
  self.rmaxsize = dataset.lmaxsize

  for i = 1, dataset.size, self.batch_size do
    --if i > 100 then
    --  break
    --end
    if self.limit > 49 and i % 50 == 1 then 
      --print(i) 
      collectgarbage() 
    end

    local batch_size = 1 --math.min(i + self.batch_size - 1, dataset.size) - i + 1    
    local targets = torch.zeros(batch_size, self.num_classes)
    local sim  = -0.1
    for j = 1, batch_size do
      if self.task == 'sic' or self.task == 'vid' or self.task == 'seme' or self.task == 'msrpar' 
        or self.task == 'smteur' or self.task == 'sts2013' or self.task == 'sts2014' or self.task == 'sts2012' then
        sim = dataset.labels[indices[i + j - 1]] * (self.num_classes - 1) + 1
      elseif self.task == 'twitter' or self.task == 'mspr' or self.task == 'qa' or self.task == 'wikiqa' then
        sim = dataset.labels[indices[i + j - 1]] 	
        --print("Sim from dataset")
        --print(sim)
      else
	     error("not possible!")
      end
      local ceil, floor = math.ceil(sim), math.floor(sim)
      if ceil == floor then
        targets[{j, floor}] = 1
      else
        targets[{j, floor}] = ceil - sim
        targets[{j, ceil}] = sim - floor
      end
    end

    local feval = function(x)
      self.grad_params:zero()
      local loss = 0
      for j = 1, batch_size do
        local idx = indices[i + j - 1]
        --local sim = dataset.labels[idx] + 1 -- read class label
        local lsent, rsent = dataset.lsents[idx], dataset.rsents[idx]
        local linputs = self.emb_vecs:index(1, lsent:long()):double() -- query --change
        local rinputs = self.emb_vecs:index(1, rsent:long()):double() -- doc --change
	---Normal
        local hiddenR = self.rlstm:forwardMultiAll(rinputs)[1] -- doc
        local Memory = self.llstm:forwardMultiAll(linputs)[1] -- memory on query
        ---Reverse
        local hiddenR_b = self.rlstm_b:forwardMultiAll(rinputs, true)[1] -- true => reverse
        local Memory_b = self.llstm_b:forwardMultiAll(linputs, true)[1]

	local part2 = self:forward2(hiddenR, Memory, hiddenR_b, Memory_b, self.rmaxsize, self.lmaxsize)
        --print(part2:size())	
	local output = self.softMaxC:forward(part2)
        
        if self.sparseTrain == 1 and (self.task == 'sic' or self.task == 'vid' or self.task == 'sts2013') then
          loss = self.criterion:forward(output, dataset.labelsSparse[idx])
        elseif self.task == 'mspr' or self.task == 'twitter' or self.task == 'qa' or self.task == 'wikiqa' then
          --print(sim)
          loss = self.criterion:forward(output, sim)		
	else
          loss = self.criterion:forward(output, targets[1])
	end

        train_looss = loss + train_looss
        local sim_grad = nil
        if self.sparseTrain == 1 and (self.task == 'sic' or self.task == 'vid' or self.task == 'sts2013') then
          sim_grad = self.criterion:backward(output, dataset.labelsSparse[idx])
          --print(dataset.labelsSparse[idx])
          --print(targets[1])
        elseif self.task == 'mspr' or self.task == 'twitter' or self.task == 'qa' or self.task == 'wikiqa' then
          sim_grad = self.criterion:backward(output, sim)
        else
          sim_grad = self.criterion:backward(output, targets[1])
        end
        local gErrorFromClassifier = self.softMaxC:backward(part2, sim_grad) -- self.yo_dim by 32 by 32
        local rgrad, lgrad, rgrad_b, lgrad_b = self:backward3(hiddenR, Memory, hiddenR_b, Memory_b, gErrorFromClassifier)
	self.llstm_b:backward(linputs, lgrad_b, true)
        self.rlstm_b:backward(rinputs, rgrad_b, true)
	self.llstm:backward(linputs, lgrad)        
	self.rlstm:backward(rinputs, rgrad)        
      end     
      local norm_dw = self.grad_params:norm()
      if norm_dw > 50 then
	      local shrink_factor = 50 / norm_dw
        self.grad_params:mul(shrink_factor)
      end
      return loss, self.grad_params
    end
    optim.rmsprop(feval, self.params, self.optim_state)    
  end
  print('Train Loss: ' .. train_looss)
end

function NTM:LSTM_backwardMulti(linputs, rep_grad)
  local lgrad
  --print('inside!')
  if self.cont_layers == 1 then
    lgrad = torch.zeros(linputs:size(1), self.cont_dim)
    lgrad[linputs:size(1)] = rep_grad:clone()
  else
    error("not possible")
  end
  self.llstm:backward(linputs, lgrad)  
end

-- Predict the similarity of a sentence pair.
function NTM:predictCombination(lsent, rsent, labelTest)
  local linputs = self.emb_vecs:index(1, lsent:long()):double() -- query
  local rinputs = self.emb_vecs:index(1, rsent:long()):double() -- doc

  --Normal
  local hiddenR = self.rlstm:forwardMultiAll(rinputs)[1] -- doc
  local Memory = self.llstm:forwardMultiAll(linputs)[1] -- memory on query
  ---Reverse
  local hiddenR_b = self.rlstm_b:forwardMultiAll(rinputs, true)[1] -- true => reverse
  local Memory_b = self.llstm_b:forwardMultiAll(linputs, true)[1]

  local part2 = self:forward2(hiddenR, Memory, hiddenR_b, Memory_b, self.rmaxsize, self.lmaxsize)
  --print(part2:size())	
  local output = self.softMaxC:forward(part2)

  local val = -1.0
  if self.task == 'sic' then
    val = torch.range(1, 5, 1):dot(output:exp())
  elseif self.task == 'vid' or self.task == 'smteur' or self.task == 'msrpar' 
        or self.task == 'sts2013' or self.task == 'sts2014' or self.task == 'sts2012' then
    val = torch.range(0, 5, 1):dot(output:exp())
  elseif self.task == 'seme' then
    val = torch.range(0, 1, 0.2):dot(output:exp())
  elseif self.task == 'twitter' or  self.task == 'mspr' or self.task == 'qa' or self.task == 'wikiqa' then
    self.confusion:add(output, labelTest)
    val = output:exp()[2]
  else
    error("not possible task")
  end
  return val
end

-- Produce similarity predictions for each sentence pair in the dataset.
function NTM:predict_dataset(dataset)
  self.lmaxsize = dataset.lmaxsize
  self.rmaxsize = dataset.lmaxsize
  self.softMaxC:evaluate()
  self.llstm:evaluate()
  self.rlstm:evaluate()
  self.master_cell:evaluate()
  self.rlstm_b:evaluate()
  self.llstm_b:evaluate()

  if self.task == 'mspr' or self.task == 'twitter' or self.task == 'qa' or self.task ==  'wikiqa' then
    self.confusion:zero()
  end

  local predictions = torch.Tensor(dataset.size)
  for i = 1, dataset.size do
    if self.limit > 48 and i % 50 == 1 then 
      --print(i) 
      collectgarbage() 
    end
    local lsent, rsent = dataset.lsents[i], dataset.rsents[i]
    predictions[i] = self:predictCombination(lsent, rsent, dataset.labels[i])
    if false and dataset.labelsReal[i] >= 3.5 then
      print("========================================================")   
      print("Left Sen:")
      print(dataset.lraw[i])
      print("Right Sen:")   
      print(dataset.rraw[i])
      print("P-Score: " .. predictions[i] .. " | True: " .. dataset.labelsReal[i]) 
    end
  end

  if self.task == 'mspr' or self.task == 'twitter' or self.task == 'qa' or self.task == 'wikiqa' then
   self.confusion:updateValids()
   local gcorrect = self.confusion.totalValid * 100
   print(self.confusion)
   print("TP: " .. self.confusion.mat[2][2] .. " " .. self.confusion.mat[2][1] .. " " .. self.confusion.mat[1][2])
   local F1=2*self.confusion.mat[2][2]/(2*self.confusion.mat[2][2]+self.confusion.mat[2][1]+self.confusion.mat[1][2])
   print("F1 score: " .. F1)
  end
  --self.rlstm:forget()
  --self:forget()
  return predictions
end

function NTM:parameters()
  local p, g = self.master_cell:parameters()
  local pi, gi = self.init_module:parameters()
  tablex.insertvalues(p, pi)
  tablex.insertvalues(g, gi)
  return p, g
end

function NTM:forget()
  self.depth = 0
  self.number_words = 0
  --self.init_module:backward(torch.Tensor{0}, self.gradInput)
  for i = 1, #self.gradInput do
    local gradInput = self.gradInput[i]
    if type(gradInput) == 'table' then
      for _, t in pairs(gradInput) do t:zero() end
    else
      self.gradInput[i]:zero()
    end
  end
end

function NTM:zeroGradParameters()
  self.master_cell:zeroGradParameters()
  self.master_cell_lstm:zeroGradParameters()
  --self.grad_params:zero()
  --self.init_module:zeroGradParameters()
end

function NTM:print_config()
  local num_params = self.params:nElement()
  print('This is NTM.bilstm.entail.share version!')
  print('num params: ' .. num_params)
  print('word vector dim: ' .. self.emb_dim)
  print('LSTM memory dim: ' .. self.cont_dim)
  print('regularization strength: ' .. self.reg)
  print('BiLSTM model with yodim: ' .. self.yo_dim)
  print('ConvNet size limit:' .. self.limit)
  print('Learning rate: ' .. self.learningRate)
  print('Sparsier target train: ' .. self.sparseTrain)
  print('Stop Word: ' .. self.stopWord)  
  print('sim module hidden dim: ' .. self.sim_nhidden)
end

