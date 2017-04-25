--[[

 Long Short-Term Memory.

--]]
require 'nn'

local LSTM, parent = torch.class('ntm.LSTM', 'nn.Module')

function LSTM:__init(config)
  parent.__init(self)

  self.in_dim = config.in_dim
  self.mem_dim = config.mem_dim or 300
  self.num_layers = config.num_layers or 1
  self.gate_output = config.gate_output
  if self.gate_output == nil then self.gate_output = true end

  self.master_cell = self:new_cell()
  self.depth = 0
  self.cells = {}  -- table of cells in a roll-out
  
  -- initial (t = 0) states for forward propagation and initial error signals
  -- for backpropagation
  local ctable_init, ctable_grad, htable_init, htable_grad
  if self.num_layers == 1 then
    ctable_init = torch.zeros(self.mem_dim)
    htable_init = torch.zeros(self.mem_dim)
    ctable_grad = torch.zeros(self.mem_dim)
    htable_grad = torch.zeros(self.mem_dim)
  else
    ctable_init, ctable_grad, htable_init, htable_grad = {}, {}, {}, {}
    for i = 1, self.num_layers do
      ctable_init[i] = torch.zeros(self.mem_dim)
      htable_init[i] = torch.zeros(self.mem_dim)
      ctable_grad[i] = torch.zeros(self.mem_dim)
      htable_grad[i] = torch.zeros(self.mem_dim)
    end
  end
  self.initial_values = {ctable_init, htable_init}
  self.gradInput = {
    torch.zeros(self.in_dim),
    ctable_grad,
    htable_grad
  }
end

-- Instantiate a new LSTM cell.
-- Each cell shares the same parameters, but the activations of their constituent
-- layers differ.
function LSTM:new_cell()
  local input = nn.Identity()()
  local ctable_p = nn.Identity()()
  local htable_p = nn.Identity()()

  -- multilayer LSTM
  local htable, ctable = {}, {}
  for layer = 1, self.num_layers do
    local h_p = (self.num_layers == 1) and htable_p or nn.SelectTable(layer)(htable_p)
    local c_p = (self.num_layers == 1) and ctable_p or nn.SelectTable(layer)(ctable_p)

    ---and nn.Linear(self.in_dim, self.mem_dim)(nn.Dropout(0.2)(input))
    local new_gate = function()
      local in_module = (layer == 1)
        and nn.Linear(self.in_dim, self.mem_dim)(input)--replace dropout here.
        or  nn.Linear(self.mem_dim, self.mem_dim)(htable[layer - 1])
      return nn.CAddTable(){
        in_module,
        nn.Linear(self.mem_dim, self.mem_dim)(h_p)
      }
    end

    -- input, forget, and output gates
    local i = nn.Sigmoid()(new_gate())
    local f = nn.Sigmoid()(new_gate())
    local update = nn.Tanh()(new_gate())

    -- update the state of the LSTM cell
    ctable[layer] = nn.CAddTable(){
      nn.CMulTable(){f, c_p},
      nn.CMulTable(){i, update}
    }

    if self.gate_output then
      local o = nn.Sigmoid()(new_gate())
      htable[layer] = nn.CMulTable(){o, nn.Tanh()(ctable[layer])}
    else
      htable[layer] = nn.Tanh()(ctable[layer])
    end
  end

  -- if LSTM is single-layered, this makes htable/ctable Tensors (instead of tables).
  -- this avoids some quirks with nngraph involving tables of size 1.
  htable, ctable = nn.Identity()(htable), nn.Identity()(ctable)
  local cell = nn.gModule({input, ctable_p, htable_p}, {ctable, htable})

  -- share parameters
  if self.master_cell then
    share_params(cell, self.master_cell, 'weight', 'bias', 'gradWeight', 'gradBias')
  end
  return cell
end

-- Forward propagate.
-- inputs: T x in_dim tensor, where T is the number of time steps.
-- reverse: if true, read the input from right to left (useful for bidirectional LSTMs).
-- Returns the final hidden state of the LSTM.
function LSTM:forward(inputs, reverse)
  local size = inputs:size(1)
  for t = 1, size do
    local input = reverse and inputs[size - t + 1] or inputs[t]
    self.depth = self.depth + 1
    local cell = self.cells[self.depth]
    if cell == nil then
      cell = self:new_cell()
      self.cells[self.depth] = cell
    end
    local prev_output
    if self.depth > 1 then
      prev_output = self.cells[self.depth - 1].output
    else
      prev_output = self.initial_values
    end

    local outputs = cell:forward({input, prev_output[1], prev_output[2]})
    local ctable, htable = unpack(outputs)
    if self.num_layers == 1 then
      self.output = htable
    else
      self.output = {}
      for i = 1, self.num_layers do
        self.output[i] = htable[i]
      end
    end
  end
  return self.output
end

function LSTM:createMaxMinMeanModel()
	local modelMulti = nn.Sequential()
	local parallelConcat = nn.Concat(1)
	
	local maxSeq = nn.Sequential()
	maxSeq:add(nn.Max())
	--maxSeq:add(nn.Reshape(1, self.mem_dim))
	local minSeq = nn.Sequential()
	minSeq:add(nn.Min())
	--minSeq:add(nn.Reshape(1, self.mem_dim))
	local meanSeq = nn.Sequential()
	meanSeq:add(nn.Mean())
	--meanSeq:add(nn.Reshape(1, self.mem_dim))
	
	parallelConcat:add(maxSeq)	
	parallelConcat:add(minSeq)
	parallelConcat:add(meanSeq)
	modelMulti:add(parallelConcat)
	return modelMulti
end


function LSTM:forwardConnect(inputs, firstHidden, reverse)
  local size = inputs:size(1)
  --local allOutput = torch.zeros(self.num_layers, size, self.mem_dim)
  self.depth = 0

  for t = 1, size do
    local input = reverse and inputs[size - t + 1] or inputs[t]
    self.depth = self.depth + 1
    local cell = self.cells[self.depth]
    if cell == nil then
      cell = self:new_cell()
      self.cells[self.depth] = cell
    end
    local prev_output
    if self.depth > 1 then
      prev_output = self.cells[self.depth - 1].output
    else
      prev_output = self.initial_values
      prev_output[2]:add(firstHidden)
    end
	
    local outputs = cell:forward({input, prev_output[1], prev_output[2]})
    local ctable, htable = unpack(outputs)
    if self.num_layers == 1 then
      self.output = htable
      --allOutput[1][t] = htable
    else      
      self.output = {}
      for i = 1, self.num_layers do
        self.output[i] = htable[i]
      	allOutput[i][t] = htable[i]
      end
    end
  end  
  return self.output
  --return allOutput -- size*mem_dim
end

function LSTM:forwardMultiAll(inputs, reverse)
  local size = inputs:size(1)
  local allOutput = torch.Tensor(self.num_layers, size, self.mem_dim)
  self.depth = 0

  for t = 1, size do
    local input = reverse and inputs[size - t + 1] or inputs[t]
    self.depth = self.depth + 1
    local cell = self.cells[self.depth]
    if cell == nil then
      cell = self:new_cell()
      self.cells[self.depth] = cell
    end
    local prev_output
    if self.depth > 1 then
      prev_output = self.cells[self.depth - 1].output
    else
      prev_output = self.initial_values
    end

    local outputs = cell:forward({input, prev_output[1], prev_output[2]})
    local ctable, htable = unpack(outputs)
    if self.num_layers == 1 then
      self.output = htable
      if not reverse then
	allOutput[1][t] = htable
      else
	allOutput[1][size - t + 1] = htable
      end
    else
      error("not possible multi layer")
      self.output = {}
      for i = 1, self.num_layers do
        self.output[i] = htable[i]
	if not reverse then
	  allOutput[1][t] = htable[i]
        else
	  allOutput[1][size - t + 1] = htable[i]
        end      	
      end
    end
  end  

  return allOutput -- layer*size*mem_dim
end

function LSTM:forwardMulti(inputs, reverse)
  local size = inputs:size(1)
  local allOutput
  
  if size == 1 then 
  	allOutput = torch.zeros(self.num_layers, 1, self.mem_dim)
  else 
  	allOutput = torch.zeros(self.num_layers, size-1, self.mem_dim)
  end
  
  local finalOutput = torch.zeros(self.num_layers, self.mem_dim)
  
  for t = 1, size do
    local input = reverse and inputs[size - t + 1] or inputs[t]
    self.depth = self.depth + 1
    local cell = self.cells[self.depth]
    if cell == nil then
      cell = self:new_cell()
      self.cells[self.depth] = cell
    end
    local prev_output
    if self.depth > 1 then
      prev_output = self.cells[self.depth - 1].output
    else
      prev_output = self.initial_values
    end

    local outputs = cell:forward({input, prev_output[1], prev_output[2]})
    local ctable, htable = unpack(outputs)
    if self.num_layers == 1 then
      self.output = htable
      if t == size then 
      	finalOutput[1] = htable
      end      
      if t < size or size == 1 then
      	allOutput[1][t] = htable
      end
    else
      self.output = {}
      for i = 1, self.num_layers do
        self.output[i] = htable[i]
        if t == size then 
      		finalOutput[i] = htable[i]
      	end
      	
      	if t < size or size == 1 then
        	allOutput[i][t] = htable[i]
        end
      end
    end
  end  

  return {allOutput, finalOutput}
end

function LSTM:backward(inputs, grad_outputs, reverse)
  local size = inputs:size(1)
  if self.depth == 0 or self.depth ~= size then
    error("No cells to backpropagate through")
  end

  local input_grads = torch.Tensor(inputs:size())
  for t = size, 1, -1 do
    local input = reverse and inputs[size - t + 1] or inputs[t]
    local grad_output = reverse and grad_outputs[size - t + 1] or grad_outputs[t]
    local cell = self.cells[self.depth]
    local grads = {self.gradInput[2], self.gradInput[3]}
    if self.num_layers == 1 then
      grads[2]:add(grad_output)
    else
      for i = 1, self.num_layers do
        grads[2][i]:add(grad_output[i])
      end
    end

    local prev_output = (self.depth > 1) and self.cells[self.depth - 1].output
                                         or self.initial_values
    self.gradInput = cell:backward({input, prev_output[1], prev_output[2]}, grads)
    if reverse then
      input_grads[size - t + 1] = self.gradInput[1]
    else
      input_grads[t] = self.gradInput[1]
    end
    self.depth = self.depth - 1
  end
  self:forget() -- important to clear out state
  return input_grads
end

function LSTM:backward_returnHidden(inputs, grad_outputs, reverse)
  local size = inputs:size(1)
  if self.depth == 0 or self.depth ~= size then
    error("No cells to backpropagate through")
  end
  local hidden = torch.Tensor(self.mem_dim)
  local input_grads = torch.Tensor(inputs:size())
  for t = size, 1, -1 do
    local input = reverse and inputs[size - t + 1] or inputs[t]
    local grad_output = reverse and grad_outputs[size - t + 1] or grad_outputs[t]
    local cell = self.cells[self.depth]
    local grads = {self.gradInput[2], self.gradInput[3]}
    if self.num_layers == 1 then
      grads[2]:add(grad_output)
    else
      for i = 1, self.num_layers do
        grads[2][i]:add(grad_output[i])
      end
    end

    local prev_output = (self.depth > 1) and self.cells[self.depth - 1].output
                                         or self.initial_values
    self.gradInput = cell:backward({input, prev_output[1], prev_output[2]}, grads)
    if reverse then
      input_grads[size - t + 1] = self.gradInput[1]
    else
      input_grads[t] = self.gradInput[1]
    end
    hidden = self.gradInput[3]:clone()
    self.depth = self.depth - 1
  end
  self:forget() -- important to clear out state
  return hidden
end


function LSTM:share(lstm, ...)
  if self.in_dim ~= lstm.in_dim then error("LSTM input dimension mismatch") end
  if self.mem_dim ~= lstm.mem_dim then error("LSTM memory dimension mismatch") end
  if self.num_layers ~= lstm.num_layers then error("LSTM layer count mismatch") end
  if self.gate_output ~= lstm.gate_output then error("LSTM output gating mismatch") end
  share_params(self.master_cell, lstm.master_cell, ...)
end

function LSTM:zeroGradParameters()
  self.master_cell:zeroGradParameters()
end

function LSTM:parameters()
  return self.master_cell:parameters()
end

-- Clear saved gradients
function LSTM:forget()
  self.depth = 0
  for i = 1, #self.gradInput do
    local gradInput = self.gradInput[i]
    if type(gradInput) == 'table' then
      for _, t in pairs(gradInput) do t:zero() end
    else
      self.gradInput[i]:zero()
    end
  end
end
