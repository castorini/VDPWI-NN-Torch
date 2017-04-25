--[[ Author: Hua He 
--]]

local AlignMax, Parent = torch.class('nn.AlignMax', 'nn.Module')

function AlignMax:__init(limitS, yoDim)
   Parent.__init(self)
   self.yo_dim = yoDim or 13
   self.limit = limitS or 32
   if self.limit > 128 or self.limit < 0 then
      error('<AlignMax> illegal limit size')
   end
   self.alignMask = torch.zeros(yoDim, self.limit, self.limit)
   self.ind1 = 10 -- 10 
   self.ind2 = 11 -- 11 
end

--Input is outputAllSimi = torch.zeros(self.yo_dim, self.limit, self.limit)
function AlignMax:updateOutput(input)
   self.alignMask:zero()
   local sDim = input[self.ind1]:reshape(self.limit*self.limit) -- limit by limit
   local sortedDim, sIndexes = torch.sort(sDim, true)

   local basketRow = {}
   local basketCol = {}
   local counter = 0
   local maxRow = 0
   local maxCol = 0
   for i = 1, self.limit*self.limit do
	--if counter > self.limit then break end
				
	  local col = sIndexes[i] % self.limit
    if col == 0 then
      col = self.limit
    end
    local row = torch.ceil(sIndexes[i]/self.limit)
  
    --- if this is just padding then we do not do anything
    if input[self.yo_dim][row][col] == 1 then	
      if maxRow < row then
        maxRow = row      
      end
      if maxCol < col then
        maxCol = col      
      end
      if basketRow[row] ~= 1 and basketCol[col] ~= 1 then 
        --this is a good spot
        basketRow[row] = 1		
        basketCol[col] = 1
        counter = counter + 1		
        self.alignMask[{{}, {row}, {col}}]=1
      else
        self.alignMask[{{}, {row}, {col}}]=0.1
        --self.alignMask[{{self.yo_dim}, {row}, {col}}]=0.1
        --self.alignMask[{{1,-2}, {row}, {col}}]=1
      end
    end
   end
   sIndexes = nil
   sDim = nil

   ----Dot product!!------------------------------
   local sDim1 = input[self.ind2]:reshape(self.limit*self.limit) -- limit by limit
   local sortedDim1, sIndexes1 = torch.sort(sDim1, true)

   basketRow = {}
   basketCol = {}
   local counterDot = 0
   for i = 1, self.limit*self.limit do
    if counterDot > counter then break end
          
    local col = sIndexes1[i] % self.limit
    if col == 0 then
      col = self.limit
    end
    local row = torch.ceil(sIndexes1[i]/self.limit)
  
    --- if this is just padding then we do not do anything
    if input[self.yo_dim][row][col] == 1 then	
      if basketRow[row] ~= 1 and basketCol[col] ~= 1 then 
        basketRow[row] = 1		
        basketCol[col] = 1
        counterDot = counterDot + 1		
        self.alignMask[{{}, {row}, {col}}]=1
      end
    end
   end
   ---------------------------------------------
   self.output:resizeAs(input):copy(input)
   self.output:cmul(self.alignMask)

   if false and maxRow <= 10 and maxCol <= 10 then
     --DEBUGGER only, set to TRUE if you want to debug
     print("AlignMax Input:")
     print(input[self.ind1]:sub(1, maxRow, 1, maxCol))
     print("After Mask:")
     print(self.output[self.ind1]:sub(1, maxRow, 1, maxCol))
     print("Mask:")
     print(self.alignMask[1]:sub(1, maxRow, 1, maxCol))
   end
   return self.output
end

function AlignMax:updateGradInput(input, gradOutput)
   self.gradInput:resizeAs(input):copy(gradOutput)
   self.gradInput:cmul(self.alignMask)
   return self.gradInput
end

function AlignMax:__tostring__()
  return string.format('%s(%d)DropLeakyMulti(%d,%d)', torch.type(self), self.limit, self.ind1,self.ind2)
end

