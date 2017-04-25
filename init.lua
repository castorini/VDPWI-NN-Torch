require('torch')
require('nn')
require('nngraph')

ntm = {}
printf = utils.printf


include('layers/Print.lua')
include('models/LSTM.lua')

--torch.setdefaulttensortype('torch.FloatTensor')

function isnan(x)
    return x ~= x
end

function share_params(cell, src, ...)
  for i = 1, #cell.forwardnodes do
    local node = cell.forwardnodes[i]
    if node.data.module then
      node.data.module:share(src.forwardnodes[i].data.module, ...)
    end
  end
end
