require 'nn'
--require 'cunn'

function createModel(inputPlane, numberClass, limit)
	print("This is Very Deep Model with " .. inputPlane)

	local backend_name = 'nn'

	local backend
	if false and backend_name == 'cudnn' then
	  require 'cudnn'
	  backend = cudnn
	else
 	  --CPU only please for now
	  backend = nn
	end
	  
	local vgg = nn.Sequential()

	-- building block
	local function ConvBNReLU(nInputPlane, nOutputPlane)
	  vgg:add(backend.SpatialConvolution(nInputPlane, nOutputPlane, 3,3, 1,1, 1,1))
	  --vgg:add(nn.SpatialBatchNormalization(nOutputPlane,1e-3))
	  vgg:add(backend.ReLU(true))
	  return vgg
	end

	-- Will use "ceil" MaxPooling because we want to save as much
	-- space as we can
	local MaxPooling = backend.SpatialMaxPooling

	vgg:add(nn.AlignMax(limit, inputPlane))
	--vgg:add(nn.SpatialDropout(0.3))
	ConvBNReLU(inputPlane,128)
	--ConvBNReLU(164,164)
	vgg:add(MaxPooling(2,2,2,2):ceil())

	ConvBNReLU(128,164)
	vgg:add(MaxPooling(2,2,2,2):ceil())

	ConvBNReLU(164,192)
	vgg:add(MaxPooling(2,2,2,2):ceil())

	ConvBNReLU(192,192)
	vgg:add(MaxPooling(2,2,2,2):ceil())

	-- In the last block of convolutions the inputs are smaller than
	-- the kernels and cudnn doesn't handle that, have to use cunn
	backend = nn
	if limit == 32 then
	  ConvBNReLU(192,128)
	  vgg:add(MaxPooling(2,2,2,2):ceil())
	elseif limit == 64 then
	  ConvBNReLU(192,256)
	  vgg:add(MaxPooling(2,2,2,2):ceil())
	  ConvBNReLU(256,128)
	  vgg:add(MaxPooling(2,2,2,2):ceil())
	elseif limit == 128 then
	  ConvBNReLU(192,256)
  	  vgg:add(MaxPooling(2,2,2,2):ceil())
	  ConvBNReLU(256,128)
	  vgg:add(MaxPooling(2,2,2,2):ceil())
	  ConvBNReLU(128,128)
	  vgg:add(MaxPooling(2,2,2,2):ceil())
	elseif limit == 48 then
	  ConvBNReLU(192,128)
	  vgg:add(MaxPooling(3,3):ceil())
	elseif limit == 96 then
	  ConvBNReLU(192,256)
	  vgg:add(MaxPooling(2,2,2,2):ceil())
	  ConvBNReLU(256,128)
	  vgg:add(MaxPooling(3,3):ceil())
	elseif limit == 16 then
	  ConvBNReLU(192,128)	  		  
	end

	vgg:add(nn.View(128))

	classifier = nn.Sequential()
	classifier:add(nn.Linear(128,128))
	classifier:add(nn.ReLU(true))
	classifier:add(nn.Linear(128,numberClass))
	classifier:add(nn.LogSoftMax())
	vgg:add(classifier)

	-- initialization from MSR
	local function MSRinit(net)
	  local function init(name)
	    for k,v in pairs(net:findModules(name)) do
	      local n = v.kW*v.kH*v.nOutputPlane
	      v.weight:normal(0,math.sqrt(2/n))
	      v.bias:zero()
	    end
	  end
	  -- have to do for both backends
	  init'cudnn.SpatialConvolution'
	  init'nn.SpatialConvolution'
	end

	MSRinit(vgg)

	return vgg
end
