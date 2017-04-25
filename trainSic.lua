--[[
  Author: Hua He
--]]

require('torch')
require('nn')
require('nngraph')
require('optim')
require('xlua')
require('sys')
require('lfs')
--require('../')
--require('./util')
--nngraph.setDebug(true)

similarityMeasure = {}

include('util/read_data.lua')
include('util/Vocab.lua')
include('CsDis.lua')
include('init.lua')

printf = utils.printf

-- global paths (modify if desired)
similarityMeasure.data_dir        = 'data'
similarityMeasure.models_dir      = 'trained_models'
similarityMeasure.predictions_dir = 'predictions'

function header(s)
  print(string.rep('-', 80))
  print(s)
  print(string.rep('-', 80))
end

-- read command line arguments
local args = lapp [[
Training script for semantic relatedness prediction on the Twitter dataset.
  -m,--model  (default dependency) Model architecture: [dependency, lstm, bilstm]
  -l,--layers (default 1)          Number of layers (ignored for Tree-LSTM)
  -d,--dim    (default 150)        LSTM memory dimension
]]

--torch.seed()
torch.manualSeed(123)
print('<torch> using the automatic seed: ' .. torch.initialSeed())

-- directory containing dataset files
local data_dir = 'data/sick/'

-- load vocab
local vocab = similarityMeasure.Vocab(data_dir .. 'vocab.txt')

-- load embeddings
print('loading word embeddings')

local emb_dir = './data/glove/'
--local emb_prefix = emb_dir .. 'glove.twitter.27B'
local emb_prefix = emb_dir .. 'glove.840B'
local emb_vocab, emb_vecs = similarityMeasure.read_embedding(emb_prefix .. '.vocab', emb_prefix .. '.300d.th')

local emb_dim = emb_vecs:size(2)

-- use only vectors in vocabulary (not necessary, but gives faster training)
local num_unk = 0
local vecs = torch.Tensor(vocab.size, emb_dim)
for i = 1, vocab.size do
  local w = vocab:token(i)
  if emb_vocab:contains(w) then
    vecs[i] = emb_vecs[emb_vocab:index(w)]
  elseif i == vocab.size then
    vecs[i]:zero()
  else
    num_unk = num_unk + 1
    vecs[i]:uniform(-0.05, 0.05)
  end
end
print('unk count = ' .. num_unk)
emb_vocab = nil
emb_vecs = nil
collectgarbage()
taskD = 'sic'
-- load datasets
print('loading ' .. taskD .. ' datasets')
local train_dir = data_dir .. 'train/'
local dev_dir = data_dir .. 'dev/'
local test_dir = data_dir .. 'test/'
local train_dataset = similarityMeasure.read_relatedness_dataset(train_dir, vocab, taskD)
local dev_dataset = similarityMeasure.read_relatedness_dataset(dev_dir, vocab, taskD)
local test_dataset = similarityMeasure.read_relatedness_dataset(test_dir, vocab, taskD)

local lmax = math.max(train_dataset.lmaxsize, dev_dataset.lmaxsize, test_dataset.lmaxsize)
local rmax = math.max(train_dataset.rmaxsize, dev_dataset.rmaxsize, test_dataset.rmaxsize)

train_dataset.lmaxsize = lmax
dev_dataset.lmaxsize = lmax
test_dataset.lmaxsize = lmax
train_dataset.rmaxsize = rmax
dev_dataset.rmaxsize = rmax
test_dataset.rmaxsize = rmax
printf('lmax = %d | train lmax = %d | dev lmax = %d\n', lmax, train_dataset.lmaxsize, dev_dataset.lmaxsize)
printf('rmax = %d | train rmax = %d | dev rmax = %d\n', rmax, train_dataset.rmaxsize, dev_dataset.rmaxsize)

printf('num train = %d\n', train_dataset.size)
printf('num dev   = %d\n', dev_dataset.size)
printf('num test  = %d\n', test_dataset.size)

-- initialize model
local config = {
  input_dim = 300,
  mem_cols = 300,
  emb_vecs   = vecs,
--  structure  = 'NTM',
  read_heads = 1,
  task       = taskD,
  cont_dim = 250,
  structure = 'lstm',
}

include('NTM.bilstm.entail.share.lua')

local num_epochs = 35

local model = nil
local loadSave = false
if loadSave then
  include('./models/AlignMaxDropLeakyMulti.lua')
  include('./models/very_deep.lua')
  local modeladdr = "/YOUR_LOCAL_ADDRESS/bestModelOnSic.th"
  print("Loading Saved Model: " .. modeladdr)  
  model = torch.load(modeladdr)
  num_epochs = 1
else
  model = ntm.NTM(config)
end

-- print information
header('model configuration')
printf('max epochs = %d\n', num_epochs)
model:print_config()

if lfs.attributes(similarityMeasure.predictions_dir) == nil then
  lfs.mkdir(similarityMeasure.predictions_dir)
end

-- train
local train_start = sys.clock()
local best_dev_score = -1.0
local best_dev_model = model

-- threads
--torch.setnumthreads(4)
--print('<torch> number of threads in used: ' .. torch.getnumthreads())

header('Training model on data: ' .. taskD)

local id = 531
print("Id: " .. id)

for i = 1, num_epochs do
  local start = sys.clock()
  print('--------------- EPOCH ' .. i .. '--- -------------')
  if not loadSave then
    model:trainCombineSeme(train_dataset)
    print('Finished epoch in ' .. ( sys.clock() - start) )
    local dev_predictions = model:predict_dataset(dev_dataset)
    dev_map_score = pearson(dev_predictions, dev_dataset.labels)
    printf('[DEV] score: %.4f\n', dev_map_score)
  end
  
  if not loadSave and dev_map_score >= best_dev_score then
    print("Saving best models onto Disk.")
    torch.save("/YOUR_LOCAL_ADDRESS/savedModel/bestModelOnSic.ep" .. id ..".th", model)
    best_dev_score = dev_map_score
  end
    
  local test_predictions = model:predict_dataset(test_dataset)
  local score = pearson(test_predictions, test_dataset.labels)
  printf('[TEST] score: %.4f\n', score)      
end

print('finished training in ' .. (sys.clock() - train_start))
