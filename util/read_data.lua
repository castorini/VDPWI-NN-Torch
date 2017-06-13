--[[

  Functions for loading data from disk.

--]]

function similarityMeasure.read_embedding(vocab_path, emb_path)
  print("Embedding: " .. vocab_path)
  local vocab = similarityMeasure.Vocab(vocab_path)
  local embedding = torch.load(emb_path)
  return vocab, embedding
end

function similarityMeasure.read_stop_words(path, vocab)
  local file = io.open(path, 'r')
  local stopWord = {}
  while true do
    line = file:read()
    if line == nil then break end
    if vocab:contains(line) then
      stopWord[vocab:index(line)] = 1        
    end    
  end  
  file:close()
  return stopWord
end

function similarityMeasure.read_sentences(path, vocab, fixed, stopWord)
  local sentences = {}
  local sentencesRaw = {}
  local sentencesStop = {}
  local maxSize = 16
  local file = io.open(path, 'r')
  local line
  local maxFix = 0
  
  fixed = false --true -- this is padding for 3. 
  while true do
    line = file:read()
    if line == nil then break end
    local tokens = stringx.split(line)
    local len = #tokens
    if len > maxFix then
 	    maxFix = len
    end
    if len > 16 then
	    --print(len.." || "..line)
    end
    --local sent = torch.IntTensor(math.max(len,3))
    local padLen = len
    if fixed and len < 3 then
      padLen = 3
    end
    local sent = torch.IntTensor(padLen)
    local stopSeq = torch.IntTensor(padLen)
    
    local counter = 0
    for i = 1, len do
      local token = tokens[i]
      sent[i] = vocab:index(token)
      --if stopWord[sent[i]] == 1 then
      --  stopSeq[i] = 1     
      --else
      --  stopSeq[i] = 0
      --end
    end
    
    ---Maynot be useful th following block
    --[[
    if fixed and len < maxSize then
      for i = padLen+1, maxSize do
	     sent[i] = vocab:index("<@<BLANKER>@>") -- sent[len]
      end
    else
    --]]
    if fixed and len < 3 then
      for i = len+1, padLen do
	     sent[i] = vocab:index("<@<BLANKER>@>") -- sent[len]
      end
    end
        
    if sent == nil then print('line: '..line) end
    sentences[#sentences + 1] = sent
    sentencesStop[#sentencesStop + 1] = stopSeq
    sentencesRaw[#sentencesRaw + 1] = line
    --print(line)
    --print(stopSeq)
  end  
  file:close()

  return sentences, maxFix, sentencesStop, sentencesRaw
end

function similarityMeasure.read_relatedness_dataset(dir, vocab, task)
  
  local dataset = {}
  dataset.vocab = vocab
  if task == 'twitter' then
	  file1 = 'tokenize_doc2.txt'
	  file2 = 'tokenize_query2.txt'
  else 
	  file1 = 'a.toks'
	  file2 = 'b.toks'
  end
  local stopWord = nil --similarityMeasure.read_stop_words('util/english', vocab)
  dataset.lsents, dataset.lmaxsize, dataset.lstop, dataset.lraw = similarityMeasure.read_sentences(dir .. file1, vocab, false, stopWord)
  dataset.rsents, dataset.rmaxsize, dataset.rstop, dataset.rraw = similarityMeasure.read_sentences(dir .. file2, vocab, false, stopWord)

  dataset.size = #dataset.lsents
  local id_file = torch.DiskFile(dir .. 'id.txt')
  local sim_file = torch.DiskFile(dir .. 'sim.txt')
  dataset.ids = torch.IntTensor(dataset.size)
  dataset.labels = torch.Tensor(dataset.size)
  dataset.labelsReal = torch.Tensor(dataset.size)
  
  local num_classes = 0
  
  if task == 'sic' then
    num_classes = 5
  elseif task == 'vid' or task == 'seme' or task ==  'smteur' or task == 'msrpar'
          or task == 'sts2013' or task == 'sts2014' or task == 'sts2012' then
    num_classes = 6
  elseif task == 'twitter' or task == 'mspr' or task == 'qa' or task == 'wikiqa' then
    num_classes = 2
    print("not ready yet for sparsier target")
  else
    error("not possible task!")
  end
    
  if task == 'vid' or task == 'sts2013' or task == 'sic' then  --only those having sparse target
    dataset.labelsSparse = torch.Tensor(dataset.size, num_classes)
    local file = io.open(dir .. 'sim.sparse.txt', 'r')
    local line
    local maxFix = 0
    for i = 1, dataset.size do
      line = file:read()
      if line == nil and i <= dataset.size then 
        error("not possible reading sparse file") 
      end
      local tokens = stringx.split(line, "\t")
      local len = #tokens
      if len ~= num_classes then
        error("not possible here" .. line)
      end      
      for j = 1, len do
        dataset.labelsSparse[i][j] = tokens[j]        
      end
      --print(dataset.labelsSparse[i]:sum())
    end
  end
  
  if task == 'twitter' or task == 'qa' or task == 'wikiqa' then  
    local boundary_file, _ = io.open(dir .. 'boundary.txt')
    local numrels_file = torch.DiskFile(dir .. 'numrels.txt')
    -- read boundary data
    local boundary, counter = {}, 0
    while true do
      line = boundary_file:read()
      if line == nil then break end
      counter = counter + 1
      boundary[counter] = tonumber(line)
    end
    boundary_file:close()
    dataset.boundary = torch.IntTensor(#boundary)
    for counter, bound in pairs(boundary) do
      dataset.boundary[counter] = bound
    end  
    -- read numrels data
    dataset.numrels = torch.IntTensor(#boundary-1)
    for i = 1, #boundary-1 do
      dataset.numrels[i] = numrels_file:readInt()
    end
    numrels_file:close()
  end

  for i = 1, dataset.size do
    --dataset.ids[i] = id_file:readInt()
    if task == 'sic' then
      dataset.labels[i] = 0.25 * (sim_file:readDouble() - 1) -- sic data
      dataset.labelsReal[i] = dataset.labels[i]*4+1 
    elseif task == 'vid' or task ==  'smteur' or task == 'msrpar'
      or task == 'sts2013' or task == 'sts2014' or task == 'sts2012' then
	    dataset.labels[i] = 0.2 * (sim_file:readDouble()) -- vid data
      dataset.labelsReal[i] = dataset.labels[i]*5
    elseif task == 'mspr' or task == 'seme' then
    	dataset.labels[i] = (sim_file:readDouble()) -- msp and seme
      dataset.labelsReal[i] = dataset.labels[i]
    elseif task == 'twitter' or task == 'qa' or task == 'wikiqa' then
    	dataset.labels[i] = sim_file:readInt() + 1 -- twitter
      dataset.labelsReal[i] = dataset.labels[i]-1
    else
      error("not possible task in read_data")
    end    
  end
  id_file:close()
  sim_file:close()
  return dataset
end

---This is filter version, only apply to train data
function similarityMeasure.read_relatedness_dataset_limit(dir, vocab, task, limit)
  local dataset = {}
  dataset.vocab = vocab
  if task == 'twitter' then
	  file1 = 'tokenize_doc2.txt'
	  file2 = 'tokenize_query2.txt'
  else 
	  file1 = 'a.toks'
	  file2 = 'b.toks'
  end
  local stopWord = similarityMeasure.read_stop_words('util/english', vocab)
  dataset.lsents_a, dataset.lmaxsize, dataset.lstop_a, dataset.lraw_a = similarityMeasure.read_sentences(dir .. file1, vocab, false, stopWord)
  dataset.rsents_a, dataset.rmaxsize, dataset.rstop_a, dataset.rraw_a = similarityMeasure.read_sentences(dir .. file2, vocab, false, stopWord)

  dataset.size_a = #dataset.lsents_a
  local id_file = torch.DiskFile(dir .. 'id.txt')
  local sim_file = torch.DiskFile(dir .. 'sim.txt')
  dataset.ids_a = torch.IntTensor(dataset.size_a)
  dataset.labels_a = torch.Tensor(dataset.size_a)
  
  local num_classes = 0  
  if task == 'sic' then
    num_classes = 5
  elseif task == 'vid' or task == 'seme' or task ==  'smteur' or task == 'msrpar'
          or task == 'sts2013' or task == 'sts2014' or task == 'sts2012' then
    num_classes = 6
  elseif task == 'twitter' or task == 'mspr' then
    num_classes = 2
    print("not ready yet for sparsier target")
  else
    error("not possible task!")
  end
  
  if task == 'vid' or task == 'sts2013' or task == 'sic' then  --only those having sparse target
    dataset.labelsSparse_a = torch.Tensor(dataset.size_a, num_classes)
    local file = io.open(dir .. 'sim.sparse.txt', 'r')
    local line
    local maxFix = 0
    for i = 1, dataset.size_a do
      line = file:read()
      if line == nil and i <= dataset.size_a then 
        error("not possible reading sparse file") 
      end
      local tokens = stringx.split(line, "\t")
      local len = #tokens
      if len ~= num_classes then
        error("not possible here" .. line)
      end      
      for j = 1, len do
        dataset.labelsSparse_a[i][j] = tokens[j]        
      end
      --print(dataset.labelsSparse[i]:sum())
    end
  end
  
  for i = 1, dataset.size_a do
    dataset.ids_a[i] = id_file:readInt()
    if task == 'sic' then
    	dataset.labels_a[i] = 0.25 * (sim_file:readDouble() - 1) -- sic data
    elseif task == 'vid' or task ==  'smteur' or task == 'msrpar'
      or task == 'sts2013' or task == 'sts2014' or task == 'sts2012' then
	    dataset.labels_a[i] = 0.2 * (sim_file:readDouble()) -- vid data
    elseif task == 'mspr' or task == 'seme' then
    	dataset.labels_a[i] = (sim_file:readDouble()) -- msp and seme
    elseif task == 'twitter' then
    	dataset.labels_a[i] = sim_file:readInt() + 1 -- twitter
    else
      error("not possible task in read_data")
    end
  end
  id_file:close()
  sim_file:close()
  
  ----Process the limit
  local cc = 0
  dataset.lsents = {}
  dataset.rsents = {}
  dataset.lstop = {}
  dataset.rstop = {}
  dataset.rraw = {}
  dataset.lraw = {}
  
  for i = 1, dataset.size_a do
    if dataset.lsents_a[i]:size(1) <= limit and dataset.rsents_a[i]:size(1) <= limit then
      cc = cc + 1
      dataset.lsents[cc] = dataset.lsents_a[i]
      dataset.rsents[cc] = dataset.rsents_a[i]
      dataset.lstop[cc] = dataset.lstop_a[i]
      dataset.rstop[cc] = dataset.rstop_a[i]
      dataset.lraw[cc] = dataset.lraw_a[i]
      dataset.rraw[cc] = dataset.rraw_a[i]      
    end
  end  
  dataset.ids = torch.IntTensor(cc)
  dataset.labels = torch.Tensor(cc)  
  dataset.labelsSparse = torch.Tensor(cc, num_classes)
  
  local cc2 = 0  
  for i = 1, dataset.size_a do
    if dataset.lsents_a[i]:size(1) <= limit and dataset.rsents_a[i]:size(1) <= limit then
      cc2 = cc2 + 1
      dataset.labels[cc2] = dataset.labels_a[i]
      dataset.ids[cc2] = dataset.ids_a[i]   
      if task == 'vid' or task == 'sts2013' or task == 'sic' then  
        dataset.labelsSparse[cc2] = dataset.labelsSparse_a[i]
      end 
    end 
  end
  if cc2 ~= cc then
    error("not possible")
  end
  dataset.size = cc
  
  return dataset
end


--------------------------POS reading version-------------
function similarityMeasure.read_sentences_pos(path, vocab)
  --local sentences = {}
  --local sentencesRaw = {}
  --local sentencesStop = {}
  local sentencesPos = {}
  local maxSize = 16
  local file = io.open(path, 'r')
  local line
  local maxFix = 0
  
  fixed = true -- this is padding for 3. 
  while true do
    line = file:read()
    if line == nil then break end
    local tokens = stringx.split(line)
    local len = #tokens
    if len > maxFix then
 	    maxFix = len
    end
    --local sent = torch.IntTensor(math.max(len,3))
    local padLen = len
    if fixed and len < 3 then
      padLen = 3
    end
    local sent = torch.IntTensor(padLen)
    
    local counter = 0
    for i = 1, len do
      local token = tokens[i]
      sent[i] = vocab:index(token)    
    end
    
    if fixed and len < 3 then
      for i = len+1, padLen do
	     sent[i] = vocab:index("</s>") -- sent[len]
      end
    end
        
    if sent == nil then print('line: '..line) end
    --sentences[#sentences + 1] = sent
    --sentencesStop[#sentencesStop + 1] = stopSeq
    --sentencesRaw[#sentencesRaw + 1] = line
    sentencesPos[#sentencesPos + 1] = sent
    --print(line)    
  end  
  file:close()

  return sentencesPos 
end

function similarityMeasure.read_relatedness_dataset_pos(dir, vocab, task, posVocab)
  local dataset = {}
  dataset.vocab = vocab
  if task == 'twitter' then
	  file1 = 'tokenize_doc2.txt'
	  file2 = 'tokenize_query2.txt'
  else 
	  file1 = 'a.toks'
	  file2 = 'b.toks'
  end
  
  local stopWord = similarityMeasure.read_stop_words('util/english', vocab)
  dataset.lsents, dataset.lmaxsize, dataset.lstop, dataset.lraw = similarityMeasure.read_sentences(dir .. file1, vocab, false, stopWord)
  dataset.rsents, dataset.rmaxsize, dataset.rstop, dataset.rraw = similarityMeasure.read_sentences(dir .. file2, vocab, false, stopWord)

  dataset.lpos = similarityMeasure.read_sentences_pos(dir .. 'a.pos', posVocab) -- ONLY DIFFERENCE
  dataset.rpos = similarityMeasure.read_sentences_pos(dir .. 'b.pos', posVocab) -- ONLY DIFFERENCE
  
  dataset.size = #dataset.lsents
  local id_file = torch.DiskFile(dir .. 'id.txt')
  local sim_file = torch.DiskFile(dir .. 'sim.txt')
  dataset.ids = torch.IntTensor(dataset.size)
  dataset.labels = torch.Tensor(dataset.size)
  dataset.labelsReal = torch.Tensor(dataset.size)
  
  local num_classes = 0
  
  if task == 'sic' then
    num_classes = 5
  elseif task == 'vid' or task == 'seme' or task ==  'smteur' or task == 'msrpar'
          or task == 'sts2013' or task == 'sts2014' or task == 'sts2012' then
    num_classes = 6
  elseif task == 'twitter' or task == 'mspr' or task == 'qa' or task == 'wikiqa' then
    num_classes = 2
    print("not ready yet for sparsier target")
  else
    error("not possible task!")
  end
    
  if task == 'vid' or task == 'sts2013' or task == 'sic' then  --only those having sparse target
    dataset.labelsSparse = torch.Tensor(dataset.size, num_classes)
    local file = io.open(dir .. 'sim.sparse.txt', 'r')
    local line
    local maxFix = 0
    for i = 1, dataset.size do
      line = file:read()
      if line == nil and i <= dataset.size then 
        error("not possible reading sparse file") 
      end
      local tokens = stringx.split(line, "\t")
      local len = #tokens
      if len ~= num_classes then
        error("not possible here" .. line)
      end      
      for j = 1, len do
        dataset.labelsSparse[i][j] = tokens[j]        
      end
      --print(dataset.labelsSparse[i]:sum())
    end
  end
  
  if task == 'twitter' or task == 'qa' or task == 'wikiqa' then  
    local boundary_file, _ = io.open(dir .. 'boundary.txt')
    local numrels_file = torch.DiskFile(dir .. 'numrels.txt')
    -- read boundary data
    local boundary, counter = {}, 0
    while true do
      line = boundary_file:read()
      if line == nil then break end
      counter = counter + 1
      boundary[counter] = tonumber(line)
    end
    boundary_file:close()
    dataset.boundary = torch.IntTensor(#boundary)
    for counter, bound in pairs(boundary) do
      dataset.boundary[counter] = bound
    end  
    -- read numrels data
    dataset.numrels = torch.IntTensor(#boundary-1)
    for i = 1, #boundary-1 do
      dataset.numrels[i] = numrels_file:readInt()
    end
    numrels_file:close()
  end

  for i = 1, dataset.size do
    --dataset.ids[i] = id_file:readInt()
    if task == 'sic' then
      dataset.labels[i] = 0.25 * (sim_file:readDouble() - 1) -- sic data
      dataset.labelsReal[i] = dataset.labels[i]*4+1 
    elseif task == 'vid' or task ==  'smteur' or task == 'msrpar'
      or task == 'sts2013' or task == 'sts2014' or task == 'sts2012' then
	    dataset.labels[i] = 0.2 * (sim_file:readDouble()) -- vid data
      dataset.labelsReal[i] = dataset.labels[i]*5
    elseif task == 'mspr' or task == 'seme' then
    	dataset.labels[i] = (sim_file:readDouble()) -- msp and seme
      dataset.labelsReal[i] = dataset.labels[i]
    elseif task == 'twitter' or task == 'qa' or task == 'wikiqa' then
    	dataset.labels[i] = sim_file:readInt() + 1 -- twitter
      dataset.labelsReal[i] = dataset.labels[i]-1
    else
      error("not possible task in read_data")
    end    
  end
  id_file:close()
  sim_file:close()
  return dataset
end

---This is filter version, only apply to train data
function similarityMeasure.read_relatedness_dataset_limit_pos(dir, vocab, task, limit, posVocab)
  local dataset = {}
  dataset.vocab = vocab
  if task == 'twitter' then
	  file1 = 'tokenize_doc2.txt'
	  file2 = 'tokenize_query2.txt'
  else 
	  file1 = 'a.toks'
	  file2 = 'b.toks'
  end
  local stopWord = similarityMeasure.read_stop_words('util/english', vocab)
  dataset.lsents_a, dataset.lmaxsize, dataset.lstop_a, dataset.lraw_a = similarityMeasure.read_sentences(dir .. file1, vocab, false, stopWord)
  dataset.rsents_a, dataset.rmaxsize, dataset.rstop_a, dataset.rraw_a = similarityMeasure.read_sentences(dir .. file2, vocab, false, stopWord)

  dataset.lpos_a = similarityMeasure.read_sentences_pos(dir .. 'a.pos', posVocab)
  dataset.rpos_a = similarityMeasure.read_sentences_pos(dir .. 'b.pos', posVocab)
  
  dataset.size_a = #dataset.lsents_a
  local id_file = torch.DiskFile(dir .. 'id.txt')
  local sim_file = torch.DiskFile(dir .. 'sim.txt')
  dataset.ids_a = torch.IntTensor(dataset.size_a)
  dataset.labels_a = torch.Tensor(dataset.size_a)
  
  local num_classes = 0  
  if task == 'sic' then
    num_classes = 5
  elseif task == 'vid' or task == 'seme' or task ==  'smteur' or task == 'msrpar'
          or task == 'sts2013' or task == 'sts2014' or task == 'sts2012' then
    num_classes = 6
  elseif task == 'twitter' or task == 'mspr' then
    num_classes = 2
    print("not ready yet for sparsier target")
  else
    error("not possible task!")
  end
  
  if task == 'vid' or task == 'sts2013' or task == 'sic' then  --only those having sparse target
    dataset.labelsSparse_a = torch.Tensor(dataset.size_a, num_classes)
    local file = io.open(dir .. 'sim.sparse.txt', 'r')
    local line
    local maxFix = 0
    for i = 1, dataset.size_a do
      line = file:read()
      if line == nil and i <= dataset.size_a then 
        error("not possible reading sparse file") 
      end
      local tokens = stringx.split(line, "\t")
      local len = #tokens
      if len ~= num_classes then
        error("not possible here" .. line)
      end      
      for j = 1, len do
        dataset.labelsSparse_a[i][j] = tokens[j]        
      end
      --print(dataset.labelsSparse[i]:sum())
    end
  end
  
  for i = 1, dataset.size_a do
    dataset.ids_a[i] = id_file:readInt()
    if task == 'sic' then
    	dataset.labels_a[i] = 0.25 * (sim_file:readDouble() - 1) -- sic data
    elseif task == 'vid' or task ==  'smteur' or task == 'msrpar'
      or task == 'sts2013' or task == 'sts2014' or task == 'sts2012' then
	    dataset.labels_a[i] = 0.2 * (sim_file:readDouble()) -- vid data
    elseif task == 'mspr' or task == 'seme' then
    	dataset.labels_a[i] = (sim_file:readDouble()) -- msp and seme
    elseif task == 'twitter' then
    	dataset.labels_a[i] = sim_file:readInt() + 1 -- twitter
    else
      error("not possible task in read_data")
    end
  end
  id_file:close()
  sim_file:close()
  
  ----Process the limit
  local cc = 0
  dataset.lsents = {}
  dataset.rsents = {}
  dataset.lstop = {}
  dataset.rstop = {}
  dataset.rraw = {}
  dataset.lraw = {}
  dataset.lpos = {}
  dataset.rpos = {}
  
  for i = 1, dataset.size_a do
    if dataset.lsents_a[i]:size(1) <= limit and dataset.rsents_a[i]:size(1) <= limit then
      cc = cc + 1
      dataset.lsents[cc] = dataset.lsents_a[i]
      dataset.rsents[cc] = dataset.rsents_a[i]
      dataset.lstop[cc] = dataset.lstop_a[i]
      dataset.rstop[cc] = dataset.rstop_a[i]
      dataset.lraw[cc] = dataset.lraw_a[i]
      dataset.rraw[cc] = dataset.rraw_a[i]      
      dataset.lpos[cc] = dataset.lpos_a[i]
      dataset.rpos[cc] = dataset.rpos_a[i]
    end
  end  
  dataset.ids = torch.IntTensor(cc)
  dataset.labels = torch.Tensor(cc)  
  dataset.labelsSparse = torch.Tensor(cc, num_classes)
  
  local cc2 = 0  
  for i = 1, dataset.size_a do
    if dataset.lsents_a[i]:size(1) <= limit and dataset.rsents_a[i]:size(1) <= limit then
      cc2 = cc2 + 1
      dataset.labels[cc2] = dataset.labels_a[i]
      dataset.ids[cc2] = dataset.ids_a[i]   
      if task == 'vid' or task == 'sts2013' or task == 'sic' then  
        dataset.labelsSparse[cc2] = dataset.labelsSparse_a[i]
      end 
    end 
  end
  if cc2 ~= cc then
    error("not possible")
  end
  dataset.size = cc
  
  return dataset
end
