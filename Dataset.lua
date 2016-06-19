--[[

Dataset Class

]]--

local Dataset = torch.class("Dataset")

function Dataset:__init(opt)
	local dataset = hdf5.open(opt.datafile,"r")
	self.source = dataset:read("source"):all()
	self.target  = dataset:read("target"):all()
	self.batchNum = math.floor(self.source:size(1) / opt.batch_size)
	self.batchSize = opt.batch_size
	self.vocab = self:buildVocab(opt)
  self.batchId = 0
end

function Dataset:buildVocab(opt)
	local id2word = {}
	local word2id = {}
	local f = torch.DiskFile(opt.vocabfile, "r")
  	f:quiet()
  	local word =  f:readString("*l") -- read file by line
  	while word ~= '' do
      	id2word[#id2word+1] = word
      	word2id[word] = #id2word
      	word = f:readString("*l")
  	end
  	return {["id2word"]=id2word,["word2id"]=word2id,["size"]=#id2word}
end

function Dataset:nextBatch()
   local source = {}
   local target = {}
   self.batchId = (self.batchId % self.batchNum)
   for i = self.batchId * self.batchSize + 1, (self.batchId + 1) * self.batchSize do
      table.insert(source,torch.totable(self.source[i]))
      table.insert(target,torch.totable(self.target[i]))
   end
   self.batchId = self.batchId + 1
   return torch.Tensor(source),torch.Tensor(target)
end