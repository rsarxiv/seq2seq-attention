require 'torch'
require 'nn'
require 'nngraph'
require 'optim'
require 'Embedding'
local senna = require 'senna'

sample = {}

function sample.output(text)
	local tokenizer = senna.Tokenizer()
	cmd = torch.CmdLine()
	cmd:text()
	cmd:text('sample a seq2seq bot model with attention mechanism')
	cmd:text()
	cmd:text('Options')

	cmd:option("-vocabfile","data/bot.dict","")
	cmd:option("-modelfile","data/bot.t7","")
	cmd:option("-seed",123,"")
	cmd:option("-text","What's your name?","")
	cmd:option("-source_length",20,"")
	cmd:option("-target_length",20,"")
	cmd:option("-attn",true,"")
	cmd:option("-seq2seq",false,"")
	cmd:option("-sample",true,"")
	cmd:option("-beamsize",3,"")

	opt = cmd:parse(arg)

	torch.manualSeed(opt.seed)

	function loadDict()
		words = {}
		local f = torch.DiskFile(opt.vocabfile, "r")
  		f:quiet()
  		local word =  f:readString("*l") -- read file by line
  		while word ~= '' do
      		words[#words+1] = word
      		word = f:readString("*l")
  		end
  		return {["words"]=words}
	end

	local vocab = {} -- word to index
	local ivocab = loadDict()["words"] -- index to word
	for k,v in pairs(ivocab) do vocab[v] = k end

	protos = torch.load(opt.modelfile)
	opt.rnn_size = protos.embed.weight:size(2)

	local seed_text = text
	-- build x vector
	-- tokenize
	local tokens = tokenizer:tokenize(seed_text)
	local words = tokens:words()
	local vocab_set = Set(ivocab)

	local x = {[1]=vocab["<bos>"]}
	for i=1,#words do
		if vocab_set[words[i]] then
			x[i+1] = vocab[words[i]]
		else
			x[i+1] = vocab["<unk>"]
		end
	end

	for i=1,opt.source_length-#x-1 do
		table.insert(x,vocab["<blank>"])
	end

	table.insert(x,vocab["<eos>"])

	x = torch.Tensor(x)

	-- print(x)

	-- end of x vector
	-- forward pass
	local prev_c = torch.zeros(1,opt.rnn_size)
	local prev_h = prev_c:clone()
	local enc_h = {}
	local enc_c = {}
	local enc_embed = {}
	local dec_embed = {}
	local dec_c = {}
	local dec_s = {}
	local predictions = {}

	for t=1,opt.source_length do
		enc_embed[t] = protos.embed:forward(torch.Tensor{x[t]})
		enc_h[t], enc_c[t] = unpack(protos.encoder:forward{enc_embed[t], prev_c, prev_h})
		prev_c:copy(enc_c[t]) 
    	prev_h:copy(enc_h[t])
	end

	local context
	if opt.attn then
    	context = enc_h
	elseif opt.seq2seq then
    	context = enc_h[opt.source_length]
	end

	dec_embed[0] = enc_embed[opt.source_length]
	dec_c[0] = enc_c[opt.source_length]
	dec_s[0] = enc_h[opt.source_length]

	--beam search 

	local k = opt.beamsize-- beam size

	function beam_search()

  		local sents = {}
  		local topk_index_prev = torch.Tensor({x[#x]}):repeatTensor(k)

  		local topk_logprob_prev = torch.Tensor({0.0}):repeatTensor(k)
		for t=1,opt.target_length do
			local score = {} -- local 
			local pred_next = {}
    		for i=1,k do
    			local index = topk_index_prev:clone()[i]
    			local logprob = topk_logprob_prev:clone()[i]
    			local dembed = protos.embed:forward(torch.Tensor{index}):clone()
    			dec_c[t],dec_s[t] = unpack(protos.decoder:forward{dembed,dec_c[t-1],context,dec_s[t-1]})
    			local pred = protos.softmax:forward(dec_s[t])[1]
    			if opt.sample then
    			
    				local probs = torch.exp(pred):squeeze()
        			probs:div(torch.sum(probs)) -- renormalize so probs sum to one
        			topk_index_next = torch.multinomial(probs:float(), k)
        			topk_logprob_next = {}
        			for j=1,k do
        				topk_logprob_next[j] = pred[topk_index_next[j]]
        			end  
        		else
        			topk_logprob_next,topk_index_next = pred:topk(k,true) -- k words
        		end
        		for j=1,k do
        			pred_next[j+(i-1)*k] = topk_index_next:clone()[j]
    				score[j+(i-1)*k] = topk_logprob_prev:clone()[i] + topk_logprob_next[j]	
    			end
    		end
    		-- rank the generated k*k words
    		local top_score,top_index = torch.Tensor(score):topk(k,true)
    		-- 
    		for i=1,k do
    			topk_index_prev[i] = pred_next[top_index:clone()[i]] -- get the top k in t
    		end
    		topk_logprob_prev = top_score:clone()
    		table.insert(sents,topk_index_prev:clone())
		end
		return sents
	end

	local sents = beam_search()

	local response = {}
	for i=1,k do
		local r = ""
		for t=1,#sents do
			word = ivocab[sents[t][i]]
			if word ~= "<blank>" and word ~= "<bos>" and word ~= "<eos>" then
				r = r .. " " .. word
			end
		end
		response[i] = r

	end

	return response
end

return sample

