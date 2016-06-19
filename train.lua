require 'nngraph'
require 'nn'
require 'torch'
require 'optim'
require 'hdf5'
require 'Embedding'
require 'Dataset' 
local attn = require 'S2SAttention'
local seq2seq = require 'Seq2Seq'
local model_utils = require 'model_utils'


cmd = torch.CmdLine()
cmd:text()
cmd:text('Train a seq2seq bot model with attention mechanism')
cmd:text()
cmd:text('Options')

cmd:option("-batch_size",20,"")
cmd:option("-rnn_size",500,"")
cmd:option("-learning_rate",2e-3,"")
cmd:option("-decay_rate",0.95,"")
cmd:option('-learning_rate_decay',0.97,'learning rate decay')
cmd:option('-learning_rate_decay_after',10,'in number of epochs, when to start decaying the learning rate')
cmd:option("-LSTM",true,"")
cmd:option("-seed",123,"")
cmd:option("-max_epochs",2,"")
cmd:option("-print_every",1,"")
cmd:option("-datafile","data/bot.hdf5","")
cmd:option("-vocabfile","data/bot.dict","")
cmd:option("-savefile","data/bot.t7","")
cmd:option("-cuda",false,"")
cmd:option("-save_every",1000,"")
cmd:option("-seq2seq",false,"")
cmd:option("-attn",true,"")

opt = cmd:parse(arg)

if opt.cuda then
	require "cutorch"
	require "cunn"
	cutorch.manualSeed(opt.seed)
else
	torch.manualSeed(opt.seed)
end

-- dataset

local dataset = Dataset(opt)
opt.vocab_size = dataset.vocab.size
opt.source_length = dataset.source:size(2)
opt.target_length = dataset.target:size(2)
print("vocabulary size is ".. opt.vocab_size)
print("source and target length is "..opt.source_length)

-- define model one time step, then clone them

local protos = {}
protos.embed = Embedding(opt.vocab_size,opt.rnn_size)
-- encoder lstm input:{x,prev_h,prev_c},output:{next_h,next_c}
if opt.seq2seq then
    protos.encoder = seq2seq.encoder(opt)
-- decoder lstm input:{prev_y,prev_s,prev_c,attention},output:{next_y}
    protos.decoder = seq2seq.decoder(opt) -- h_length = opt.seq_length
elseif opt.attn then
    protos.encoder = attn.encoder(opt)
    protos.decoder = attn.decoder(opt)
end
-- output
protos.softmax = nn.Sequential():add(nn.Linear(opt.rnn_size, opt.vocab_size)):add(nn.LogSoftMax())
-- criterion 
protos.criterion = nn.ClassNLLCriterion() 

-- put the above things into one flattened parameters tensor
local params, grad_params = model_utils.combine_all_parameters(protos.embed, protos.encoder, protos.decoder,protos.softmax)
params:uniform(-0.08, 0.08)

print('number of parameters in the model: ' .. params:nElement())

if opt.cuda then
    for k,v in pairs(protos) do v:cuda() end
end

-- make a bunch of clones, AFTER flattening, as that reallocates memory
local clones = {}
for name,proto in pairs(protos) do
    print('cloning '..name)
    if name == "encoder" then
    	clones[name] = model_utils.clone_many_times(proto, opt.source_length, not proto.parameters)
    elseif name == "decoder" then
    	clones[name] = model_utils.clone_many_times(proto, opt.target_length, not proto.parameters)
    else 
    	clones[name] = model_utils.clone_many_times(proto,opt.source_length+opt.target_length,not proto.parameters)
    end
end

-- encoder initial state (zero initially)
local enc_c0 = torch.zeros(opt.batch_size,opt.rnn_size)
if opt.cuda then
	enc_c0 = enc_c0:cuda()
end
local enc_h0 = enc_c0:clone()
-- decoder initial state (zero initially)
local dec_c0 = enc_c0:clone()
local dec_s0 = enc_c0:clone()

-- decoder final state's backward message (dloss/dfinalstate) is 0, since it doesn't influence predictions
local ddecfinalstate_c = enc_c0:clone()
local dencfinalstate_h = enc_c0:clone()
local dencfinalstate_c = enc_c0:clone()

-- do fwd/bwd and return loss, grad_params
function feval()

    grad_params:zero()
    ------------------ get minibatch -------------------
    x,y = dataset:nextBatch()
    if opt.cuda then
    	x = x:cuda()
    	y = y:cuda()
    end

    ------------------- forward pass -------------------
    local enc_embed = {}            -- enc embeddings
    local dec_embed = {}			-- dec embeddings
    local enc_c = {[0]=enc_c0} -- internal cell states of encoder
    local enc_h = {[0]=enc_h0} -- output values in hidden states of encoder
    local dec_c = {[0]=dec_c0} -- internal cell states of decoder
    local dec_s = {} -- output values in hidden states of decoder
    local predictions = {}           -- softmax outputs
    local loss =  0

    -- encoder forward pass
    for t=1,opt.source_length do
        enc_embed[t] = clones.embed[t]:forward(x[{{}, t}])
        enc_h[t], enc_c[t] = unpack(clones.encoder[t]:forward{enc_embed[t], enc_h[t-1], enc_c[t-1]})
    end

    local context
    if opt.attn then
   	    enc_h[0] = nil -- drop the enc_h[0]
        context = enc_h
    elseif opt.seq2seq then
        context = enc_h[opt.source_length]
    end

    context = nn.JoinTable(1):forward(context)
    context = nn.Reshape(opt.batch_size,opt.source_length,opt.rnn_size):forward(context) -- context 

   	dec_embed[0] = enc_embed[opt.source_length]
    -- dec_c[0] = enc_c[opt.source_length]
    dec_s[0] = enc_h[opt.source_length]

    for t=1,opt.target_length do
    	-- {h,prev_y,prev_s,prev_c}
        -- {prev_y,prev_h,context,prev_c}
        dec_c[t],dec_s[t] = unpack(clones.decoder[t]:forward{dec_embed[t-1],dec_s[t-1],context,dec_c[t-1]})
        predictions[t] = clones.softmax[t]:forward(dec_s[t])
        dec_embed[t] = clones.embed[t]:forward(y[{{}, t}])
        loss = loss + clones.criterion[t]:forward(predictions[t], y[{{}, t}])
    end

    loss = loss / opt.target_length

    -- print("loss="..loss)

    ---------------- backward pass -------------------
    -- complete reverse order of the above
    local ddec_embeddings = {}                              -- d loss / d input embeddings
    local denc_embeddings = {}
    local ddec_c = {[opt.target_length]=ddecfinalstate_c}    -- internal cell states of LSTM
    local ddec_s = {} 
    -- initialize gradient at time t to be zeros (there's no influence from future)
    local denc_c = {[opt.source_length]=dencfinalstate_c}   
    local denc_h = {}--{[opt.source_length]=dencfinalstate_h}

    for t=opt.target_length,1,-1 do
        local doutput_t = clones.criterion[t]:backward(predictions[t],y[{{}, t}]) 
        -- backprop through decoder LSTM timestep
        if t == opt.target_length then
            ddec_s[t] = clones.softmax[t]:backward(dec_s[t], doutput_t)
        else
            ddec_s[t]:add(clones.softmax[t]:backward(dec_s[t], doutput_t))
        end
        ddec_embeddings[t], ddec_s[t-1],dcontext,ddec_c[t-1] = unpack(clones.decoder[t]:backward(
            {dec_embed[t], dec_s[t-1],context,dec_c[t-1]},
            {ddec_c[t], ddec_s[t]}
        ))
        clones.embed[t]:backward(y[{{}, t}], ddec_embeddings[t])

        -- backprop through decoder embeddings
        
    end

    denc_h[opt.source_length] = ddec_s[0]
    -- denc_c[opt.source_length] = ddec_c[0]

    for t=opt.source_length,1,-1 do
    	if opt.cuda then
    		denc_h[t] = denc_h[t]:cuda()
    	end
        denc_embeddings[t], denc_h[t-1], denc_c[t-1] = unpack(clones.encoder[t]:backward(
            {enc_embed[t], enc_h[t-1], enc_c[t-1]},
            {denc_c[t], denc_h[t]}
        ))

        -- print(denc_c[t-1])
        clones.embed[t]:backward(x[{{}, t}], denc_embeddings[t])
    end

    -- ------------------------ misc ----------------------
    -- -- transfer final state to initial state (BPTT)
    enc_c0:copy(enc_c[#enc_c])
    enc_h0:copy(enc_h[#enc_h])

    -- -- clip gradient element-wise
    grad_params:clamp(-5, 5)

    return loss, grad_params

end

-- optimization stuff
local losses = {}
local optim_state = {learningRate = opt.learning_rate, alpha = opt.decay_rate}
local iterations = opt.max_epochs * dataset.batchNum
print("totally needs training iterations "..iterations)
for i = 1, iterations do
    local epoch = i / dataset.batchNum
    local timer = torch.Timer()
    local _, loss = optim.rmsprop(feval, params, optim_state) -- rmsprop
    local time = timer:time().real
    losses[#losses + 1] = loss[1]
    -- exponential learning rate decay
    if i % dataset.batchNum == 0 and opt.learning_rate_decay < 1 then
        if epoch >= opt.learning_rate_decay_after then
            local decay_factor = opt.learning_rate_decay
            optim_state.learningRate = optim_state.learningRate * decay_factor -- decay it
            print('decayed learning rate by a factor ' .. decay_factor .. ' to ' .. optim_state.learningRate)
        end
    end
    if i % opt.save_every == 0 then
    	print("saving model...")
        torch.save(opt.savefile, protos)
    end

    if i % opt.print_every == 0 then
        print(string.format("%d/%d (epoch %.3f), train_loss = %6.8f, grad/param norm = %6.4e, time/batch = %.4fs", i, iterations, epoch, loss[1], grad_params:norm() / params:norm(), time))
    end
end








