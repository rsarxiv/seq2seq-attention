local S2SAttention = {}

function S2SAttention.encoder(opt)
	if opt.LSTM then
		local enc = require "LSTM"
		return enc.lstm(opt)
	elseif opt.GRU then
		local enc = require "GRU"
		return enc.gru(opt) 
	elseif opt.RNN then
		local enc = require 'ReNN'
		return enc.rnn(opt)
	end
end

function S2SAttention.decoder(opt)
	-- h: all hidden states in encoder,h is a table
	-- if opt.LSTM h includes two parts, one part is cell,one part is hidden state
	-- if opt.GRU h just includes one part, hidden state.
	local inputs = {}
	local outputs = {}
	local h = nn.Identity()() -- encoder hidden states matrix
	local prev_y = nn.Identity()() -- previous input vector y(t-1)
	local prev_s = nn.Identity()() -- previous state vector s(t-1)
	-- local next_s = nn.Identity()() -- next state vector s(t)
	table.insert(inputs,prev_y)
	table.insert(inputs,prev_s)
	table.insert(inputs,h)
	-- calcualte attention vector at each time step
	local s2e = nn.Linear(opt.rnn_size,opt.rnn_size)(prev_s) -- target_t
	local attn = nn.MM()({h, nn.Replicate(1,3)(s2e)}) -- batch_l x source_l x 1
    attn = nn.Sum(3)(attn)
    attn = nn.SoftMax()(attn)
   	attn = nn.Replicate(1,2)(attn) -- batch_l x  1 x source_l

   	-- apply attention to context
   	local context_combined = nn.MM()({attn, h}) -- batch_l x 1 x rnn_size
   	context_combined = nn.Sum(2)(context_combined) -- batch_l x rnn_size
   	context = nn.CAddTable()({context_combined,prev_s})
	-- common function 
	function new_input_sum()
		local i2h = nn.Linear(opt.rnn_size,opt.rnn_size)(prev_y)
		local h2h = nn.Linear(opt.rnn_size,opt.rnn_size)(prev_s)
		local c2h = nn.Linear(opt.rnn_size,opt.rnn_size)(context)
		return nn.CAddTable()({i2h,h2h,c2h})
	end

	if opt.LSTM then
		local prev_c = nn.Identity()()
		table.insert(inputs,prev_c)
		-- start of lstm time step
		local in_gate = nn.Sigmoid()(new_input_sum())
		local forget_gate = nn.Sigmoid()(new_input_sum())
		local out_gate = nn.Sigmoid()(new_input_sum())
		local in_tranform = nn.Tanh()(new_input_sum())
		local next_c = nn.CAddTable()({
			nn.CMulTable()({forget_gate,prev_c}),
			nn.CMulTable()({in_gate,in_tranform})
		})
		table.insert(outputs,next_c)
		next_s = nn.CMulTable()({out_gate,nn.Tanh()(next_c)})
		-- end of lstm time step
	elseif opt.GRU then
		-- start of gru time step
		local z = nn.Sigmoid()(new_input_sum())
		local r = nn.Sigmoid()(new_input_sum())
		local h_hat = nn.Tanh()(nn.CAddTable()({
			nn.Linear(opt.rnn_size,opt.rnn_size)(prev_y),
			nn.Linear(opt.rnn_size,opt.rnn_size)(nn.CMulTable()({r,prev_s}))
		}))
		next_s = nn.CAddTable()({
			nn.CMulTable()({nn.AddConstant(1,false)(nn.MulConstant(-1,false)(z)),prev_s}),
			nn.CMulTable()({z,h_hat})
		})
		-- end of gru time step
	elseif opt.RNN then
		-- start of rnn time step 
		next_s = nn.Sigmoid()(new_input_sum())
		-- end of rnn time step
	end
	
	table.insert(outputs,next_s)

	return nn.gModule(inputs,outputs)

end

return S2SAttention