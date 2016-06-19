
local Seq2Seq = {}

function Seq2Seq.encoder(opt)
	if opt.LSTM then
		local enc = require "LSTM"
		return enc.lstm(opt)
	elseif opt.GRU then
		local enc = require "GRU"
		return enc.gru(opt)
	elseif opt.RNN then
		local enc = require "ReNN"
		return enc.rnn(opt)
	end
end

function Seq2Seq.decoder(opt)
	local inputs = {}
	local outputs = {}
	local prev_y = nn.Identity()()
	local prev_h = nn.Identity()()
	local next_h = nn.Identity()() -- output
	local context = nn.Identity()() -- last hidden state of encoder
	table.insert(inputs,prev_y)
	table.insert(inputs,prev_h)
	table.insert(inputs,context)
	-- common function 
	function new_input_sum()
		local i2h = nn.Linear(opt.rnn_size,opt.rnn_size)(prev_y)
		local h2h = nn.Linear(opt.rnn_size,opt.rnn_size)(prev_h)
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
		next_h = nn.CMulTable()({out_gate,nn.Tanh()(next_c)})
		table.insert(outputs,next_c)
		--  end of lstm time step 
	elseif opt.GRU then
		-- start of gru time step
		local z = nn.Sigmoid()(new_input_sum())
		local r = nn.Sigmoid()(new_input_sum())
		local h_hat = nn.Tanh()(nn.CAddTable()({
			nn.Linear(opt.rnn_size,opt.rnn_size)(prev_y),
			nn.Linear(opt.rnn_size,opt.rnn_size)(nn.CMulTable()({r,prev_h}))
		}))
		next_h = nn.CAddTable()({
			nn.CMulTable()({nn.AddConstant(1,false)(nn.MulConstant(-1,false)(z)),prev_h}),
			nn.CMulTable()({z,h_hat})
		})
		-- end of gru time step
	elseif opt.RNN then
		next_h = nn.Sigmoid()(new_input_sum())
	end

	table.insert(outputs,next_h)

	return nn.gModule(inputs,outputs) -- outputs:next_y_softmax, word probability in vocabulary
end

return Seq2Seq
