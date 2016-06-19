
local LSTM = {}

function LSTM.lstm(opt)
	local x = nn.Identity()()
	local prev_c = nn.Identity()()
	local prev_h = nn.Identity()()

	function new_input_sum()
		local i2h = nn.Linear(opt.rnn_size,opt.rnn_size)(x)
		local h2h = nn.Linear(opt.rnn_size,opt.rnn_size)(prev_h)
		return nn.CAddTable()({i2h,h2h})
	end

	local in_gate = nn.Sigmoid()(new_input_sum())
	local forget_gate = nn.Sigmoid()(new_input_sum())
	local out_gate = nn.Sigmoid()(new_input_sum())
	local in_tranform = nn.Tanh()(new_input_sum())

	local next_c = nn.CAddTable()({
		nn.CMulTable()({forget_gate,prev_c}),
		nn.CMulTable()({in_gate,in_tranform})
	})

	local next_h = nn.CMulTable()({out_gate,nn.Tanh()(next_c)})

	return nn.gModule({x,prev_h,prev_c},{next_h,next_c})

end

return LSTM