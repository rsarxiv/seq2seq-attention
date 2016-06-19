
local GRU = {}

function GRU.gru(opt)

	local x = nn.Identity()()
	local prev_h = nn.Identity()()

	function new_input_sum()
		local i2h = nn.Linear(opt.rnn_size,opt.rnn_size)(x)
		local h2h = nn.Linear(opt.rnn_size,opt.rnn_size)(prev_h)
		return nn.CAddTable()({i2h,h2h})
	end

	local z = nn.Sigmoid()(new_input_sum())
	local r = nn.Sigmoid()(new_input_sum())
	local h_hat = nn.Tanh()(nn.CAddTable()({
		nn.Linear(opt.rnn_size,opt.rnn_size)(x),
		nn.Linear(opt.rnn_size,opt.rnn_size)(nn.CMulTable()({r,prev_h}))
	}))

	local next_h = nn.CAddTable()({
		nn.CMulTable()({nn.AddConstant(1,false)(nn.MulConstant(-1,false)(z)),prev_h}),
		nn.CMulTable()({z,h_hat})
	})
	
	return nn.gModule({x,prev_h},{next_h})

end

return GRU