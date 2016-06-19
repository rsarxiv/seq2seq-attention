
local ReNN = {}

function ReNN.rnn(opt)
	local x = nn.Identity()()
	local prev_h = nn.Identity()()
	local i2h = nn.Linear(opt.rnn_size,opt.rnn_size)(x)
	local h2h = nn.Linear(opt.rnn_size,opt.rnn_size)(prev_h)
	local next_h = nn.Sigmoid()(nn.CAddTable()({i2h,h2h}))

	return nn.gModule({x,prev_h},{next_h})
end

return ReNN