local app = require('waffle')
local sample = require('sample')

opt = {}
opt.port = 8080

app.set('host', '0.0.0.0')
app.set('port', opt.port)
app.set('debug', true)

function url_decode(str)
  str = string.gsub(str, '+', ' ')
  str = string.gsub(str, '%%(%x%x)', function(h) return string.char(tonumber(h,16)) end)
  str = string.gsub(str, '\r\n', '\n')
  return str
end

app.post('/bot', function(req, res)
	local original_start_text = req.form.input
  	local processed_start_text = ''
  	if original_start_text ~= nil then -- lua's 'not equals' is weird
    	processed_start_text = url_decode(original_start_text):gsub("%s+"," ")
	end
	local output = sample.output(processed_start_text)
	res.json{output=output}

end)
app.listen()