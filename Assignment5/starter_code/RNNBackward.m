function model = RNNBackward( output_msg, x, model, options )

% Inputs:
% - output_msg: back-propage message from top output layer - size (N, M).
% - x: Input data for the entire timeseries - size (N, T, D).
% - model: RNN model containing weights, states, derivatives

% Outputs:
%   - model: RNN model containing weights, states and updated derivatives

% Weights/States from model you may want to use
% - model.state.h: hidden states for the current batch - size(N, H, T)  (computed during forward propagation)
% - model.param.Wx: weight matrix for input-to-hidden connections - size (D, H)
% - model.param.Wh: weight matrix for hidden-to-hidden connections - size (H, H)
% - model.param.Wo: weight matrix for hidden-to-output connections - size (H, M)
% - model.param.b: biases for hidden state, of shape (1, H)
% - model.param.bo: biases for output, of shape (1, M)

[N, H, T] = size(model.state.h);
[N, T, D] = size(x);
[N, M] = size(output_msg);

model.param.dWx = zeros(D, H);
model.param.dWh = zeros(H, H);
model.param.dWo = zeros(H, M);
model.param.db = zeros(1, H);
model.param.dbo = zeros(1, M);

%% YOUR CODE goes here - modify the following lines!

%% output layer backward propagation
model.param.dWo = model.state.h(:,:,end)' * output_msg / N ;
model.param.dbo = sum(output_msg)/N;

backward_fc_layer_msg = (model.param.Wo * output_msg')';

%% rnn step backward propagation
% use RNNBackwardStep function for each time step within the window
% i.e., you need to have a backwards loop: for t = T:-1:1   ...
% and use backward_fc_layer_msg
dprev_h = backward_fc_layer_msg;
for t=T:-1:1
    [dprev_h, dWx, dWh, db] = RNNBackwardStep( dprev_h, x, model, t );
    model.param.dWx = model.param.dWx + dWx;
    model.param.dWh = model.param.dWh + dWh;
    model.param.db = model.param.db + db;
end

model.param.dWx = model.param.dWx/N;
model.param.dWh = model.param.dWh/N;
model.param.db = model.param.db/N;
% backward_msg_passing_through_time_step = ?

%% L2 regularization (weight_decay)
model.param.dWo = model.param.dWo + options.weight_decay*model.param.Wo;
model.param.dbo = model.param.dbo + options.weight_decay*model.param.bo;
model.param.dWx = model.param.dWx + options.weight_decay*model.param.Wx;
model.param.dWh = model.param.dWh + options.weight_decay*model.param.Wh;
model.param.db = model.param.db + options.weight_decay*model.param.db;

end

