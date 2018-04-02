function model = RNNBackward( output_msg, x, model )

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

model.param.dWx = zeros(D, H);
model.param.dWh = zeros(H, H);
model.param.dWo = zeros(H, M);
model.param.db = zeros(1, H);
model.param.dbo = zeros(1, M);

%% YOUR CODE goes here - modify the following lines!

%% output layer backward propagation
% model.param.dWo = ?
% model.param.dbo = ?
% backward_fc_layer_msg = ?

%% rnn step backward propagation
% use RNNBackwardStep function for each time step within the window
% i.e., you need to have a backwards loop: for t = T:-1:1   ...
% and use backward_fc_layer_msg

% model.param.dWx = ?
% model.param.dWh = ?
% model.param.db = ?
% backward_msg_passing_through_time_step = ?

%% L2 regularization (weight_decay)
% model.param.dWo = ?
% model.param.dbo = ?
% model.param.dWx = ?
% model.param.dWh = ?
% model.param.db = ?


end

