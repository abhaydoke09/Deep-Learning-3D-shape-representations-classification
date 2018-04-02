function next_h = RNNForwardStep( x, prev_h, model )

% Inputs:
% - x: Input data for this timestep -  size (N, D).
% - prev_h: hidden state from previous timestep - size (N, H)
% - model: trained model containing weights and states.

% Outputs a tuple of:
% - next_h: Next hidden state - size (N, H)

% Weights/States from model you may want to use
% - model.param.Wx: weight matrix for input-to-hidden connections - size (D, H)
% - model.param.Wh: weight matrix for hidden-to-hidden connections  - size (H, H)
% - model.param.b: biases of shape (1, H)

[N, D] = size(x);

%% YOUR CODE goes here - modify the following lines to compute hidden state!
% next_h = ?
next_h = tanh(x*model.param.Wx + prev_h*model.param.Wh + model.param.b);
end

