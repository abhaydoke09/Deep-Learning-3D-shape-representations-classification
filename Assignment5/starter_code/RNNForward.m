function [output, model] = RNNForward(x, model)

% Inputs:
% - x: input data of size (N, T, D) per batch (see trainRNN, testRNN)
% - model: RNN model containing weights and states.

% Outputs:
% - output: landmark prediction of size (N, M)  per batch
% - model: RNN model containing weights and updated states (useful for  backpropagation)

% Weights/States from model you may want to use here
% - model.state.h: hidden states for your batch - it has size (N, H, T)
% - model.param.Wo: weight matrix for hidden-to-output connections - it has size (H, M)
% - model.param.bo: biases - it has size (1, M)

[N, T, D] = size(x);
[H, ~] = size(model.param.Wh);

% h0: hidden state with all-zeros for first window frame - it has size of size (N, H)
% h: hidden states for all window frames - it must have size (N, H, T)
h0 = zeros(N, H);
model.state.h = zeros(N, H, T);

% stepwise forward propagation for the time-delayed RNN's hidden layer
model.state.h(:,:,1) = RNNForwardStep(squeeze(x(:,1,:)), h0, model);
for t = 1:T-1
    model.state.h(:,:,t+1) = RNNForwardStep(squeeze(x(:,t+1,:)), model.state.h(:,:,t), model);
end

%% YOUR CODE goes here - modify the following lines!

%% use model.state.h(:,:,end) to compute the output!
output = model.state.h(:,:,end) * model.param.Wo + model.param.bo;

end


