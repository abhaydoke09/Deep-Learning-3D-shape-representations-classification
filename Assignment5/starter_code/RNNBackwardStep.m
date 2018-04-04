function [ dprev_h, dWx, dWh, db ] = RNNBackwardStep( dnext_h, x, model, t )

% Inputs:
%  - dnext_h: backpropagated message from hidden state of next timestep/output layer - size (N, H)
%  - x: Input data for batch - size (N, T, D).
%  - model: trained model containing weights, states, derivatives
%  - t: time step index

% Outputs:
%  - dnext_h: backpropagated message towards previous timestep  - size (N, H)
%  - dWx: Derivative of weight matrix for input-to-hidden connections, Wx
%  - dWh: Derivative of weight matrix for hidden-to-hidden connections, Wh
%  - db: Derivative of biases for hidden state, b

% Weights/States from model you may want to use
%  - model.state.h: hidden states for the entire timeseries - size (N, H, T)
%  - model.param.Wx: weight matrix for input-to-hidden connections - size (D, H)
%  - model.param.Wh: weight matrix for hidden-to-hidden connections - size (H, H)
%  - model.param.Wo: weight matrix for hidden-to-output connections - size (H, M)
%  - model.param.b: biases for hidden state - size (1, H)
%  - model.param.bo: biases for output - size  (1, M)

x = squeeze(x(:,t,:)); % squeeze the time-step dimension

%% YOUR CODE goes here - modify the following lines!

% derivative based on chain rule
[N, H] = size(dnext_h);

dnext_h = dnext_h.*(1 - model.state.h(:,:,t).^2);

if t==1
    dWh = 0.0;
else
    dWh = dnext_h' * model.state.h(:,:,t-1);
end
    
dWx = x' * dnext_h;
db = sum(dnext_h);

% backward message based on chain rule
dprev_h = dnext_h * model.param.Wh;

end

