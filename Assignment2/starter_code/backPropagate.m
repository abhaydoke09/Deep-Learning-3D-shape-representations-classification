%function [sent_msg, derivatives] = backPropagate(received_msg, layer_param, layer_output, layer_input, weight_decay, is_output_layer)
function [sent_msg, param_derivatives, bias_derivatives] = backPropagate(received_msg, layer_id, model, weight_decay, activation);
% code for back propagation through a sigmoid layer
% you need to complete this script!

% Input:
% 'received_msg': are the messages sent to this layer from another layer in the net
% 'layer_param' represents the parameters of the layer used to transform the input 
% data (see trainNN.m, forwardPropagate.m)
% 'layer_output' represents the output data produced by this layer during
% forward propagation (N x Do matrix, where Do is the number of output nodes of this layer)
% 'layer_input' represents the input data to this layer during during 
% forward propagation  (N x Di matrix, where N is the number of input samples, 
% Di is the number of dimensions)
% 'weight_decay' is the L2 regularization parameter
% 'is_output_layer' is a binary variable which should be true if this layer is the
% output classification layer, and false otherwise

% Output:
% 'derivatives' should store the derivatives of the parameters of this layer
% 'sent_msg' should store the messages emitted by this layer 

% feel free to modify the input and output arguments if necessary

%% YOUR CODE goes here - modify the following lines!

param_derivatives = model.outputs{layer_id-1}' * received_msg;
bias_derivatives = sum(model.biases{layer_id} .* received_msg, 1);
sent_msg = (model.param{layer_id} * received_msg')';
if layer_id>2
    % Checking which kind of activation was used. If activation=2, then
    % sigmoid was used else relu was used
    if activation == 2
        sigmoid_derivative = (model.outputs{layer_id-1} .* (1-model.outputs{layer_id-1}));
        sent_msg = sent_msg .* sigmoid_derivative;
    else
        relu_derivative = model.outputs{layer_id-1};
        relu_derivative(relu_derivative<=0) = 0;
        relu_derivative(relu_derivative>0) = 1;
        sent_msg = sent_msg .* relu_derivative;
    end 
end

end
