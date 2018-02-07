function [sent_msg, derivatives] = backPropagate(received_msg, layer_param, layer_output, layer_input, weight_decay, is_output_layer)
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

N = size(layer_input, 1);
Di = size(layer_input, 2);
Do = size(layer_output, 2);
sent_msg = zeros(N, Di);  % change this (obviously)    
derivatives = zeros(Di+1, Do);  % change this (obviously)    

end
