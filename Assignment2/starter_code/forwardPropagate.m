function layer_output = forwardPropagate( layer_input, layer_param, layer_biases, activation )
% code for forward propagation through a sigmoid layer
% you need to complete this script!

% Input:
% 'layer_input' is the input data to the layer (N x Di matrix, where
% N is the number of input samples, Di is the number of dimensions 
% of your input data). 
% 'layer_param' represents the parameters of the layer
% used to transform the input data (see trainNN.m)

% Output:
% 'layer_output' represents the output data produced by the fully connected, 
% sigmoid layer (N x Do matrix, where Do is the number of output nodes of this layer)

%% YOUR CODE goes here - modify the following lines!

N = size(layer_input, 1);
Do = size( layer_param, 2);
%layer_output = zeros(N, Do);  % change this (obviously)   

if activation == 1
    layer_output = relu(layer_input*layer_param+layer_biases);
else
    layer_output = sigmoid(layer_input*layer_param+layer_biases);
end


%disp('layer parameters');
%disp(layer_param);


end
