function [Ypred,model] = testNN(model, X, threshold, Y)
% code for applying a neural network on new data
% you need to complete this script!

% Input:
% 'model' stores the net parameters (including the weights you need here) - see trainNN.m

% 'X' is your test data represented as a N x D matrix, 
% where N is the number of test instances and D is
% the number of dimensions (number of pixels)

% 'threshold': your model outputs a continuous value 0...1 through the topmost sigmoid
% function. You can threshold it to decide whether you have eye or not
% (it does not need to be 0.5, but something larger depending on the desired
% true positive rate - see ROC curves)

% Output:
% Ypred is your prediction (1 - eye, 0 - non-eye) per training instance
% it should be a Nx1 vector

% feel free to modify the input and output arguments if necessary

if nargin == 2
    threshold = .5;
end

% normalize (standardize) input given training mean/std dev.
N = size(X, 1);
X = X - repmat( model.param{1}.mean, N, 1);
X = X ./ repmat( model.param{1}.std+1e-6, N, 1); % 1e-6 helps avoid any divisions by 0

%% YOUR CODE goes here - change the following lines %%
% call forwardPropagate appropriately
% use the output probabilities to determine the predictions (eye/not eye)
model.outputs{1} = X;
for layer_id=2:model.num_layers
    model.outputs{layer_id} = forwardPropagate(model.outputs{layer_id-1}, model.param{layer_id}, model.biases{layer_id}, 1);
end


Ypred = model.outputs{model.num_layers} > threshold;  % change this (obviously)    

end