function model = trainNN(X, Y, options)
% code for learning a neural network
% you need to complete this script!

% Input:
% 'X' is your input training data represented as a N x D matrix,
% where N is the number of training examples and D is
% the number of dimensions (number of pixels)

% 'Y' represent the training labels (0 or 1 / non-eye or eye respectively)
% and is a N x 1 vector

% 'options' is an optional structure defining:
% 'options.num_hidden_nodes_per_layer' stores the desired number of nodes per hidden layer
% 'options.iterations' stores the number of maximum training iterations to use
% 'options.learning_rate' stores the desired step size for gradient descent
% 'options.momentum' stores the desired momentum parameter for accelerating gradient descent
% 'options.weight_decay' stores the desired L2-norm regularization parameter

% Output:
% a matlab structure storing:
% 'model.num_nodes' stores number of nodes per layer (already computed for you, see below)
% 'model.num_layers' stores number of nodes per layer (already computed for you, see below)
% 'model.param' stores the weights per layer that your code needs to learn
% 'model.outputs' stores the output per layer that your code needs to compute for the training data

% feel free to modify the input and output arguments if necessary

%% initialize default options
if ~exist('options', 'var')
    options = [];
end
if ~isfield(options, 'num_hidden_nodes_per_layer')
    options.num_hidden_nodes_per_layer = [16 8 4];
end
if ~isfield(options, 'iterations')
    %options.iterations = 1000;
    options.iterations = 1;
end
if ~isfield(options, 'initial_learning_rate')
    options.learning_rate =  .5;
end
if ~isfield(options, 'momentum')
    options.momentum =  .9;
end
if ~isfield(options, 'weight_decay')
    options.weight_decay =  .001;
end
if ~isfield(options, 'activation')
    options.activation =  2;
    disp('Activation is different');
end


%% initialize model
disp('Input Size');
N = size(X, 1); % number of training sample images
D = size(X, 2); % number of input features (input nodes) (in this assignment: #pixels)
M = size(Y, 2); % number of outputs - for this assignment, we just have a single classification output: prob(eye) 

model.num_nodes = [D options.num_hidden_nodes_per_layer M]; % including input & output nodes
model.num_layers = length( model.num_nodes ); % number of layers, including input and output classification layer
fprintf('\n Number of layers (including input and output layer): %d', model.num_layers );
fprintf('\n Number of nodes in each layer will be: %s', num2str( model.num_nodes ) );
fprintf('\n Will run for %d iterations', options.iterations );
fprintf('\n Learning rate: %f, Momentum: %f , Weight decay: \n', options.learning_rate, options.momentum, options.weight_decay );

% initialize model parameters
for layer_id=2:model.num_layers    
    model.param{layer_id} = .1 * randn( model.num_nodes(layer_id-1),  model.num_nodes(layer_id) ); % plus one unit (+1) for bias
    % I am handling biases separetely. This helps for regularization loss
    % as well as the back propagation
    model.biases{layer_id} = 0.01.*ones(1,model.num_nodes(layer_id));
end

% normalize (standardize) input, store mean/std in the input layer parameters
model.param{1}.mean = mean( X );
model.param{1}.std = std( X );
X = X - repmat( model.param{1}.mean, N, 1);
X = X ./ repmat( model.param{1}.std+1e-6, N, 1); % 1e-6 helps avoid any divisions by 0


%% YOUR CODE goes here - change the following lines %%
iter = 1;

while true       
    % complete this loop for learning
    % call forwardPropagate.m, backPropagate.m appropriately, update net parameters
       
    model.outputs{1} = X; % the input layer provides the input data to the net
    
    % Check if we want to use relu
    if options.activation == 2
        for layer_id=2:model.num_layers-1
        %model.outputs{layer_id} = rand( N, model.num_nodes(layer_id) ); % change this (obviously)
        model.outputs{layer_id} = relu(model.outputs{layer_id-1}*model.param{layer_id}+model.biases{layer_id});
        end
        model.outputs{model.num_layers} = sigmoid(model.outputs{layer_id-1}*model.param{layer_id}+model.biases{layer_id});
    else
        for layer_id=2:model.num_layers
        %model.outputs{layer_id} = rand( N, model.num_nodes(layer_id) ); % change this (obviously)
        model.outputs{layer_id} = sigmoid(model.outputs{layer_id-1}*model.param{layer_id}+model.biases{layer_id});
        end
    end
        
    
    for layer_id=2:model.num_layers
        disp('Input size');
        disp(size(model.outputs{layer_id-1}));
        disp('Weight size');
        disp(size(model.param{layer_id}));
        disp('Bias size');
        disp(size(model.biases{layer_id}));
    end
    
    
    disp('Final output');
    disp(model.outputs{model.num_layers}(1:10,1));
    
    Yp = rand(N, 1) > .5; % change this (obviously)    
    cost_function = cost(Y, model); % change this (obviously)
    classification_error = sum( Y ~= Yp ) / N;
    fprintf('Iteration %d, Cost function: %f, classification error: %f %%\n', iter, cost_function, classification_error * 100);
    
    iter = iter + 1;
    if iter > options.iterations
        break;
    end
end

end