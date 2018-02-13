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
    options.iterations = 1000;
end
if ~isfield(options, 'initial_learning_rate')
    options.learning_rate =  0.1;
end
if ~isfield(options, 'momentum')
    options.momentum =  .9;
end
if ~isfield(options, 'weight_decay')
    options.weight_decay =  .0001;
end
if ~isfield(options, 'activation')
    options.activation =  1;
    disp('Activation is different');
else
    options.activation =  2;
end

%####################################
% Following options for extra credit
use_decay = false;
use_batch_gradient_descent = true;
use_relu = false;

if use_batch_gradient_descent
    options.learning_rate = 0.1;
    options.weight_decay = 0.0001;
%####################################


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
    %model.param{layer_id} = .1 * randn( model.num_nodes(layer_id-1),  model.num_nodes(layer_id) ); % plus one unit (+1) for bias
    model.param{layer_id} = (sqrt(2/model.num_nodes(layer_id-1))) .* randn( model.num_nodes(layer_id-1),  model.num_nodes(layer_id) ); % plus one unit (+1) for bias
    % I am handling biases separetely. This helps for regularization loss
    % as well as the back propagation
    model.biases{layer_id} = 0.1.*ones(1,model.num_nodes(layer_id));
    model.param_derivatives{layer_id} = zeros( model.num_nodes(layer_id-1),  model.num_nodes(layer_id) );
    model.bias_derivatives{layer_id} = zeros(1,model.num_nodes(layer_id));
end



% normalize (standardize) input, store mean/std in the input layer parameters
model.param{1}.mean = mean( X );
model.param{1}.std = std( X );
X = X - repmat( model.param{1}.mean, N, 1);
X = X ./ repmat( model.param{1}.std+1e-6, N, 1); % 1e-6 helps avoid any divisions by 0


%% YOUR CODE goes here - change the following lines %%
iter = 1;

for layer_id=2:model.num_layers    
    model.param_derivatives{layer_id} = model.param_derivatives{layer_id}.*0;
    model.bias_derivatives{layer_id} = model.bias_derivatives{layer_id}.*0;
end

while true       
    
    % For batch gradient descent learning, shuffle the data randomly.
    shuffle_order = randperm(size(X, 1));
    X = X(shuffle_order, :);
    Y = Y(shuffle_order, :);
    
    %If we dont want to use batch gradient descent then the batch size will
    %be N. Else we can set it to anything else
    
    batch_size = N;
    if use_batch_gradient_descent
        batch_size = 128;
    end
    
    for n = 1:batch_size:N
        X_batch = X;
        if n+batch_size > N
            X_batch = X(n:N,:);
            Y_batch = Y(n:N,:);
        else
            X_batch = X(n:n+batch_size,:);
            Y_batch = Y(n:n+batch_size,:); 
        end
    
        model.outputs{1} = X_batch;
        for layer_id=2:model.num_layers
            model.outputs{layer_id} = forwardPropagate(model.outputs{layer_id-1}, model.param{layer_id}, model.biases{layer_id}, 1);
        end

        % I am computing the messeges emitted by the last layer here itself.
        received_msg = model.outputs{model.num_layers} - Y_batch;

        if use_decay
            current_learning_rate = step_decay(options.learning_rate, 0.000, iter);
            % I am using momentum annealing technique. Momentum is initially starts with
            % 0.5 and keep on annealing until it reaches 0.9.
            % Reference = http://cs231n.github.io/neural-networks-3/#sgd
            momentum_value = 0.5 + ((0.9-0.5)/(options.iterations-1))*(iter-1);
        else
            current_learning_rate = options.learning_rate;
            momentum_value = options.momentum;
        end

        for layer_id = model.num_layers:-1:2
            %disp(sum(sum(model.param{layer_id})));
            [received_msg, param_derivatives, bias_derivatives] = backPropagate(received_msg, layer_id, model, options.weight_decay);

            model.param_derivatives{layer_id} = momentum_value.*model.param_derivatives{layer_id} + (current_learning_rate/size(X_batch,1)*1.0)*param_derivatives;
            model.param{layer_id} = model.param{layer_id} - model.param_derivatives{layer_id} - options.weight_decay*model.param{layer_id};

            model.bias_derivatives{layer_id} = momentum_value.*model.bias_derivatives{layer_id} + (current_learning_rate/size(X_batch,1)*1.0)*bias_derivatives;
            model.biases{layer_id} = model.biases{layer_id} - model.bias_derivatives{layer_id};
            %disp(sum(sum(model.param_derivatives{layer_id})));
            %disp(sum(sum(model.param{layer_id})));
        end
    end
    
    model.outputs{1} = X;
    for layer_id=2:model.num_layers
        model.outputs{layer_id} = forwardPropagate(model.outputs{layer_id-1}, model.param{layer_id}, model.biases{layer_id}, 1);
    end
    Yp = model.outputs{model.num_layers} > 0.7;
    [cost_function] = cost(Y, Yp, model, options.weight_decay);
    classification_error = sum( Y ~= Yp ) / N;
    fprintf('Iteration %d, Cost function: %f, classification error: %f %%\n', iter, cost_function, classification_error * 100);

    iter = iter + 1;
    if iter > options.iterations
        break;
    end
           
    
end

end