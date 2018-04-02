function [model, options] = trainRNN(X, Y, neutral_face, options)

%% initialize default options
if ~exist('options', 'var')
    options = [];
end
if ~isfield(options, 'num_hidden_nodes')
    options.num_hidden_nodes_rnn_layer = 256;
end
if ~isfield(options, 'time_step')
    options.time_step = 25;
end
if ~isfield(options, 'iterations')
    options.iterations = 15;
end
if ~isfield(options, 'initial_learning_rate')
    options.learning_rate =  0.01;
end
if ~isfield(options, 'momentum')
    options.momentum =  0.9;
end
if ~isfield(options, 'weight_decay')
    options.weight_decay =  0.001;
end
%% initialize model
D = size(X, 2); % number of input features (in this assignment: #audio features pre-extracted for you per frame)
M = size(Y, 2); % number of outputs - for this assignment, 38 2D facial landmark (x, y) positions concatenated in a vector per frame
H = options.num_hidden_nodes_rnn_layer; % number of RNN hidden nodes

model.num_nodes = [D H M];
model.num_layers = length( model.num_nodes ); % number of layers, including input, hidden and output layers
model.weight_decay = options.weight_decay;
fprintf('\n Number of RNN layers (including input and output layer): %d', model.num_layers );
fprintf('\n Number of RNN nodes in each layer will be: %s', num2str( model.num_nodes ) );
fprintf('\n Will run for %d iterations', options.iterations );
fprintf('\n Learning rate: %f, Momentum: %f \n', options.learning_rate, options.momentum );

% Initialize model parameters
%   - Wx: Weight matrix for input-to-hidden layer connections of size (D, H)
%   - Wh: Weight matrix for hidden-to-hidden layer connections of size (H, H)
%   - Wo: Weight matrix for hidden-to-output layer connections of size (H, M)
%   - b: Biases for hidden layer of size (1, H)
%   - bo: Biases for output of size (1, M)
model.param.Wx = (2 / (D + H)) * randn( D,  H );
model.param.Wh = (2 / (H + H)) * randn( H,  H );
model.param.b = zeros( 1,  H );
model.param.Wo = (2 / (M + H)) * randn( H,  M );
model.param.bo =  zeros( 1,  M ); 

% reshape input by aggregating audio features within a small window of
% past frames (of size 'options.time_step') 
% your toy RNN will take as input the features within each small window
% produce hidden states per each window frame, and use the hidden state
% of the last window frame to output a single prediction.
% this kind of ''time-delayed'' or many-to-one RNN has the benefit 
% of considering a broader temporal context to perform better predictions. 
% There are  several other more advanced variations (LSTMs, GRUs, other RNN
% variations:  http://karpathy.github.io/2015/05/21/rnn-effectiveness/ , 
% which we don't need to consider for this assignment.
% due to this aggregation, the input will now become [N, T, D] where:
% N: number of training instances
% T: size of window 
% D: number of input features
X_train = zeros( size(X,1) - options.time_step, options.time_step, D);
for i = 1:( size(X,1) - options.time_step + 1 )
    X_train(i,:,:) = X(i:i+options.time_step-1, :);
end
N = size(X_train, 1); % total number of reshaped training frames

% Y has size N x M
Y_train = Y - repmat(neutral_face, size(X,1), 1); % subtract neutral landmark face, only consider the displacement
Y_train = Y_train(options.time_step:end, :); % cutting the first 'options.time_step' frames due to the used time delay


%% Train the model through batch gradient descent
% initialize iteration (epoch) index, batch size, and training loss placeholder for
% making a plot of loss wrt each epoch
iter = 1;
batch_size = 100;
training_loss_for_plot = zeros(1, options.iterations);

% you may want to use these variables to store current derivatives (for implementing momentum)
model.deriv.Wx = zeros(size(model.param.Wx));
model.deriv.Wh = zeros(size(model.param.Wh));
model.deriv.Wo = zeros(size(model.param.Wo));
model.deriv.b = zeros(size(model.param.b));
model.deriv.bo = zeros(size(model.param.bo));

while true   
    shuffle_idx = randperm(N); % shuffle the order of training data for each iteration
    for ib = 1:ceil(N/batch_size)
        % pick training data in batch size from shuffled index
        picked_batch_idx = shuffle_idx( ((ib-1)*batch_size+1) : min(ib*batch_size,N) );
        x = X_train(picked_batch_idx, :, :);
        y = Y_train(picked_batch_idx, :);
        
        %% YOUR CODE goes here - change the following lines %%
        
        % TODO: forward propagatation
        RNNForward(x, model); 

        % TODO:  back propagatation
        % output_msg = ?
        % RNNBackward(?);

        % TODO:  update weight matrix with momentum
        % model.param.Wx = ?
        % model.param.Wh = ?
        % model.param.b = ?
        % model.param.Wo = ?
        % model.param.bo = ?

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    end
    
    %% YOUR CODE goes here - change the following lines %%
    
    % TODO: produce preduction for each training frame and calculate loss
    % Y_output = ?
    train_loss = 0; % change this line obviously
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    %% visualize training progress (loss function wrt epoch index)
    training_loss_for_plot(iter) = train_loss;
    plot(training_loss_for_plot(1:iter), 'k.-');
    pause(0.01);
 
    fprintf('Iteration %d, Cost function: %f \n', iter, training_loss_for_plot(iter));
    
    iter = iter + 1;
    if iter > options.iterations
        break;
    end
end

end