%% apply trained RNN model on 3 different test sequences

for test_idx=1:3
    
    % reshape test input into similar shape as training input - it will have size [N, T, D]
    x = zeros(size(X_test{test_idx},1)-options.time_step, options.time_step, size(X_test{test_idx},2));
    for i = 1:size(X_test{test_idx},1)-options.time_step+1
        x(i,:,:) = X_test{test_idx}(i:i+options.time_step-1,:);
    end
    y = Y_test{test_idx} - repmat(neutral_face, size(X_test{test_idx},1), 1); % subtract neutral landmark face, only consider the displacement
    y = y(options.time_step:end, :); % cutting the first 'options.time_step' frames due to the used time delay (our prediction will have a small latency!)
    
    
    %% YOUR CODE goes here - change the following lines %%    
    % produce output predictions and test loss
    [Y_output, model] = RNNForward(x, model) ; % change this line obviously
    test_loss = sum(vecnorm(Y_output - y))/size(x,1); % change this line obviously
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    fprintf('Test set %d, Average L2 loss: %f \n', test_idx, test_loss);
    
    
    %% a post-processing for temporal smoothing is useful given
    % that we used a simple, shallow RNN
    % we also visualize the prediction results here
    Y_output = Y_output + repmat(neutral_face, size(Y_output,1), 1);
    y = y + repmat(neutral_face, size(y,1), 1);
    for i = 1:size(Y_output,2)
        Y_output(:,i)=smooth(Y_output(:,i), 15);
        y(:,i)=smooth(y(:,i), 15);
    end
    
    % draw ground-truth in red on left, and your estimation in blue on right
    % saved as pred_xxx.avi, where xxx = test_idx
    CreateAvi(Y_output, y, ['pred_', int2str(test_idx),'.avi']);        
end