%% load training dataset
load trainSet
X = [eyeIm nonIm]'; % input image data represented as matrix of size #training images x #dimensions (=#pixels)
Y = [ones( size(eyeIm, 2), 1 );  zeros( size(nonIm, 2), 1 ) ]; % binary labels (eye/non-eye) represented as a vector (#train images]x1)

%% want to see the images? Uncomment the following lines
% figure
% imshow( reshape( X(1, :)/255, [25 20]) );      % 'positive' example
% figure
% imshow( reshape( X(3000, :)/255, [25 20]) );   % 'negative' example

% normalize each image according to the mean and standard deviation of its pixel intensities
% this makes our classifier more robust to brightness/constrast changes
for i=1:size(X,1)
    X(i, :) = X(i, :) - mean( X(i, :) );
    X(i, :) = X(i, :) / std( X(i, :) + 1e-6 );    
end

%% perform neural network training - you need to complete the trainNN.m script!

% For extra credit. Following line will select type of activation to be
% used. 1 for Relu and 2 for sigmoid.

% For using sigmoid activation
options.activation = 2;  

% For using relu activation
%options.activation = 1;  

% To use ensemble averaging set this varieble to true
use_ensemble = false;


if use_ensemble
    %###################Ensemble method#################
    % I am creating 5 different neural networks with different
    % random initializations.
    
    options.num_hidden_nodes_per_layer = [32 16 8];
    model1 = trainNN(X,Y,options);
    options.num_hidden_nodes_per_layer = [32 16 8];
    model2 = trainNN(X,Y,options);
    options.num_hidden_nodes_per_layer = [32 16 8];
    model3 = trainNN(X,Y,options);
    options.num_hidden_nodes_per_layer = [32 16 8];
    model4 = trainNN(X,Y,options);
    options.num_hidden_nodes_per_layer = [32 16 8];
    model5 = trainNN(X,Y,options);
    %###################################################
else
    model = trainNN(X, Y, options);
end

%% load the test dataset
load testSet
Xtest = [testEyeIm testNonIm]'; % [#test images] x [#dimensions]
Ytest = [ones( size(testEyeIm, 2), 1 );  zeros( size(testNonIm, 2), 1 ) ]; % [#test images] x [1]

% normalize each image as above
for i=1:size(Xtest,1)
    Xtest(i, :) = Xtest(i, :) - mean( Xtest(i, :) );
    Xtest(i, :) = Xtest(i, :) / std( Xtest(i, :) + 1e-6 );    
end

%% evaluate your network - you need to complete the testNN.m script!

if use_ensemble
    Ypred1 = testNN(model1, Xtest, options.activation);
    Ypred2 = testNN(model2, Xtest, options.activation);
    Ypred3 = testNN(model3, Xtest, options.activation);
    Ypred4 = testNN(model4, Xtest, options.activation);
    Ypred5 = testNN(model5, Xtest, options.activation);
    YPred = Ypred1+Ypred2+Ypred3+Ypred4+Ypred5;
    YPred(YPred<4) = 0;
    YPred(YPred>0) = 1;
    err = sum( YPred ~= Ytest ) / length(Ytest);
    fprintf('Test error is %.2f%%\n', 100 * err);
    
else
    [Ypred,model] = testNN(model, Xtest, options.activation);
    [X,Y,T,AUC] = perfcurve(Ytest,model.outputs{model.num_layers},1);
    
    true_positives = zeros(200,1);
    false_positives = zeros(200,1);
    thresholds = linspace(0,1,200);
    positive_indices = Ypred(Ypred==1);
    negative_indices = Ypred(Ypred==0);
    for i=1:200
        predictions = Ypred;
        predictions(predictions < thresholds(i)) = 0;
        predictions(predictions > 0) = 1;
        true_positives(i) = sum(predictions(positive_indices))/size(Ypred,1);
        false_positives(i) = sum(predictions(negative_indices))/size(Ypred,1);
    end
    
    %plot(X,Y);
    plot(false_positives, true_positives);
    title('ROC Curve');
    xlabel('False Positive Rate');
    ylabel('True Positive Rate');
    err = sum( Ypred ~= Ytest ) / length(Ytest);
    disp(size(Ytest,1));
    disp(sum( Ypred ~= Ytest ));
    fprintf('Test error is %.2f%%\n', 100 * err);
end

%% check results on a 'real' image
img = 'star_trek1.pgm';
if img == 'star_trek1.pgm'
    threshold = 0.99999;
else
    threshold = 0.998;
end
%  your model outputs a continuous value 0...1 through the topmost sigmoid
% function. You can threshold it to decide whether you have eye or not
% (it does not need to be 0.5, but something larger depending on the desired
% true positive rate - explain why in your report)
if use_ensemble
    tryEyeDetectorWithEnsemble(model1, model2, model3, model4, model5, img, sizeIm, threshold, options.activation);
else
    tryEyeDetector(model, img, sizeIm, threshold, options.activation);
end



% note: if you don't have the image processing toolbox, download the
% corners.zip file and run instead the following lines:
% corners_mat_filename = 'star_trek1_corners.mat';
% tryEyeDetector(model, img, sizeIm, threshold, corners_mat_filename);

