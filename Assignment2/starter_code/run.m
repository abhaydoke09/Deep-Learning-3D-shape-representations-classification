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

% For extra credit. Following line will select relu instead of sigmoid for
% all layers except the last one.
%options.activation = 'relu';  
%X = X([1,size(X,1)],:);
%Y = Y([1,size(Y,1)],:);
model = trainNN(X, Y);

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
Ypred = testNN(model, Xtest);
err = sum( Ypred ~= Ytest ) / length(Ytest);
fprintf('Test error is %.2f%%\n', 100 * err);

%% check results on a 'real' image
img = 'star_trek1.pgm';
threshold = 0.5;
%  your model outputs a continuous value 0...1 through the topmost sigmoid
% function. You can threshold it to decide whether you have eye or not
% (it does not need to be 0.5, but something larger depending on the desired
% true positive rate - explain why in your report)
tryEyeDetector(model, img, sizeIm, threshold);

% note: if you don't have the image processing toolbox, download the
% corners.zip file and run instead the following lines:
% corners_mat_filename = 'star_trek1_corners.mat';
% tryEyeDetector(model, img, sizeIm, threshold, corners_mat_filename);

