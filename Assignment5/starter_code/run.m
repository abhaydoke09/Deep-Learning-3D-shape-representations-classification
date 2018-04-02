% run script

%% load train/test dataset
load('starter_data.mat'); % command this line if data is loaded

%% want to see the training facial landmarks? Uncomment the following lines
% figure;
% for i = 1:size(Y_train, 1)
%     showFacialLandmarks(Y_train(i, :), 0, 'r-');
%     pause(0.01);
% end
%%  Uncomment the following line to see a single, neutral (silent) face. 
% % The net will produce displacements of landmarks relative to this neutral face
% showFacialLandmarks(neutral_face, 1.5, 'r-'); 

%% train the RNN model
% the code will run on your CPU (it might take around 10-30 min to finish training depending on your machine)
% obviously, it would be more desirable to train a deep RNN on GPUs, yet
% for the purpose of this course assignment, a small RNN is enough
tic; 
[model, options]  = trainRNN(X_train, Y_train, neutral_face);
toc; 

%% test model on test sequences (will use trained model & options)
testRNN;

