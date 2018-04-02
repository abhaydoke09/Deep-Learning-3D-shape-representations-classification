function [net, info] = trainMVShapeClassifier(dataset_path, matconvnet_path)
% this function trains a multi-view convnet for 3D shape classification
% YOU HAVE TO MODIFY THIS FUNCTION!

% dataset_path is the name of the folder that contains PNG images of
% rendered 3D shapes. The folder should have the following structure:
%   => category_1 [folder]
%       => shape1_id [folder]
%           => shape1_id_001.png [grayscale image]
%           => shape1_id_002.png [grayscale image]
%              ...
%       => shape2_id [folder]
%           => shape2_id_001.png [grayscale image]
%           => shape2_id_002.png [grayscale image]
%              ...
%       => ...
%   => category_2 [folder]
%      ...
%
% matconvnet_path is the path to the place you installed matconvnet (its
% root folder)
%
% the function should return a trained convnet stored in the matconvnet 
% 'simplenn' format. 
% it also returnes an 'info' data structure which stores various stats
% on the training dataset
%
%  you may execute the function as follows:
% [net, info] = trainMVShapeClassifier('dataset/train/', 'matconvnet');


%% setup matconvnet (assumes you installed matconvnet in <matconvnet_path>) 
addpath( sprintf('%s/matlab', matconvnet_path) );
vl_setupnn;

%% get stats on the training data
num_shapes = 0;
num_images = 0;
category_dirs = dir( dataset_path ); % will return folders corresponding to categories, including the default directories '.' and '..'
category_dirs = category_dirs(~ismember({category_dirs.name},{'.','..'})); % skip the default directories
info.category_names = { category_dirs(:).name }; % convert the struct into names
info.num_views = 0;
num_categories = length(info.category_names);
shape_names = cell( length( num_categories ), 1 );

for c=1:length(info.category_names)
    category_full_dir = sprintf('%s/%s', dataset_path, info.category_names{c} );
    shape_dirs = dir( category_full_dir );
    shape_dirs = shape_dirs(~ismember({shape_dirs.name},{'.','..'})); % skip the default directories
    shape_names{c} = {shape_dirs(:).name};
    num_shapes = num_shapes + length(shape_names{c});
    for s=1:length(shape_names{c})
        num_views = length( dir( sprintf( '%s/%s/%s/%s*.png', dataset_path, info.category_names{c}, shape_names{c}{s}, shape_names{c}{s} ) ) );
        info.num_views = max( num_views, info.num_views);
        num_images = num_images + num_views;
    end
end
fprintf('Found %d categories, %d rendered shapes, %d total images, %d (max) num views per shape\n', num_categories, num_shapes, num_images, info.num_views);

% %% prepare the training image database
% imdb.images.data = [];
% imdb.images.labels = zeros( 1, num_images, 'single' );
% imdb.images.set = zeros( 1, num_images, 'single' ); 
% image_id = 0;
% shape_id = 0;
% for c=1:num_categories
%     for s=1:length(shape_names{c})
%         shape_id = shape_id + 1;
%         fprintf('Loading shape data %d/%d: %s\\%s \n', shape_id, num_shapes, info.category_names{c}, shape_names{c}{s} );                
%         num_views = length( dir( sprintf( '%s/%s/%s/%s*.png', dataset_path, info.category_names{c}, shape_names{c}{s}, shape_names{c}{s} ) ) );
%         for v=1:num_views % we assume that images of shapes have filenames in this format: shape1_id_001.png, shape1_id_002.png ...
%             image_id = image_id + 1;            
%             image_full_filename = sprintf('%s/%s/%s/%s_%03d.png', dataset_path, info.category_names{c}, shape_names{c}{s}, shape_names{c}{s}, v );
%             fprintf(' => Loading image: %s \n', image_full_filename);
%             im = single( imread( image_full_filename ) ) / 255;
%             if isempty( imdb.images.data )
%                 % assumes all images have the same size
%                 imdb.images.data = zeros( size(im, 1), size(im, 2), 1, num_images, 'single' );
%             end
%             imdb.images.data(:, :, 1, image_id) = im; % convert range [0, 255]=>[0,1]
%             imdb.images.labels(image_id) = c;
%             if s < 0.9*length(shape_names{c}) % use 90% for training, 10% for validation
%                 imdb.images.set(image_id) = 1; % '1' means training
%             else
%                 imdb.images.set(image_id) = 2; % '2' means validation
%             end            
%         end        
%     end
% end
% 
% imdb.images.data_mean = mean(imdb.images.data(:,:,:,:), 4); % computes image mean
% info.data_mean = imdb.images.data_mean;
% imdb.images.data = bsxfun(@minus, imdb.images.data, imdb.images.data_mean); % subtracts image mean
% end
% % if you want to avoid reloading the training data each time you run the code,
% % you may use the save command below to save all the variables so far,
% % then at your next run, comment all the code above (including the save command), 
% % and use the load command below
%save training_data; 
load training_data;


%% general net options
% feel free to play with these parameters
rng('default'); % use default random generator
opts.gpus = []; % use only CPU for these experiments, specify your GPU id e.g., [0] if you want to use your GPU (given enough mem)
opts.continue = false;
opts.plotStatistics = true;
opts.learningRate = 0.001; 
opts.numEpochs = 20;
opts.weightDecay = 0.0001;
opts.momentum = 0.9;
opts.batchSize = info.num_views;
opts.errorFunction = 'multiclass';

%% network definition
net.meta.inputSize = [size(imdb.images.data, 1) size(imdb.images.data, 2) size(imdb.images.data, 3)] ;
net.layers = {} ;
% net.layers{end+1} = struct('type', 'conv', ...
%     'weights', {{0.05*randn(112,112,1,num_categories, 'single'), zeros(1, num_categories, 'single')}}, ...
%     'stride', 1, ...
%     'pad', 0) ;
net.layers{end+1} = struct('type', 'conv', ...
                           'weights', {{0.01*randn(8,8,1,16, 'single'), zeros(1, 16, 'single')}}, ...
                           'stride', 2, ...
                           'pad', 0);
net.layers{end+1} = struct('type', 'relu','leak',0.1) ;
net.layers{end+1} = struct('type', 'pool', ...
                           'method', 'max', ...
                           'pool', [2 2], ...
                           'stride', 2, ...
                           'pad', 0) ;
net.layers{end+1} = struct('type', 'conv', ...
                           'weights', {{0.01*randn(4,4,16,32, 'single'), zeros(1, 32, 'single')}}, ...
                           'stride', 2, ...
                           'pad', 0);
net.layers{end+1} = struct('type', 'relu','leak',0.1) ;
net.layers{end+1} = struct('type', 'pool', ...
                           'method', 'max', ...
                           'pool', [2 2], ...
                           'stride', 2, ...
                           'pad', 0) ;
net.layers{end+1} = struct('type', 'conv', ...
                           'weights', {{0.01*randn(6,6,32,num_categories, 'single'), zeros(1, num_categories, 'single')}}, ...
                           'stride', 1, ...
                           'pad', 0);                     
%%%% COMPLETE THE NETWORK ACCORDING TO GIVEN SPECS!!! %%%%
net.layers{end+1} = struct('type', 'softmaxloss') ; 

batchNormalization = false;
if batchNormalization
  net = insertBnorm(net, 1) ;
  net = insertBnorm(net, 5) ;
end
%% call matconvnet's training functions
net = vl_simplenn_tidy(net); % initialize the net parameters
net = cnn_train(net, imdb, @(x,y) getSimpleNNBatch(x,y), opts); % train the net

end

% --------------------------------------------------------------------
function [images, labels] = getSimpleNNBatch(imdb, batch)
% --------------------------------------------------------------------
images = imdb.images.data(:, :, :, batch) ;
labels = imdb.images.labels(1, batch) ; 
end

% --------------------------------------------------------------------
function net = insertBnorm(net, l)
% --------------------------------------------------------------------
ndim = size(net.layers{l}.weights{1}, 4);
layer = struct('type', 'bnorm', ...
               'weights', {{ones(ndim, 1, 'single'), zeros(ndim, 1, 'single')}}, ...
               'learningRate', [1 1 0.05], ...
               'weightDecay', [0 0]) ;
net.layers{l}.biases = [] ;
net.layers = horzcat(net.layers(1:l), layer, net.layers(l+1:end)) ;
end
