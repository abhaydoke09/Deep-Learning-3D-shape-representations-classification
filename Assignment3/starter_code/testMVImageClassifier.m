function [predicted_labels, test_err] = testMVImageClassifier(dataset_path, matconvnet_path, net, info)
% this function applies a trained net to classify rendered images of 3D shapes
% YOU HAVE TO MODIFY THIS FUNCTION!
%
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
% net is the trained net returned by trainMVShapeClassifier.m
%
% info is the data structure which stores various stats on the training 
% dataset, also produced by trainMVShapeClassifier.m

% the function should return 'predicted_labels', a vector storing
% the predicted label id per image, as well as the test error 'test_err'

% for example, execute the function as follows:
% [predicted_labels, testerr] = testMVImageClassifier('dataset/test', 'matconvnet', net, info);

%% setup matconvnet (assumes you installed matconvnet in <matconvnet_path>) 
addpath( sprintf('%s/matlab', matconvnet_path) );
vl_setupnn;

%% crop last layer (loss layer) from net
net_no_loss = net; 
net_no_loss.layers(end) = [];

%% evaluate net on our test data
num_categories = length(info.category_names);
predicted_labels_mean_pooling = [];
predicted_labels_max_pooling = [];
test_err_mean_pooling = 0;
test_err_max_pooling = 0;
image_id = 0;
shape_id = 0;


for c=1:num_categories % for each category
    category_full_dir = sprintf('%s/%s', dataset_path, info.category_names{c} );
    shape_dirs = dir( category_full_dir );
    shape_dirs = shape_dirs(~ismember({shape_dirs.name},{'.','..','.DS_Store'})); % skip the default directories
    shape_names{c} = {shape_dirs(:).name};
    
    prediction_results = zeros(length(shape_names{c}), 12);
    for s=1:length(shape_names{c}) % for each shape
        shape_id = shape_id + 1;
        
        %fprintf('=>Loading shape data: %s\\%s\n', info.category_names{c}, shape_names{c}{s} );        
        
        num_views = length( dir( sprintf( '%s/%s/%s/%s*.png', dataset_path, info.category_names{c}, shape_names{c}{s}, shape_names{c}{s} ) ) );
        
        mean_prediction_scores = zeros(10, 1);
        max_prediction_scores = zeros(10, 12);
        
        for v=1:num_views % for each view
            image_id = image_id + 1;
            image_full_filename = sprintf('%s/%s/%s/%s_%03d.png', dataset_path, info.category_names{c}, shape_names{c}{s}, shape_names{c}{s}, v );
            %fprintf(' => Loading image: %s ... ', image_full_filename);
            im = single( imread( image_full_filename ) ) / 255; 
            im = bsxfun(@minus, im, info.data_mean) ;
            res = vl_simplenn(net_no_loss, im);
            scores = squeeze(gather(res(end).x));
            
            mean_prediction_scores = mean_prediction_scores + scores;
            max_prediction_scores(:,v) = scores;
            
%             [~, predicted_label] = max(scores);
%             predicted_labels(end+1) = predicted_label;
%             if predicted_label ~= c
%                 test_err = test_err + 1;
%             end            
%             fprintf('view %d: predicted label:  %s, ground-truth label: %s\n', v, info.category_names{predicted_label}, info.category_names{c});            
        end  
        
        % ############## For Mean view-pooling ##############
        [~, predicted_label] = max(mean_prediction_scores);
        predicted_labels_mean_pooling(end+1) = predicted_label;
        if predicted_label ~= c
            test_err_mean_pooling = test_err_mean_pooling + 1;
        end  
        %#####################################################
        
        % ############## For Max view-pooling ##############
        [M,I] = max(max_prediction_scores(:));
        [I_row, I_col] = ind2sub(size(max_prediction_scores),I);
        predicted_labels_max_pooling(end+1) = I_row;
        if I_row ~= c
            test_err_max_pooling = test_err_max_pooling + 1;
        end  
        %#####################################################
        
        
    end
end

test_err_mean_pooling = test_err_mean_pooling*100 / shape_id;
fprintf('Test error Mean view-pooling: %f%%\n',  test_err_mean_pooling);
test_err_max_pooling = test_err_max_pooling*100 / shape_id;
fprintf('Test error Max view-pooling: %f%%\n',  test_err_max_pooling);











