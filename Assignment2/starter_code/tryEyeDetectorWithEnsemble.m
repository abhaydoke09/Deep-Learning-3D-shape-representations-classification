function tryEyeDetectorWithEnsemble(model1, model2, model3, model4, model5,image_filename, window_size, threshold, activation, corners_mat_filename)
% code for detecting eyes in an image with a ML classifier

% Input:
% 'model' is storing the neural net parameters (see trainNN.m)
% 'image_filename' is the file name of the input image
% 'window_size' should be the training image size: [25 20] 
% 'threshold': your model outputs a continuous value 0...1 through the topmost sigmoid
% function. You can threshold it to decide whether you have eye or not
% (it does not need to be 0.5, but something larger depending on the desired
% true positive rate - see ROC curves)
% 'corners_mat_filename' is an optional mat file storing the positions of
% corners (from a Harris corner detector) in the input image - use this
% argument if you don't have the image processing toolbox

% feel free to modify the input and output arguments if necessary

if nargin == 3
    threshold = .9998;
end

whole_image = imread(image_filename);
save_x = [];
save_y = [];

% note if you don't have the image processing toolbox, the script will
% attempt to load the corners from the given mat file.
try
    corners = corner(whole_image, 'N', 1000, 'QualityLevel', 0.001);
catch
    load( corners_mat_filename );
end
% if you want to see the corners in the image:
% imshow(whole_image);
% hold on
% plot(corners(:,1), corners(:,2), 'r*');
% hold off;
% pause

% for each corner
for c=1:size(corners, 1)    
    % search along perturbed windows (+/-5 pixels horizontally/vertically) around the corner 
    for px=-5:5
        for py=-5:5
            min_x = corners(c, 2) - ceil(window_size(1)/2) + px;
            min_y = corners(c, 1) - ceil(window_size(2)/2) + py;   
            if ( min_x + window_size(1) - 1 > size(whole_image, 1) ) || ...
            ( min_y + window_size(2) - 1 > size(whole_image, 2) ) || ...
            ( min_x <= 0) || ...
            ( min_y <= 0)
                 continue
            end
            
            % take 20x25 pixels image, size [25 20]
            subim = double( whole_image( min_x : min_x + window_size(1) - 1, min_y : min_y + window_size(2) - 1 ) );
            
            % normalize it as in training & testing
            Xtest = subim(:)';
            Xtest = Xtest - mean( Xtest );
            Xtest = Xtest / (std( Xtest ) + 1e-6);
            
            % call testNN.m - you need to complete this function
            [Y1,model1] = testNN(model1, Xtest, activation, threshold);
            [Y2,model2] = testNN(model2, Xtest, activation, threshold);
            [Y3,model3] = testNN(model3, Xtest, activation, threshold);
            [Y4,model4] = testNN(model4, Xtest, activation, threshold);
            [Y5,model5] = testNN(model5, Xtest, activation, threshold);
            
            % store window coordinates if majority of models agree
            if Y1+Y2+Y3+Y4+Y5 > 3
                save_x = [save_x min_x];
                save_y = [save_y min_y];
            end
        end
    end
end

% show image + detected eyes as green rectangles
imshow(whole_image);
hold on;
for n=1:length(save_x)
       rectangle('Position', [save_y(n) save_x(n) window_size(2) window_size(1)], 'EdgeColor', 'g', 'LineWidth', 3 );
end

hold off;