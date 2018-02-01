function [w, min_y, max_y] = logisticRegressionTrain(train_dir, number_of_bins)
% complete this function to train a logistic regression
% classifier

% input:
% train_dir is the path to a directory containing meshes
% in OBJ format used for training. The directory must
% also contain a ground_truth_labels.txt file that
% contains the training labels (category id) for each mesh
% number_of_bins specifies the number of bins to use
% for your histogram-based mesh descriptor

% output:
% a row vector storing the learned classifier parameters (you must compute it)
% histogram range (this is computed for you)

addpath('mesh_toolbox');

% open ground_truth_labels.txt
[ground_truth_labels_file, errmsg] = fopen( sprintf('%s/ground_truth_labels.txt', train_dir ), 'rt' );
if ground_truth_labels_file < 0
    error(errmsg);
end

% read dataset labels
shape_filenames = {}; % a cell array of strings storing mesh filenames
shape_labels = [];    % an array storinng the mesh label ids (integers)
while( ~feof(ground_truth_labels_file) )
    shape_filenames{end+1} = fscanf(ground_truth_labels_file, '%s', 1);
    if isempty( shape_filenames{end} ) % ignore empty line
        shape_filenames(end) = [];     % remove empty entry
        continue;
    end
    shape_labels(end+1) = fscanf(ground_truth_labels_file, '%d', 1);
end
fclose( ground_truth_labels_file );

% read the training meshes, compute 'lowest' and 'highest' surface
% point across all meshes, move meshes such that their centroid is
% at (0, 0, 0), scale meshes such that average vertical distance to
% mesh centroid is one.
meshes = {}; % a cell array storing all meshes
N = length( shape_filenames ); % number of training meshes
min_y = Inf;  % smallest y-axis position in dataset
max_y = -Inf; % largest y-axis position in dataset
for n=1:N
    meshes{n} = loadMesh( sprintf('%s/%s', train_dir, shape_filenames{n} ) ); %#ok<*AGROW> %=suppress matlab editor warnings
    number_of_mesh_vertices = size( meshes{n}.V, 2 ); % number of mesh vertices
    mesh_centroid = mean(meshes{n}.V, 2); % compute mesh centroid
    meshes{n}.V = meshes{n}.V - repmat(mesh_centroid, 1, number_of_mesh_vertices ); % center mesh at origin
        average_distance_to_centroid_along_y = mean( abs( meshes{n}.V(2, :) ) ); %  average vertical distance to centroid
    meshes{n}.V = meshes{n}.V / average_distance_to_centroid_along_y; % scale meshes
    min_y = min(min_y, min( meshes{n}.V(2, :) ) );
    max_y = max(max_y, max( meshes{n}.V(2, :) ) );
end
 
% % if you want to avoid reloading meshes each time you run the code,
% % you may use the save command below to save all the variables so far,
% % then at your next run, comment all the code above (including the save command), 
% % and use the load command below
% save tmp_train meshes min_y max_y N shape_labels;
% load tmp_train;

% this loop calls your histogram computation code!
% all shape descriptors are organized into a NxD matrix
% N training meshes, D dimensions (number of bins in your histogram)
X = zeros(N, number_of_bins);
for n=1:N
    meshes{n} = computeShapeHistogram( meshes{n}, min_y, max_y, number_of_bins );
    X(n, :) = meshes{n}.H;
end

w = .1 * randn( 1, number_of_bins+1 ); % +1 for bias, random initialization

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ADD CODE HERE TO LEARN PARAMETERS w
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
bias_column = ones(N,1);
X = [bias_column X];

learning_rate = 0.1;
n_iterations = 10000;
lambda = 0.1;


for itr=1:10000
    y_predicted = 1./(1 + exp(-1*(w*X')));
%     if itr%100 == 0
%         disp(y_predicted);
%     end
    output_diff = shape_labels - y_predicted;
%     disp('Size of X and output diff');
%     disp(size(X));
%     disp(size(output_diff'));
    loss_gradients = -1.0*sum(X.*output_diff');
    regularization_loss = lambda*sign(w);
    loss_gradients(1,2:number_of_bins+1) = loss_gradients(1,2:number_of_bins+1) + regularization_loss(1,2:number_of_bins+1);
    w = w - learning_rate*loss_gradients;
end

y_predicted = 1./(1 + exp(-1*(w*X')));
y_predicted(y_predicted<0.5) = 0;
y_predicted(y_predicted>=0.5) = 1;
disp('Number of correct labels');
disp(sum(y_predicted == shape_labels));





