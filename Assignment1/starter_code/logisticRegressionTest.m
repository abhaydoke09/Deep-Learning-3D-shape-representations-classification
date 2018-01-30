function [t,test_err]  = logisticRegressionTest(test_dir, w, min_y, max_y)
% complete this function to test a logistic regression
% classifier on a specified dataset

% input:
% test_dir is the path to a directory containing meshes
% in OBJ format. The directory must also contain a 
% ground_truth_labels.txt file that contains labels 
% (category id) for each mesh (necessary here to compute 
% test error)
% w are the classifier parameters learned by logisticRegressionTrain
% min_y, max_y are used to compute the range of the histogram
% for the shape descriptor (produced by logisticRegressionTrain)

% output:
% t: a row vector storing the probability of table (category '1') per test mesh
% test_err: test error (fraction of shapes whose label was mispredicted)
% 
addpath('mesh_toolbox');

% open ground_truth_labels.txt
[ground_truth_labels_file, errmsg] = fopen( sprintf('%s/ground_truth_labels.txt', test_dir ), 'rt' );
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

% read the training meshes, move meshes such that their centroid is
% at (0, 0, 0), scale meshes such that average vertical distance to
% mesh centroid is one.
meshes = {}; % a cell array storing all meshes
N = length( shape_filenames ); % number of training meshes
for n=1:N
    meshes{n} = loadMesh( sprintf('%s/%s', test_dir, shape_filenames{n} ) ); %#ok<*AGROW> %=suppress matlab editor warnings
    number_of_mesh_vertices = size( meshes{n}.V, 2 ); % number of mesh vertices
    mesh_centroid = mean(meshes{n}.V, 2); % compute mesh centroid
    meshes{n}.V = meshes{n}.V - repmat(mesh_centroid, 1, number_of_mesh_vertices ); % center mesh at origin
    average_distance_to_centroid_along_y = mean( abs( meshes{n}.V(2, :) ) ); %  average vertical distance to centroid
    meshes{n}.V = meshes{n}.V / average_distance_to_centroid_along_y; % scale meshes
end

% % if you want to avoid reloading meshes each time you run the code,
% % you may use the save command below to save all the variables so far,
% % then at your next run, comment all the code above (including the save command), 
% % and use the load command below
% save tmp_test meshes N shape_labels; 
% load tmp_test;

% this loop calls your histogram computation code!
% all shape descriptors are organized into a NxD matrix
% N training meshes, D dimensions (number of bins in your histogram)
number_of_bins = length(w) - 1; %the parameter vector should be equal to number_of_bins+1
X = zeros(N, number_of_bins);
for n=1:N
    meshes{n} = computeShapeHistogram( meshes{n}, min_y, max_y, number_of_bins );
    X(n, :) = meshes{n}.H;
end

t = rand(N, 1) > .5; % your predictions (change this!), report t values in your report!!!
test_err = 1;        % recompute this based on your predictions, report it in your report!!!

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ADD CODE HERE TO TEST CLASSIFIER
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


fprintf('Test classification error: %.2f%%\n', test_err*100);



