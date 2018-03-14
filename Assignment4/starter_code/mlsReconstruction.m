function mesh = mlsReconstruction(input_point_cloud_filename)
% surface reconstruction with an implicit function f(x,y,z) representing
% MLS distance to the tangent plane of the input surface points 
% input: filename of a point cloud
% output: reconstructed mesh

% load the point cloud
data = load(input_point_cloud_filename)';
points = data(1:3, :);
normals = data(4:6, :);

% construct a 3D NxNxN grid containing the point cloud
% each grid point stores the implicit function value
% set N=16 for quick debugging, use *N=64* for reporting results
N = 64;
max_dimensions = max(points(1:3,:),[],2); % largest x, largest y, largest z coordinates among all surface points
min_dimensions = min(points(1:3,:),[],2); % smallest x, smallest y, smallest z coordinates among all surface points
bounding_box_dimensions = max_dimensions - min_dimensions; % compute the bounding box dimensions of the point cloud
grid_spacing = max( bounding_box_dimensions ) / (N-9); % each cell in the grid will have the same size
[X,Y,Z] = meshgrid( min_dimensions(1)-grid_spacing*4:grid_spacing:max_dimensions(1)+grid_spacing*4, ...
                    min_dimensions(2)-grid_spacing*4:grid_spacing:max_dimensions(2)+grid_spacing*4, ...
                    min_dimensions(3)-grid_spacing*4:grid_spacing:max_dimensions(3)+grid_spacing*4 );                
fprintf('Constructed grid with %d x %d x %d points and spacing %f\n', size(X,1), size(X,2), size(X,3), grid_spacing);
IF = zeros( size(X) ); % this is your implicit function - fill it with correct values!

% toy implicit function of a sphere - replace this code with the correct
% implicit function based on your input point cloud!!!
IF = (X-(max_dimensions(1)+min_dimensions(1))/2).^2 ...
   + (Y-(max_dimensions(2)+min_dimensions(2))/2).^2 ...
   + (Z-(max_dimensions(3)+min_dimensions(3))/2).^2 ...
   - (max( bounding_box_dimensions )/4)^2;

% idx stores the index to the nearest surface point for each grid point
% use this...
K = 20;
% idx = knnsearch( points', [X(:), Y(:), Z(:)], 'K', K ); % matlab's knnsearch
idx = knnsearch( [X(:), Y(:), Z(:)], points', K );
id_nearest = knnsearch( points', points', 1 );
sigma = 0;
total_distance = 0;
for i=1:size(points,2)
    point1 = points(:,i);
    point2 = points(:,id_nearest(i));
    total_distance = total_distance + sqrt(sum((point1 - point2) .^ 2));
end
sigma = 2.0*total_distance/size(points,2);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% write code...
id_count = 0;
for x = 1:size(X,3)
    for y = 1:size(X,2)
        for z = 1:size(X,1)
            id_count = id_count+1;
            neighbors = idx(id_count,:);
            numerator = 0;
            denominator = 0;
            for i=1:size(neighbors,2)
                phi = exp(-1*sum((points(:,idx(id_count,i)) - [X(z,y,x);Y(z,y,x);Z(z,y,x)]).^2)/(sigma^2));
                numerator = numerator+dot(normals(:, idx(id_count,i)), points(:,idx(id_count,i)) - [X(z,y,x);Y(z,y,x);Z(z,y,x)])*phi;
                denominator = denominator+phi;
            end
            IF(z,y,x) = numerator/denominator;
        end
    end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

[mesh.F, mesh.V] = isosurface(X, Y, Z, IF, 0);
mesh.F = mesh.F';
mesh.V = mesh.V';
plotMesh(mesh, 'solid');
