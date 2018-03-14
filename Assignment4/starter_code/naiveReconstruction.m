function mesh = naiveReconstruction(input_point_cloud_filename)
% surface reconstruction with an implicit function f(x,y,z) representing
% signed distance to the tangent plane of the surface point nearest to each 
% point (x,y,z)
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
%idx = knnsearch( points', [X(:), Y(:), Z(:)], 'K', 1 ); %matlab's knnsearxh
idx = knnsearch( [X(:), Y(:), Z(:)], points', 1 );

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% write code...
% id_count = 0;
% for x = 1:size(X,1)
%     for y = 1:size(X,2)
%         for z = 1:size(X,3)
%             id_count = id_count+1;
%             IF(x,y,z) = dot(normals(:, idx(id_count)), points(:,idx(id_count)) - [X(x,y,z);Y(x,y,z);Z(x,y,z)]);
%         end
%     end
% end

id_count = 0;
for x = 1:size(X,3)
    for y = 1:size(X,2)
        for z = 1:size(X,1)
            id_count = id_count+1;
            IF(z,y,x) = dot(normals(:, idx(id_count)), points(:,idx(id_count)) - [X(z,y,x);Y(z,y,x);Z(z,y,x)]);
        end
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% extract isosurface 
[mesh.F, mesh.V] = isosurface(X, Y, Z, IF, 0);
mesh.F = mesh.F'; % plotMesh expects 3 x #vertices/#faces
mesh.V = mesh.V';
plotMesh(mesh, 'solid');
