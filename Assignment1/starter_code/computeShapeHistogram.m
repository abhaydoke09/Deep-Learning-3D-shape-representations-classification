function mesh = computeShapeHistogram(mesh, y_min, y_max, number_of_bins)
% complete this function to compute a histogram capturing the 
% distribution of surface point locations along the upright 
% axis (y-axis) in the given range [y_min, y_max] for a mesh. 
% The histogram should be normalized i.e., represent a 
% discrete probability distribution.

% input: 
% mesh structure from 'loadMesh' function
%  => mesh.V contains the 3D locations of V mesh vertices (3xV matrix)
%  => mesh.F contains the mesh faces (triangles). 
%            Each triangle contains indices to three vertices (3xF matrix)
%
% output: 
% shape histogram (a column vector)

addpath('mesh_toolbox');
mesh.H = zeros(number_of_bins, 1);
  
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ADD CODE HERE TO COMPUTE mesh.H
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% get y column
y_column = mesh.V(2,:);
% filter values in the given range
filtered_y = y_column( y_column>=y_min & y_column<=y_max);
%  create bin intervals
bin_intervals = linspace(y_min, y_max, number_of_bins+1)


for idy = 1:numel(filtered_y)
    element = filtered_y(idy);
    for idb = 2:numel(bin_intervals)
        if element < bin_intervals(idb)
            mesh.H(idb-1) = mesh.H(idb-1)+1;
            break
        end    
    end
end

mesh.H = mesh.H/sum(mesh.H);







