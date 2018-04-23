function mesh = laplacianSmoothing(mesh, iterations, lambda)
% input: mesh structure (mesh.V has the vertices, mesh.F has the faces)
% parameters: iterations - number of iterations for laplacian smoothing
%             lambda - parameter controlling the smoothing strength 
% output mesh structure with updated mesh.V

if ~isfield(mesh,'Adj')
    mesh = adjacencyMatrix(mesh); % represent mesh graph with adjacency matrix
    % you can now get the vertices adjacent to vertex i by calling:
    % adjV = find( mesh.Adj(i,:) );
end

%%%%%%%%%%%%%%%%%%%%%%%%
% fill code here 
for iter=1:iterations
    temp_V = zeros(size(mesh.V));
    for i = 1:size(mesh.V,2)
        adjV = find( mesh.Adj(i,:));
        neighbour_count = size(adjV,2);
        for adj_index = 1:size(adjV,2)
            temp_V(:,i) = temp_V(:,i) + mesh.V(:,adjV(adj_index));
        end
        temp_V(:,i) = temp_V(:,i)./neighbour_count;
    end
    mesh.V = mesh.V + lambda.*temp_V;
    clf;
    plotMesh(mesh,'solidbw');
    drawnow;
end
%%%%%%%%%%%%%%%%%%%%%%%%

% code, code, code...
% at each iteration you can see the smoothed mesh by
% inserting the following lines of code in your loop:
%
% clf;
% plotMesh(mesh,'solidbw');
% drawnow;

%%%%%%%%%%%%%%%%%%%%%%%%


