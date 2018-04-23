function mesh = taubinSmoothing(mesh, iterations, lambda, mu)
% input: mesh structure (mesh.V has the vertices, mesh.F has the faces)
% parameters: iterations - number of iterations for laplacian smoothing
%             lambda, mu - parameters controlling the smoothing/inflating strength 
%             mu should be larger than lambda, be careful with its sign
% output mesh structure with updated mesh.V

if ~isfield(mesh,'Adj')
    mesh = adjacencyMatrix(mesh); % represent mesh graph with adjacency matrix
    % you can now get the vertices adjacent to vertex i by calling:
    % adjV = find( mesh.Adj(i,:) );    
end


%%%%%%%%%%%%%%%%%%%%%%%%
% fill code here 
%%%%%%%%%%%%%%%%%%%%%%%%

% code, code, code...
% at each iteration you can see the smoothed mesh by
% inserting the following lines of code in your loop:
%
% clf;
% plotMesh(mesh,'solidbw');
% drawnow;

%%%%%%%%%%%%%%%%%%%%%%%%




