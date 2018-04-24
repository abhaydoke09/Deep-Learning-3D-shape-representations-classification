function mesh = surfacePriorSmoothing(mesh, iterations, omega)
% input: mesh structure (mesh.V has the vertices, mesh.F has the faces)
% parameters:iterations - number of optimization iterations
%            omega - parameter controlling deviation from original surface
% output mesh structure with updated mesh.V

% running 1-2 iterations of classical smoothing improves initialization
mesh = taubinSmoothing(mesh, 2, 0.9, -1.0);

% fminunc works with double-precision floating-point numbers
mesh.V = double(mesh.V);

% PRECISION matrix of the unary potential - do not take inv of this later.
W = eye(3) * omega;

% store the original vertex positions
mesh.originalV = mesh.V;

% need normals for this type of smoothing
mesh = normals(mesh);

% below mesh.adjF stores pairs of adjacent faces
% e.g. it contains
% [1 3]
% [1 10]
% [1 24]
% etc where {1, 3}, {1, 10}, {1, 24} are adjacent face ids
mesh = faceAdjacencyMatrix(mesh);
mesh.adjF = [];
for k=1:size(mesh.F, 2)
    adjF = find( mesh.FaceAdj(k, :) );
    for j = adjF
        mesh.adjF = [mesh.adjF; [k j]];
    end
end


% fminunc minimizes a cost function 
% for question 3, the parameter GradObj should be off (derivatives will be
% estimated with finite differences - this is slow)
% for questions 4-5, the parameter GradObj should be 'on'
% after you finish writing the gradient, check it by setting the option 
% DerivativeCheck 'on' (then after you verify your gradients, deactivate
% this option - it is slow)
options = optimset('FinDiffType', 'central', 'DerivativeCheck', 'off', 'GradObj','off', 'LargeScale','off', 'MaxIter', iterations,'TolFun', 1e-12, 'TolX', 1e-12, 'Display', 'iter-detailed');
V = fminunc(@(V) costFunction(V, mesh, W), mesh.V(:), options);
mesh.V = reshape(V, 3, size(mesh.V, 2));


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


function [E, G] = costFunction(V, mesh, W)
% input: V is a vector that stores the positions of vertices
%        mesh is your mesh structure
%        W covariance matrix of the unary potential
mesh.V = reshape(V, 3, size(mesh.V, 2));

%clf;
%plotMesh(mesh,'solidbw');
%drawnow;


%%%%%%%%%%%%%%%%%%%%%%%%
% fill code here 
% for question 3, compute E (leave G=0)
% for questions 4-5, compute E and G

%%%%%%%%%%%%%%%%%%%%%%%%

E = 0; % delete this line
G = 0; % delete this line

%Computing E
mesh = normals(mesh);
unary_potentials = 0.0;
% for i = 1:size(mesh.V,2)
%     unary_potentials = unary_potentials + (mesh.originalV(:,i) - mesh.V(:,i))'*W*(mesh.originalV(:,i) - mesh.V(:,i));
% end
unary_potentials = sum(sum((mesh.originalV - mesh.V)'*W*(mesh.originalV - mesh.V)));
pairwise_potentials = 0.0;
for i = 1:size(mesh.adjF,1)
    k = mesh.adjF(i,1);
    j = mesh.adjF(i,2);
    pairwise_potentials = pairwise_potentials + sum(((mesh.Nf(:,k) - mesh.Nf(:,j)).^2));
end
E = unary_potentials + 1.0 * pairwise_potentials;   %lambda = 1.0
% code, code, code...
%%%%%%%%%%%%%%%%%%%%%%%%
