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
options = optimset('FinDiffType', 'central', 'DerivativeCheck', 'off', 'GradObj','on', 'LargeScale','off', 'MaxIter', iterations,'TolFun', 1e-12, 'TolX', 1e-12, 'Display', 'iter-detailed');
options.MaxFunctionEvaluations = 1000000;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%% For Laplacian set this to true %%%%%
laplacian_flag = false;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if laplacian_flag
    V = fminunc(@(V) costFunctionLaplacian(V, mesh, W), mesh.V(:), options);
else
    V = fminunc(@(V) costFunction(V, mesh, W), mesh.V(:), options);
end

mesh.V = reshape(V, 3, size(mesh.V, 2));


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


function [E, G] = costFunction(V, mesh, W)
% input: V is a vector that stores the positions of vertices
%        mesh is your mesh structure
%        W covariance matrix of the unary potential
mesh.V = reshape(V, 3, size(mesh.V, 2));

clf;
plotMesh(mesh,'solidbw');
drawnow;


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
for i = 1:size(mesh.V,2)
     unary_potentials = unary_potentials + (mesh.originalV(:,i) - mesh.V(:,i))'*W*(mesh.originalV(:,i) - mesh.V(:,i));
end
% unary_potentials = sum(sum((mesh.originalV - mesh.V)'*W*(mesh.originalV - mesh.V)));

pairwise_potentials = 0.0;
for i = 1:size(mesh.adjF,1)
    k = mesh.adjF(i,1);
    j = mesh.adjF(i,2);
    pairwise_potentials = pairwise_potentials + sum(((mesh.Nf(:,k) - mesh.Nf(:,j)).^2));
end

E = unary_potentials + 1.0 * pairwise_potentials;   %lambda = 1.0

%Unary gradients
unary_grads = -2*W*(mesh.originalV - mesh.V);

%Pairwise gradients
pairwise_grads = zeros(3,100);
for i = 1:size(mesh.V,2)
    grads = zeros(3,1);
    for adj = 1:size(mesh.adjF,1)
         p = mesh.adjF(adj,1);
         m = mesh.adjF(adj,2);
         grad_npx = nxi(mesh, p, i);
         grad_nmx = nxi(mesh, m, i);
         disp(2 * (grad_npx - grad_nmx)' * (mesh.Nf(:,p) - mesh.Nf(:,m)));
         grads = grads + 2 * (grad_npx - grad_nmx)' * (mesh.Nf(:,p) - mesh.Nf(:,m));
    end
    pairwise_grads(:,i) = grads;
end

G = unary_grads + pairwise_grads;

%%%%%%%%%%%%%%%%%%%%%%%






function [E, G] = costFunctionLaplacian(V, mesh, W)
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
    pairwise_potentials = pairwise_potentials + sum(((mesh.Nf(:,k) - mesh.Nf(:,j)).^2))^0.5;
end

E = unary_potentials + 1.0 * pairwise_potentials;   %lambda = 1.0

%Unary gradients
unary_grads = -2*W*(mesh.originalV - mesh.V);

%Pairwise gradients
pairwise_grads = zeros(3,100);
for i = 1:size(mesh.V,2)
    grads = zeros(3,1);
    for adj = 1:size(mesh.adjF,1)
         p = mesh.adjF(i,1);
         m = mesh.adjF(i,2);
         grad_npx = nxi(mesh, p, i);
         grad_nmx = nxi(mesh, m, i);
         grads = grads + (grad_npx - grad_nmx)' * (mesh.Nf(:,p) - mesh.Nf(:,m))/vecnorm(mesh.Nf(:,p) - mesh.Nf(:,m));
    end
    pairwise_grads(:,i) = grads;
end

G = unary_grads + pairwise_grads;

mesh.V = mesh.V - 0.001 * G;

% code, code, code...
%%%%%%%%%%%%%%%%%%%%%%%%










