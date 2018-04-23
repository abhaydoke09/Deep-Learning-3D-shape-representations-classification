function mesh = addNormalNoise(mesh, variance)
% for torus, I used variance = 0.5
% for cube, I used variance = 0.5

mesh = normals(mesh);

% normal noise
for i=1:size(mesh.V, 2)
    mesh.V(:, i) = mesh.V(:, i) + mesh.Nv(:, i) * variance * randn( 1, 1);
end

% and a bit random noise in all directions
for i=1:size(mesh.V, 2)
    mesh.V(:, i) = mesh.V(:, i) + 0.25 * variance * randn( 3, 1);
end
