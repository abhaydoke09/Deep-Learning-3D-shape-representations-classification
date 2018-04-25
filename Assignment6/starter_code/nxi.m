function grad_npx = nxi(mesh, p, i)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
    grad_npx = zeros(3,3);
    
    vertices_p = mesh.F(:,p);
    
    if ismember(i, vertices_p) || ismember(i, vertices_p)
        if ismember(i, vertices_p)
            index = find(vertices_p == i);
            i_1 = mod(index+1,3);
            i_2 = mod(index+2,3);
            if i_1 == 0
                i_1 = 3;
            end
            if i_2 == 0
                i_2 = 3;
            end
            i_1 = vertices_p(i_1);
            i_2 = vertices_p(i_2);
            
            np = mesh.Nf(:,p);
            numerator = eye(3) - np*np';
            denominator = vecnorm(cross((mesh.V(:,i_1) - mesh.V(:,i)),(mesh.V(:,i_2) - mesh.V(:,i))));
            
            left_val = numerator./denominator;
            
            right_val = [cross([1 0 0]', (mesh.V(:,i_1) - mesh.V(:,i_2))) cross([0 1 0]', (mesh.V(:,i_1) - mesh.V(:,i_2))) cross([0 0 1]', (mesh.V(:,i_1) - mesh.V(:,i_2)))];
            
            grad_npx = left_val * right_val;
        end
        
    end
end

