function plotMesh(mesh, style, az, el)

% mesh.F = mesh.f;
% mesh.V = mesh.v;
% plotMesh(mesh)
% plotMesh(mesh, style)
%
% style     solid | solidbw

if nargin < 2
    style = 'solidbws';
end
    
if strcmpi(style, 'solid')            
    if ~exist('OCTAVE_VERSION')
      h = trimesh(mesh.F',mesh.V(1,:)',mesh.V(2,:)',mesh.V(3,:)', 'FaceColor', 'w', 'EdgeColor', 'none', 'AmbientStrength', 0.2, 'FaceLighting', 'phong', 'SpecularStrength', 1.0, 'SpecularExponent', 100);
      if isfield(mesh, 'Nv')
         set(h, 'VertexNormals', -mesh.Nv(1:3,:)'); 
      end
      set(gcf, 'Color', 'w', 'Renderer', 'OpenGL');
      set(gca, 'Projection', 'perspective');    
      axis equal;    
      axis off;
      if nargin >= 4 view(az,el); else view(3); end
      camlight;
    else                
      h = trimesh(mesh.F',mesh.V(1,:)',mesh.V(2,:)',mesh.V(3,:)', 'FaceColor', 'b', 'EdgeColor', 'none', 'AmbientStrength', 0.2, 'FaceLighting', 'phong', 'SpecularStrength', 1.0, 'SpecularExponent', 100);
      axis off;
    end
    
elseif strcmpi(style, 'solidbw')                
    if ~exist('OCTAVE_VERSION')
        h = trimesh(mesh.F',mesh.V(1,:)',mesh.V(2,:)',mesh.V(3,:)', 'FaceColor', 'w', 'EdgeColor','k','AmbientStrength', 0.7, 'FaceLighting', 'flat', 'EdgeLighting', 'none');
	set(gcf, 'Color', 'w', 'Renderer', 'OpenGL');
    	set(gca, 'Projection', 'perspective');    
	axis equal;    
	axis off;
	if nargin >= 4 view(az,el); else view(3); end
        camlight;
    else         
        h = trimesh(mesh.F',mesh.V(1,:)',mesh.V(2,:)',mesh.V(3,:)', 'EdgeColor','k','AmbientStrength', 0.7, 'FaceLighting', 'flat', 'EdgeLighting', 'none');       
        axis off;
    end   

elseif strcmpi(style, 'solidbws')

    if ~exist('OCTAVE_VERSION')    
      h = trimesh(mesh.F',mesh.V(1,:)',mesh.V(2,:)',mesh.V(3,:)', 'FaceColor', 'w', 'EdgeColor', 'k', ...
        'AmbientStrength', 0.7, 'FaceLighting', 'phong', 'EdgeLighting', 'none');        
      set(gcf, 'Color', 'w', 'Renderer', 'OpenGL');
      set(gca, 'Projection', 'perspective');
      axis equal;    
      axis off;
      if nargin >= 4 view(az,el); else view(3); end
    else
        h = trimesh(mesh.F',mesh.V(1,:)',mesh.V(2,:)',mesh.V(3,:)',  'EdgeColor', 'k', 'AmbientStrength', 0.7, 'FaceLighting', 'phong', 'EdgeLighting', 'none');        
      axis off
    end
end



