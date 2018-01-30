function [mesh, labels, labelsmap] = loadMesh(filestr, labels, labelsmap)
if nargin == 1
    labels = {};
    labelsmap = {};
end
if nargin == 2
    labelsmap = {};
end


MAX_NUMBER_OF_FACES_OR_VERTS = 10^7;

fprintf(1, '\nReading %s..\n', filestr);
file = fopen( strtrim( filestr ), 'rt');
if file == -1
    warning(['Could not open mesh file: ' filestr]);
    mesh = [];
    return;
end
mesh.filename = strtrim( filestr );

if strcmp( filestr(end-3:end), '.off')
    fgetl(file);
    line = strtrim(fgetl(file));
    [token,line] = strtok(line);
    numverts = eval(token);
    [token,line] = strtok(line);
    numfaces = eval(token);
    mesh.V = zeros( 3, numverts, 'single' );
    mesh.F = zeros( 3, numfaces, 'single' );
    
    try
        data = dlmread(filestr, ' ', 2, 0);
        data = data(1:numverts+numfaces, :);
        mesh.V(1:3, 1:numverts) = data(1:numverts, 1:3)';
        mesh.F(1:3, 1:numfaces) = data(numverts+1:numverts+numfaces, 2:4)' + 1;
    catch 
        warning('OFF file is corrupt - attempting to compensate...');
        datav = textscan(file,'%f %f %f', numverts, ...
            'Delimiter', '\n', ...
            'CollectOutput', true);
        datav = datav{1};
        dataf = textscan(file,'%d %d %d %d', numfaces, ...
            'Delimiter', '\n', ...
            'CollectOutput', true);
        dataf = dataf{1};
        mesh.V(1:3, 1:numverts) = datav(1:numverts, 1:3)';
        mesh.F(1:3, 1:numfaces) = dataf(1:numfaces, 2:4)' + 1;
    end
    
elseif strcmp( filestr(end-3:end), '.obj')
    mesh.V = zeros(3, MAX_NUMBER_OF_FACES_OR_VERTS, 'single');
    mesh.Nv = zeros(3, MAX_NUMBER_OF_FACES_OR_VERTS, 'single');
    mesh.F = zeros(3, MAX_NUMBER_OF_FACES_OR_VERTS, 'uint32');
    mesh.parts = struct;
    mesh.faceLabels = zeros(1, MAX_NUMBER_OF_FACES_OR_VERTS, 'uint32');
    v = 0;
    f = 0;
    vn = 0;
    p = 0;
    cur_label_id = 0;
    cur_f_in_part = 0;
    
    % global group
    %     p = 1;
    %     label = 'null';
    %     mesh.parts(p).name = label;
    %     mesh.parts(p).faces = zeros(1, MAX_NUMBER_OF_FACES_OR_VERTS, 'uint32');
    %     cur_f_in_part = 0;
    %     cur_label_id = 1;
    %     labelsmap{cur_label_id} = {};
    %     labelsmap{cur_label_id} = [ labelsmap{cur_label_id} {mesh.filename} ];
    
    while(~feof(file))
        line_type = fscanf(file,'%c',2);
        switch line_type(1)
            case '#'
                line = fgets(file);
            case 'v'
                if line_type(2) == 'n'
                    vn = vn + 1;
                    normal  = fscanf(file, '%f%f%f');
                    mesh.Nv(:, vn) = normal;
                elseif isspace( line_type(2) )
                    v = v + 1;
                    point = fscanf(file, '%f%f%f');
                    mesh.V(:, v) = point;
                else
                    fgets(file);
                end
            case 'f'
                face = fscanf(file, '%u%u%u');
                if isempty(face)
                    break;
                end
                f = f + 1;
                mesh.F(:, f) = face;
                cur_f_in_part = cur_f_in_part + 1;
                if p == 0
                    p = p + 1;
                    label = '__null__';
                    mesh.parts(p).name = label;
                    disp(['Read label: ' mesh.parts(p).name]);
                    mesh.parts(p).faces = zeros(1, MAX_NUMBER_OF_FACES_OR_VERTS, 'uint32');
                    [labels, cur_label_id] = searchLabels( labels, label );
                    if length( labelsmap ) < cur_label_id
                        labelsmap{cur_label_id} = {};
                    end
                    labelsmap{cur_label_id} = [ labelsmap{cur_label_id} {mesh.filename} ];
                end
                mesh.parts(p).faces(cur_f_in_part) = f;
                mesh.faceLabels(f) = cur_label_id;
            case 'g'
                if p ~= 0
                    mesh.parts(p).faces = mesh.parts(p).faces( 1:cur_f_in_part );
                end
                p = p + 1;
                label = fgetl(file);
                mesh.parts(p).name = label;
                label( (label >= '0' & label <= '9') | label == '_' | label == '-' ) = [];
                label = lower(label);
                disp(['Read label: ' mesh.parts(p).name]);
                mesh.parts(p).faces = zeros(1, MAX_NUMBER_OF_FACES_OR_VERTS, 'uint32');
                [labels, cur_label_id] = searchLabels( labels, label );
                cur_f_in_part = 0;
                if length( labelsmap ) < cur_label_id
                    labelsmap{cur_label_id} = {};
                end
                labelsmap{cur_label_id} = [ labelsmap{cur_label_id} {mesh.filename} ];
            otherwise
                if isspace(line_type(1))
                    fseek(file, -1, 'cof');
                    continue;
                end
                fprintf('last string read: %c%c %s', line_type(1), line_type(2), fgets(file));
                fclose(file);
                error('only triangular obj meshes are supported with vertices, normals, groups, and vertex normals.');
        end
    end
    if p ~= 0
        mesh.parts(p).faces = mesh.parts(p).faces( 1:cur_f_in_part );
    end
    
    mesh.V = mesh.V(:, 1:v);
    mesh.F = mesh.F(:, 1:f);
    mesh.Nv = mesh.Nv(:, 1:v);
    mesh.faceLabels  = mesh.faceLabels(1:f);
    
    %     nullfaces = find( mesh.faceLabels == 0);
    %     if ~isempty( nullfaces )
    % %        warning('faces with no label found; this can be very bad!');
    %         mesh.parts(p+1).name = '__null__';
    %         mesh.parts(p+1).labelid = 0;
    %         mesh.parts(p+1).faces = nullfaces;
    %     end
end

fclose(file);

end



function [labels,pos] = searchLabels( labels, currentlabel )
pos = strmatch(currentlabel, labels, 'exact');
if isempty( pos )
    pos = strmatch(currentlabel(1:end-1), labels, 'exact');
    if isempty( pos )
        labels{end+1} = currentlabel;
        pos = length(labels);
    else
        return;
    end
else
    return;
end
end
