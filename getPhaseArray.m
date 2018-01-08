function [ Tx ] = getArray( shift, maxlen )
%GETARRAY Will return the Tx structure with all the transducer geometry
%information.
%
%   This function will parse the first *.cfg* file it finds in its
%   directory, and save the transducer array to a file as well as return
%   it. If it finds the 'savedArray.mat' file, it will skip the parsing and
%   simply return that.
%
%   The `shift` input is a vector of length 3 that represents the shift
%   from HIFU to MRI space (in metres). Typically obtained from the Affine
%   matrix from the treatment head files.
%
%   `maxlen` is a single value in meters for the maximum length of the side
%   of each subelement.
%   NOTE: Each transducer element is broken up into subelements to reach
%   this size. This is indicated in the Nx2 vector `subelems`

len = 1.35e-3; %CONST length of each side of square transducer elements (Each element is approximately 1.35mm x 1.35mm surface area)

%% Error check
if nargin < 2
    error('getArray requires shift and size inputs');
elseif ~isvector(shift) || length(shift) ~= 3
    error('shift must be a vector of length 3');
elseif ~isscalar(maxlen)
    error('maxlen must be a scalar');
end


%% Returned saved array if it exists
if (exist('savedArray.mat', 'file') == 2)
    load('savedArray.mat', 'Tx');
    return;
end


%% Assume no saved array exists
% initalize structures
Tx = struct;
units = struct;

% Open file
list = dir('*.cfg*');
fileID = fopen(list(1).name,'r');

% Loop through all lines and find [Titles]
tline = fgetl(fileID); %read first line
while ischar(tline)
    
    switch tline
        case '[unitGeometry]'
            [units, tline] = readUnitGeometry(fileID, units);
        case '[arrayGeometry]'
            [array, inactives, tline] = readArrayGeometry(fileID, units);
        otherwise
            tline = fgetl(fileID); %read next line
    end
    
end


% Scale to meters and save to Tx
Tx.center = array * 1e-3;
Tx.inactives = inactives;

% Perform affine transform on each vector in Tx.center
for i=1:size(Tx.center,1)
    Tx.center(i,:) = Tx.center(i,:) + reshape(shift, size(Tx.center(i,:)));
end

% Split elements into subelements to obtain desired size
Tx.subelems = [(1:length(Tx.center))',(1:length(Tx.center))'];
while len > maxlen
    len = len/2;
    
    center_old = Tx.center;
    Tx.center = zeros(size(center_old,1)*4, 3);
    for i = 1:size(center_old,1)
        Tx.center(4*i - 3, :) = center_old(i,:) + [ len/2  len/2 0];
        Tx.center(4*i - 2, :) = center_old(i,:) + [-len/2  len/2 0];
        Tx.center(4*i - 1, :) = center_old(i,:) + [ len/2 -len/2 0];
        Tx.center(4*i    , :) = center_old(i,:) + [-len/2 -len/2 0];
    end
    
    Tx.subelems = [Tx.subelems(:,1)*4-3,Tx.subelems(:,2)*4];
end

% Assume all normals are UP
Tx.normal = repmat([0 0 1], size(Tx.center,1), 1);

% Calculate area of transducer subelements
Tx.ds = (len)^2 * ones(size(Tx.center,1), 1);
% Close the file
fclose(fileID);

end


function [units, tline] = readUnitGeometry(fileID, units)
% Read all unit geometries

tline = fgetl(fileID); %go to next line

% First, get the name of the unit
unitName = sscanf(tline,'name = %s');

%Now, parse all the coordinates
coords = zeros(0, 3); %coordinates matrix

tline = fgetl(fileID); %go to first coordinate line
C = textscan(tline,'%f, %f, %f'); %get first coordinate
while (length(C{1}) == 1 && length(C{2}) == 1 && length(C{3}) == 1) %corrected to C{3} from C{2} by Tom 21/06/2017
    % while it finds 3D coordinates
    coords(end + 1, :) = cell2mat(C); %save found coordinate

    tline = fgetl(fileID); %go to next line
    C = textscan(tline,'%f, %f, %f'); %get next coordinate
end

% Save unit geometry to structure
units.(unitName) = coords;

end



function [array, inactives, tline] = readArrayGeometry(fileID, units)

    array = zeros(0,3); %array centers
    inactives = zeros(0,1); %inactive element idices
    tline = fgetl(fileID); %read next line
    elementInd = 0;
    
    while ischar(tline) && tline(1) ~= '['
        
        % Get module centre
        if ~isempty(strfind(tline, 'module'))
            tline = fgetl(fileID); %read next line with coordinates
            
            C = textscan(tline,'%f, %f, %f, %f, %f, %f, %f, %f, %s');
            unitName = cell2mat(C{end});
            coords = [C{1:3}]; %centre coordinates of module
            
            if ~isfield(units, unitName)
                error('A module referenced a unit geometry "%s" which was not found in file.\n', unitName);
            end
            
            elementInd = size(array,1) + 1; %get the index of the 1st of the newly added elements
            array = cat(1, array, units.(unitName) + coords); %appen element to array
        
        % Get the incactive element list
        elseif ~isempty(strfind(tline, 'Inactive'))
            tline = fgetl(fileID); %read next line with element list
            
            C = textscan(tline, '%f', 'delimiter', ','); %incides in file are index 0
            inactives = cat(1, inactives, cell2mat(C) + elementInd);
        end
        
        tline = fgetl(fileID); %read next line
    end
       
end