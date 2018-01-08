% This is a quick function for using the position, velocity, and normal of
% a phantom part to generate visualisation data of the velocity distribution.

function VelVisualise(phantom, velstring, firsttime)

    if firsttime == 1
        figure;
        firsttime = 0;
        vels = abs(phantom.vels_T_l);
    else
        switch velstring
            case 'vels_T_l'
                vels = abs(phantom.vels_T_l);
            case 'vels_R_l'
                vels = abs(phantom.vels_R_l);
            case 'vels_T_s'
                if ~isequal(phantom.vels_T_s, [])
                    vels = abs(phantom.vels_T_s);
                else
                    vels = abs(phantom.vels_T_l);
                end
            case 'vels_R_s'
                vels = abs(phantom.vels_R_s);
            otherwise
                disp('You did not specify an appropriate velocity type.');
                return
        end
    end
            
    quiver3(phantom.center(:,1),phantom.center(:,2),phantom.center(:,3),...
       vels.*phantom.normal(:,1),vels.*phantom.normal(:,2), vels.*phantom.normal(:,3));
   
    hold on;
    daspect([1,1,1]);
   
    if isfield(phantom, 'children')
        for i = 1:length(phantom.children)
            VelVisualise(phantom.children{i},velstring,firsttime);
        end
    end

end
    