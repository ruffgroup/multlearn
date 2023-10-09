function DoBetaSeries(rootdir, subjects, spmdir, seedroi, Conds, MapNames)
% function BetaSeries(rootdir, subjects, spmdir, seedroi, Conds, MapNames)
% INPUT:
%  rootdir  - Path to where subjects are stored
%  subjects - List of subjects (can concatenate with brackets)
%  spmdir   - Path to folder containing SPM file
%  seedroi  - Absolute path to file containing ROI in NIFTI format
%  Conds    - List of conditions for beta series analysis
%  MapNames - Output for list of maps, one per Conds cell
%       
%   Example use:
%       BetaSeries('/data/study1/fmri/', [101 102 103],
%       'model/BetaSeriesDir/', '/data/study1/Masks/RightM1.nii',
%       {'TapLeft'}, {'TapLeft_RightM1'})
%
%       conditions to be averaged together should be placed together in a cell
%       separate correlation maps will be made for each cell
%       Conds = {'Incongruent' 'Congruent' {'Error' 'Late'}}; 
%
%       For MapNames, there should be one per Conds cell above
%       e.g., with the Conds above, MapNames = {'Incongruent_Map',
%       'Congruent_Map', 'Error_Late_Map'}
%
%       Once you have z-maps, these can be entered into a 2nd-level
%       1-sample t-test, or contrasts can be taken between z-maps and these
%       contrasts can be taken to a 1-sample t-test as well.
%
% 

if nargin < 5
 disp('Need rootdir, subjects, spmdir, seedroi, Conds, MapNames. See "help BetaSeries" for more information.')
    return
end


%Find XYZ coordinates of ROI
Y = spm_read_vols(spm_vol(seedroi),1);
indx = find(Y>0);
[x,y,z] = ind2sub(size(Y),indx);
XYZ = [x y z]';


%Find each occurrence of a trial for a given condition
%These will be stacked together in the Betas array
for i = 1:length(subjects)
    subj = num2str(subjects(i));
    disp(['Loading SPM for subject ' subj]);
    %Can change the following line of code to CD to the directory
    %containing your SPM file, if your directory structure is different
    cd([rootdir subj filesep spmdir]);
    load SPM;
    
    for cond = 1:length(Conds)
        Betas = [];
        currCond = Conds{cond};
        if ~iscell(currCond)
            currCond = {currCond};
        end
        for j = 1:length(SPM.Vbeta) 
            for k = 1:length(currCond)
                if ~isempty(strfind(SPM.Vbeta(j).descrip,[currCond{k} '_'])) 
                    Betas = strvcat(Betas,SPM.Vbeta(j).fname);
                end
            end
        end
              

    %Extract beta series time course from ROI
    %This will be correlated with every other voxel in the brain
    if ischar(Betas)
        P = spm_vol(Betas);
    end

    est = spm_get_data(P,XYZ);
    est = nanmean(est,2);

        
        
        %----Do beta series correlation between ROI and rest of brain---%

            MapName = MapNames{cond};
            disp(['Performing beta series correlation for ' MapName]);
            
            Vin = spm_vol(Betas);
            nimgo = size(Vin,1);
            nslices = Vin(1).dim(3);

            % create new header files
            Vout_r = Vin(1);   
            Vout_p = Vin(1);
            [pth,nm,xt] = fileparts(deblank(Vin(1).fname));
            Vout_r.fname = fullfile(pth,[MapNames{cond} '_r.img']);
            Vout_p.fname = fullfile(pth,[MapNames{cond} '_p.img']);

            Vout_r.descrip = ['correlation map'];
            Vout_p.descrip = ['p-value map'];

            Vout_r.dt(1) = 16;
            Vout_p.dt(1) = 16;

            Vout_r = spm_create_vol(Vout_r);
            Vout_p = spm_create_vol(Vout_p);

            % Set up large matrix for holding image info
            % Organization is time by voxels
            slices = zeros([Vin(1).dim(1:2) nimgo]);
            stack = zeros([nimgo Vin(1).dim(1)]);
            out_r = zeros(Vin(1).dim);
            out_p = zeros(Vin(1).dim);


            for i = 1:nslices
                B = spm_matrix([0 0 i]);
                %creates plane x time
                for j = 1:nimgo
                    slices(:,:,j) = spm_slice_vol(Vin(j),B,Vin(1).dim(1:2),1);
                end

                for j = 1:Vin(1).dim(2)
                    stack = reshape(slices(:,j,:),[Vin(1).dim(1) nimgo])';
                    [r p] = corr(stack,est);
                    out_r(:,j,i) = r;
                    out_p(:,j,i) = p;

                end

                Vout_r = spm_write_plane(Vout_r,out_r(:,:,i),i);
                Vout_p = spm_write_plane(Vout_p,out_p(:,:,i),i);

            end

                
            %Convert correlation maps to z-scores
            %NOTE: If uneven number of trials in conditions you are
            %comparing, leave out the "./(1/sqrt(n-3)" term in the zdata
            %variable assignment, as this can bias towards conditions which
            %have more trials
            
                disp(['Converting correlation maps for subject ' subj ', condition ' MapNames{cond} ' to z-scores'])
                P = [MapNames{cond} '_r.img'];
                n = size(Betas,1);
                Q = MapNames{cond};
                Vin = spm_vol([MapNames{cond} '_r.img']);

                % create new header files
                Vout = Vin;   

                [pth,nm,xt] = fileparts(deblank(Vin(1).fname));
                Vout.fname = fullfile(pth,[Q '_z.img']);

                Vout.descrip = ['z map'];

                Vout = spm_create_vol(Vout);

                data = spm_read_vols(Vin);
                zdata = atanh(data)./(1/sqrt(n-3));

                spm_write_vol(Vout,zdata);

    end
end