function func_contrast_level1_BothPmods(path, sub, model_version, del_old_con)

% path of data
data.source = [fullfile(path.folder_processed , sub)];
%path of results
data.destination = fullfile(path.SPM_folder,'/results', model_version, sub);

spm_jobman('initcfg');

load(fullfile(data.destination,'SPM.mat'));
%%
runs_present = 6;

model = split(model_version, "_");
    var_int1 = model(1,1);
    var_int2 = model(2,1);
    num_pmods1 = find(contains(SPM.xX.name,{['x' var_int1 'Tactile^' ], ['x' var_int1 'Audio^']}));
    num_pmods2 = find(contains(SPM.xX.name,{['x' var_int2 'Tactile^' ], ['x' var_int2 'Audio^']}));
    if numel(num_pmods1) + numel(num_pmods2) ~= runs_present*2
        error('Error occurred. Pmods not correct!')
    end



%% contrasts

contrasts_SPE = double(contains(SPM.xX.name,{['x' var_int1 'Tactile^' ], ['x' var_int1 'Audio^']}));
contrasts_audioSPE = double(contains(SPM.xX.name,['x' var_int1 'Audio^']));
contrasts_tactileSPE = double(contains(SPM.xX.name,['x' var_int1 'Tactile^' ]));

contrasts_RPE = double(contains(SPM.xX.name,{['x' var_int2 'Tactile^' ], ['x' var_int2 'Audio^']}));
contrasts_audioRPE = double(contains(SPM.xX.name,['x' var_int2 'Audio^']));
contrasts_tactileRPE = double(contains(SPM.xX.name,['x' var_int2 'Tactile^' ]));

contrasts_both = (contrasts_SPE + contrasts_RPE) / 2;
contrasts_audioBoth = (contrasts_audioSPE + contrasts_audioRPE) / 2;
contrasts_tactileBoth = (contrasts_tactileSPE + contrasts_tactileRPE) / 2;


%% SPE

matlabbatch{1}.spm.stats.con.spmmat = cellstr(fullfile(data.destination, 'SPM.mat'));
matlabbatch{1}.spm.stats.con.consess{1}.tcon.name = 'SPE-RPE';
matlabbatch{1}.spm.stats.con.consess{1}.tcon.weights = contrasts_SPE - contrasts_RPE;
matlabbatch{1}.spm.stats.con.consess{1}.tcon.sessrep = 'none';

matlabbatch{1}.spm.stats.con.consess{2}.tcon.name = 'RPE-SPE';
matlabbatch{1}.spm.stats.con.consess{2}.tcon.weights = contrasts_RPE - contrasts_SPE;
matlabbatch{1}.spm.stats.con.consess{2}.tcon.sessrep = 'none';

matlabbatch{1}.spm.stats.con.consess{3}.tcon.name = 'tactileSPE-tactileRPE';
matlabbatch{1}.spm.stats.con.consess{3}.tcon.weights = contrasts_tactileSPE - contrasts_tactileRPE;
matlabbatch{1}.spm.stats.con.consess{3}.tcon.sessrep = 'none';

matlabbatch{1}.spm.stats.con.consess{4}.tcon.name = 'tactileRPE-tactileSPE';
matlabbatch{1}.spm.stats.con.consess{4}.tcon.weights = contrasts_tactileRPE - contrasts_tactileSPE;
matlabbatch{1}.spm.stats.con.consess{4}.tcon.sessrep = 'none';

matlabbatch{1}.spm.stats.con.consess{5}.tcon.name = 'audioSPE-audioRPE';
matlabbatch{1}.spm.stats.con.consess{5}.tcon.weights = contrasts_audioSPE - contrasts_audioRPE;
matlabbatch{1}.spm.stats.con.consess{5}.tcon.sessrep = 'none';

matlabbatch{1}.spm.stats.con.consess{6}.tcon.name = 'audioRPE-audioSPE';
matlabbatch{1}.spm.stats.con.consess{6}.tcon.weights = contrasts_audioRPE - contrasts_audioSPE;
matlabbatch{1}.spm.stats.con.consess{6}.tcon.sessrep = 'none';

matlabbatch{1}.spm.stats.con.consess{7}.tcon.name = 'audio-tactile';
matlabbatch{1}.spm.stats.con.consess{7}.tcon.weights = contrasts_audioBoth - contrasts_tactileBoth;
matlabbatch{1}.spm.stats.con.consess{7}.tcon.sessrep = 'none';

matlabbatch{1}.spm.stats.con.consess{8}.tcon.name = 'tactile-audio';
matlabbatch{1}.spm.stats.con.consess{8}.tcon.weights = contrasts_tactileBoth - contrasts_audioBoth;
matlabbatch{1}.spm.stats.con.consess{8}.tcon.sessrep = 'none';

matlabbatch{1}.spm.stats.con.consess{9}.tcon.name = 'audio';
matlabbatch{1}.spm.stats.con.consess{9}.tcon.weights = contrasts_audioBoth;
matlabbatch{1}.spm.stats.con.consess{9}.tcon.sessrep = 'none';

matlabbatch{1}.spm.stats.con.consess{10}.tcon.name = 'tactile';
matlabbatch{1}.spm.stats.con.consess{10}.tcon.weights = contrasts_tactileBoth;
matlabbatch{1}.spm.stats.con.consess{10}.tcon.sessrep = 'none';

matlabbatch{1}.spm.stats.con.consess{11}.tcon.name = 'both';
matlabbatch{1}.spm.stats.con.consess{11}.tcon.weights = contrasts_both;
matlabbatch{1}.spm.stats.con.consess{11}.tcon.sessrep = 'none';



matlabbatch{1}.spm.stats.con.delete = del_old_con;
%%
spm('defaults', 'fMRI');
spm_jobman('run', matlabbatch);
clear matlabbatch

end



