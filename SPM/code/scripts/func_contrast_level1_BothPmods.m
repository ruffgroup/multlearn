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
    var_int1 = char(model(1,1));
    var_int2 = char(model(2,1));
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

if numel(find(contains(SPM.xX.name,['Sn(1) ChoiceAudio']))) > 0
    contrasts_audio = double(contains(SPM.xX.name, {['Sn(1) constant'], ['Sn(3) constant'], ['Sn(5) constant']}));
    contrasts_tactile = double(contains(SPM.xX.name, {['Sn(2) constant'], ['Sn(4) constant'], ['Sn(6) constant']}));
    contrasts_choice = double(contains(SPM.xX.name,{['Sn(1) ChoiceAudio*bf(1)'], ['Sn(2) ChoiceTactile*bf(1)'], ...
        ['Sn(3) ChoiceAudio*bf(1)'], ['Sn(4) ChoiceTactile*bf(1)'], ['Sn(5) ChoiceAudio*bf(1)'], ...
        ['Sn(6) ChoiceTactile*bf(1)']}));
    contrasts_feedback = double(contains(SPM.xX.name,{['Sn(1) FeedbackAudio*bf(1)'], ['Sn(2) FeedbackTactile*bf(1)'], ...
        ['Sn(3) FeedbackAudio*bf(1)'], ['Sn(4) FeedbackTactile*bf(1)'], ['Sn(5) FeedbackAudio*bf(1)'], ...
        ['Sn(6) FeedbackTactile*bf(1)']}));
elseif numel(find(contains(SPM.xX.name,['Sn(1) ChoiceTactile']))) > 0
    contrasts_tactile = double(contains(SPM.xX.name, {['Sn(1) constant'], ['Sn(3) constant'], ['Sn(5) constant']}));
    contrasts_audio = double(contains(SPM.xX.name, {['Sn(2) constant'], ['Sn(4) constant'], ['Sn(6) constant']}));
    contrasts_choice = double(contains(SPM.xX.name,{['Sn(2) ChoiceAudio*bf(1)'], ['Sn(1) ChoiceTactile*bf(1)'], ...
        ['Sn(4) ChoiceAudio*bf(1)'], ['Sn(3) ChoiceTactile*bf(1)'], ['Sn(6) ChoiceAudio*bf(1)'], ...
        ['Sn(5) ChoiceTactile*bf(1)']}));
    contrasts_feedback = double(contains(SPM.xX.name,{['Sn(2) FeedbackAudio*bf(1)'], ['Sn(1) FeedbackTactile*bf(1)'], ...
        ['Sn(4) FeedbackAudio*bf(1)'], ['Sn(3) FeedbackTactile*bf(1)'], ['Sn(6) FeedbackAudio*bf(1)'], ...
        ['Sn(5) FeedbackTactile*bf(1)']}));
else
    error("Constants audio and tactile not defined")
end


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

matlabbatch{1}.spm.stats.con.consess{7}.tcon.name = 'audioB-tactileB';
matlabbatch{1}.spm.stats.con.consess{7}.tcon.weights = contrasts_audioBoth - contrasts_tactileBoth;
matlabbatch{1}.spm.stats.con.consess{7}.tcon.sessrep = 'none';

matlabbatch{1}.spm.stats.con.consess{8}.tcon.name = 'tactileB-audioB';
matlabbatch{1}.spm.stats.con.consess{8}.tcon.weights = contrasts_tactileBoth - contrasts_audioBoth;
matlabbatch{1}.spm.stats.con.consess{8}.tcon.sessrep = 'none';

matlabbatch{1}.spm.stats.con.consess{9}.tcon.name = 'audioB';
matlabbatch{1}.spm.stats.con.consess{9}.tcon.weights = contrasts_audioBoth;
matlabbatch{1}.spm.stats.con.consess{9}.tcon.sessrep = 'none';

matlabbatch{1}.spm.stats.con.consess{10}.tcon.name = 'tactileB';
matlabbatch{1}.spm.stats.con.consess{10}.tcon.weights = contrasts_tactileBoth;
matlabbatch{1}.spm.stats.con.consess{10}.tcon.sessrep = 'none';

matlabbatch{1}.spm.stats.con.consess{11}.tcon.name = 'both';
matlabbatch{1}.spm.stats.con.consess{11}.tcon.weights = contrasts_both;
matlabbatch{1}.spm.stats.con.consess{11}.tcon.sessrep = 'none';

matlabbatch{1}.spm.stats.con.consess{12}.tcon.name = 'tactile-audio';
matlabbatch{1}.spm.stats.con.consess{12}.tcon.weights = contrasts_tactile - contrasts_audio;
matlabbatch{1}.spm.stats.con.consess{12}.tcon.sessrep = 'none';

matlabbatch{1}.spm.stats.con.consess{13}.tcon.name = 'audio-tactile';
matlabbatch{1}.spm.stats.con.consess{13}.tcon.weights = contrasts_audio - contrasts_tactile;
matlabbatch{1}.spm.stats.con.consess{13}.tcon.sessrep = 'none';

matlabbatch{1}.spm.stats.con.consess{14}.tcon.name = 'choice';
matlabbatch{1}.spm.stats.con.consess{14}.tcon.weights = contrasts_choice;
matlabbatch{1}.spm.stats.con.consess{14}.tcon.sessrep = 'none';

matlabbatch{1}.spm.stats.con.consess{15}.tcon.name = 'feedback';
matlabbatch{1}.spm.stats.con.consess{15}.tcon.weights = contrasts_feedback;
matlabbatch{1}.spm.stats.con.consess{15}.tcon.sessrep = 'none';

matlabbatch{1}.spm.stats.con.consess{16}.tcon.name = 'choice-feedback';
matlabbatch{1}.spm.stats.con.consess{16}.tcon.weights = contrasts_choice - contrasts_feedback;
matlabbatch{1}.spm.stats.con.consess{16}.tcon.sessrep = 'none';

matlabbatch{1}.spm.stats.con.consess{17}.tcon.name = 'feedback-choice';
matlabbatch{1}.spm.stats.con.consess{17}.tcon.weights = contrasts_feedback - contrasts_choice;
matlabbatch{1}.spm.stats.con.consess{17}.tcon.sessrep = 'none';




matlabbatch{1}.spm.stats.con.delete = del_old_con;
%%
spm('defaults', 'fMRI');
spm_jobman('run', matlabbatch);
clear matlabbatch

end



