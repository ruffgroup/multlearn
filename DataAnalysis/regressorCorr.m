
% Getting location of current directory where files are
filePath = fileparts(fileparts(mfilename('fullpath')));
addpath(fullfile((filePath), 'Fitting'))


rpe_simple = load('rpe_simple.mat');
spe_simple = load('spe_simple.mat');
rpe_up = load('rpe_up.mat');
spe_up = load('spe_up.mat');
rpe_init = load('rpe_init.mat');
spe_init = load('spe_init.mat');
rpe_simple = rpe_simple.RPE_arr;
spe_simple = spe_simple.SPE_arr;
rpe_up = rpe_up.RPE_arr;
spe_up = spe_up.SPE_arr;
rpe_init = rpe_init.RPE_arr;
spe_init = spe_init.SPE_arr;

corr_simpleUp = [];
for i = 1:size(rpe_simple,1)
    for j = 1:size(rpe_simple,2)
        c = corrcoef(rpe_simple(i,j,~isnan(rpe_simple(i,j,:))), rpe_up(i,j,~isnan(rpe_up(i,j,:))));
        corr_simpleUp = [corr_simpleUp, c(1,2)];

    end
end


hist(corr_simpleUp)
h = findobj(gca,'Type','patch');
set(h,'FaceColor','w','EdgeColor','b', 'linewidth', 2)
xline(mean(corr_simpleUp), 'linewidth', 2)
xlabel('correlation values', 'FontSize', 14);
ylabel('number of correlations', 'FontSize', 14);
title('Correlations between RPE of simple model and RPE of',' trasfer learning model (3 paramerters) for last 6 participants')



corr_simpleInit = [];
for i = 1:size(rpe_simple,1)
    for j = 1:size(rpe_simple,2)
        c = corrcoef(rpe_simple(i,j,~isnan(rpe_simple(i,j,:))), rpe_init(i,j,~isnan(rpe_init(i,j,:))));
        corr_simpleInit = [corr_simpleInit, c(1,2)];

    end
end

% hist(corr_simpleInit)
% h = findobj(gca,'Type','patch');
% set(h,'FaceColor','w','EdgeColor','b', 'linewidth', 2)
% xline(mean(corr_simpleInit), 'linewidth', 2)
% xlabel('correlation values', 'FontSize', 14);
% ylabel('number of correlations', 'FontSize', 14);
% title('Correlations between RPE of simple model and RPE of',' initial value learning model (3 paramerters) for last 6 participants')
% 


% corr_init = [];
% 
% for i = 1:size(rpe_simple,1)
%     for j = 1:size(rpe_simple,2)
%         c = corrcoef(rpe_simple(i,j,~isnan(rpe_simple(i,j,:))), spe_simple(i,j,~isnan(rpe_simple(i,j,:))));
%         corr_init = [corr_init, c(1,2)];
% 
%     end
% end
% 
% 
% hist(corr_init)
% h = findobj(gca,'Type','patch');
% set(h,'FaceColor','w','EdgeColor','b', 'linewidth', 2)
% xline(mean(corr_init), 'linewidth', 2)
% xlabel('correlation values', 'FontSize', 14);
% ylabel('number of correlations', 'FontSize', 14);
% title('Correlations between RPE and surprise',' trasfer learning model (3 paramerters) for last 6 participants')
% 
% 


