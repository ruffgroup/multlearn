
% Getting location of current directory where files are


subjects = [01 02 03 04 05 06 07 09 10 11 12 14 15 17 18 19 20 21 22 23 24 25 26 27 28 29 30 33 34 35 36 37 38 39 40 41 42 43 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63 64];
%subjects = [01 02];
corr_spe_rpe = [];
for subject=subjects
    subject = num2str(subject, '%02d');
    spe = load(['/data/fittedParameters/sub-' subject '/spe.mat']).spe;
    rpe_best = load(['/data/fittedParameters/sub-' subject '/rpeBestOverall.mat']).rpe.rpe;
    for run=1:6
        c = corrcoef(spe(run,:),rpe_best(run,:), 'rows','complete');
        corr_spe_rpe = [corr_spe_rpe c(1,2)];
    end
end

corr_R = mean(corr_spe_rpe);
hist(corr_spe_rpe)
h = findobj(gca,'Type','patch');
set(h,'FaceColor','w','EdgeColor','b', 'linewidth', 2)
xline(mean(corr_spre_rpe), 'linewidth', 2)
xlabel('correlation values', 'FontSize', 14);
ylabel('number of correlations', 'FontSize', 14);
title('Correlations between RPE and SPE')




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


