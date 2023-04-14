nrTrials = 60;

combinations = ["0","A"; "0","B"; "0","C"; "1","A"; "1","B"; "1","C"; "2","A"; "2","B"; "2","C"];


modality0A = 0.5;
modality0B = 0.35;
modality0C = 0.15;
modality1A = 0.15;
modality1B = 0.50;
modality1C = 0.35;
modality2A = 0.35;
modality2B = 0.15;
modality2C = 0.50;

count = 0;

for i = 1:50000000
    randomLayout = randperm(nrTrials);
    allTrialStructures(randomLayout,1,i) = repelem(combinations(:,1), [modality0A*nrTrials/3, modality0B*nrTrials/3, modality0C*nrTrials/3, modality1A*nrTrials/3, modality1B*nrTrials/3,modality1C*nrTrials/3,modality2A*nrTrials/3,modality2B*nrTrials/3,modality2C*nrTrials/3]);
    allTrialStructures(randomLayout,2,i) = repelem(combinations(:,2), [modality0A*nrTrials/3, modality0B*nrTrials/3, modality0C*nrTrials/3, modality1A*nrTrials/3, modality1B*nrTrials/3,modality1C*nrTrials/3,modality2A*nrTrials/3,modality2B*nrTrials/3,modality2C*nrTrials/3]);

end


for i = 1:50000000                    


    subCond10 = sum(allTrialStructures(1:20,1,i) == "0" & allTrialStructures(1:20,2,i) == "A") >= 3;
    subCond20 = sum(allTrialStructures(20:40,1,i) == "0" & allTrialStructures(20:40,2,i) == "A") >= 3;
    subCond30 = sum(allTrialStructures(40:60,1,i) == "0" & allTrialStructures(40:60,2,i) == "A") >= 3;
    subCond11 = sum(allTrialStructures(1:20,1,i) == "1" & allTrialStructures(1:20,2,i) == "B") >= 3;
    subCond21 = sum(allTrialStructures(20:40,1,i) == "1" & allTrialStructures(20:40,2,i) == "B") >= 3;
    subCond31 = sum(allTrialStructures(40:60,1,i) == "1" & allTrialStructures(40:60,2,i) == "B") >= 3;
    subCond12 = sum(allTrialStructures(1:20,1,i) == "2" & allTrialStructures(1:20,2,i) == "C") >= 3;
    subCond22 = sum(allTrialStructures(20:40,1,i) == "2" & allTrialStructures(20:40,2,i) == "C") >= 3;
    subCond32 = sum(allTrialStructures(40:60,1,i) == "2" & allTrialStructures(40:60,2,i) == "C") >= 3;
    cond1 = subCond10 & subCond20 & subCond30 & subCond11 & subCond21 & subCond31 & subCond12 & subCond22 & subCond32;
    subCond40 = sum(allTrialStructures(1:20,1,i) == "0" & allTrialStructures(1:20,2,i) == "B") >= 2;
    subCond50 = sum(allTrialStructures(20:40,1,i) == "0" & allTrialStructures(20:40,2,i) == "B") >= 2;
    subCond60 = sum(allTrialStructures(40:60,1,i) == "0" & allTrialStructures(40:60,2,i) == "B") >= 2;
    subCond41 = sum(allTrialStructures(1:20,1,i) == "1" & allTrialStructures(1:20,2,i) == "C") >= 2;
    subCond51 = sum(allTrialStructures(20:40,1,i) == "1" & allTrialStructures(20:40,2,i) == "C") >= 2;
    subCond61 = sum(allTrialStructures(40:60,1,i) == "1" & allTrialStructures(40:60,2,i) == "C") >= 2;
    subCond42 = sum(allTrialStructures(1:20,1,i) == "2" & allTrialStructures(1:20,2,i) == "A") >= 2;
    subCond52 = sum(allTrialStructures(20:40,1,i) == "2" & allTrialStructures(20:40,2,i) == "A") >= 2;
    subCond62 = sum(allTrialStructures(40:60,1,i) == "2" & allTrialStructures(40:60,2,i) == "A") >= 2;
    cond2 = subCond40 & subCond50 & subCond60 & subCond41 & subCond51 & subCond61 & subCond42 & subCond52 & subCond62;
    subCond70 = sum(allTrialStructures(1:20,1,i) == "0" & allTrialStructures(1:20,2,i) == "C") == 1;
    subCond80 = sum(allTrialStructures(20:40,1,i) == "0" & allTrialStructures(20:40,2,i) == "C") == 1;
    subCond90 = sum(allTrialStructures(40:60,1,i) == "0" & allTrialStructures(40:60,2,i) == "C") == 1;
    subCond71 = sum(allTrialStructures(1:20,1,i) == "1" & allTrialStructures(1:20,2,i) == "A") == 1;
    subCond81 = sum(allTrialStructures(20:40,1,i) == "1" & allTrialStructures(20:40,2,i) == "A") == 1;
    subCond91 = sum(allTrialStructures(40:60,1,i) == "1" & allTrialStructures(40:60,2,i) == "A") == 1;
    subCond72 = sum(allTrialStructures(1:20,1,i) == "2" & allTrialStructures(1:20,2,i) == "B") == 1;
    subCond82 = sum(allTrialStructures(20:40,1,i) == "2" & allTrialStructures(20:40,2,i) == "B") == 1;
    subCond92 = sum(allTrialStructures(40:60,1,i) == "2" & allTrialStructures(40:60,2,i) == "B") == 1;
    cond3 = subCond70 & subCond80 & subCond90 & subCond71 & subCond81 & subCond91 & subCond72 & subCond82 & subCond92;
    
    % Putting the condition such that they are always exactly in proportion
    % of 10, 7 and 3 for each 20 trials in terms of common, medium and
    % rare events.
    subC11 = sum((allTrialStructures(1:20,1,i) == "0" & allTrialStructures(1:20,2,i) == "A") |(allTrialStructures(1:20,1,i) == "1" & allTrialStructures(1:20,2,i) == "B")|(allTrialStructures(1:20,1,i) == "2" & allTrialStructures(1:20,2,i) == "C")) == 10;
    subC12 = sum((allTrialStructures(20:40,1,i) == "0" & allTrialStructures(20:40,2,i) == "A") |(allTrialStructures(20:40,1,i) == "1" & allTrialStructures(20:40,2,i) == "B")|(allTrialStructures(20:40,1,i) == "2" & allTrialStructures(20:40,2,i) == "C")) == 10;
    subC13 = sum((allTrialStructures(40:60,1,i) == "0" & allTrialStructures(40:60,2,i) == "A") |(allTrialStructures(40:60,1,i) == "1" & allTrialStructures(40:60,2,i) == "B")|(allTrialStructures(40:60,1,i) == "2" & allTrialStructures(40:60,2,i) == "C")) == 10;
    subC21 = sum((allTrialStructures(1:20,1,i) == "0" & allTrialStructures(1:20,2,i) == "B") |(allTrialStructures(1:20,1,i) == "1" & allTrialStructures(1:20,2,i) == "C")|(allTrialStructures(1:20,1,i) == "2" & allTrialStructures(1:20,2,i) == "A")) == 7;
    subC22 = sum((allTrialStructures(20:40,1,i) == "0" & allTrialStructures(20:40,2,i) == "B") |(allTrialStructures(20:40,1,i) == "1" & allTrialStructures(20:40,2,i) == "C")|(allTrialStructures(20:40,1,i) == "2" & allTrialStructures(20:40,2,i) == "A")) == 7;
    subC23 = sum((allTrialStructures(40:60,1,i) == "0" & allTrialStructures(40:60,2,i) == "B") |(allTrialStructures(40:60,1,i) == "1" & allTrialStructures(40:60,2,i) == "C")|(allTrialStructures(40:60,1,i) == "2" & allTrialStructures(40:60,2,i) == "A")) == 7;
    subC = subC11 & subC12 & subC13 & subC21 & subC22 & subC23;

    conditions(i) = cond1 & cond2 & cond3 & subC;
    if conditions(i) == 1
        count = count + 1;
    end

    if conditions(i)
        trialStructure(:,1,count) = allTrialStructures(:,1,i);
        trialStructure(:,2,count) = allTrialStructures(:,2,i);
    end


end

%checking if all structures are unique
[s1,s2,s3] = size(trialStructure);
trialStructure = reshape(trialStructure,s1*s2,s3,1)';
trialStructure = unique(trialStructure,'rows','stable');
trialStructure = reshape(trialStructure',s1,s2,[]);