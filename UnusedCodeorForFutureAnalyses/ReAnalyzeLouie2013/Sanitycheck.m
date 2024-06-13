%% Code for re-analyzing the data from Louie et al., 2013

%% define directories
[os, ~, ~] = computer;
if os == 'MACI64'
    rootdir = '/Users/bs3667/Dropbox (NYU Langone Health)/';
    fitdir = '/Users/bs3667/Noise/modelfit';
end
addpath(genpath(Gitdir));
datadir = fullfile(rootdir,'Bo Shen Working files/NoiseProject/Louie2013/IndividualTrinaryChoiceData');

svdir = fullfile(rootdir,'Bo Shen Working files/NoiseProject/Louie2013', 'Results');
if ~exist(svdir, 'dir')
    mkdir(svdir);
end

AnalysName = 'Indvchoice';
outdir = fullfile(svdir, AnalysName);
if ~exist(outdir, 'dir')
    mkdir(outdir);
end
plotdir = fullfile(outdir, 'plot');
if ~exist(plotdir, 'dir')
    mkdir(plotdir);
end
mtrxdir = fullfile(outdir, 'Objs');
if ~exist(mtrxdir, 'dir')
    mkdir(mtrxdir);
end

%% make color pallete
OKeeffe = [
    255, 192, 203;  % Pink (Pale Violet Red)
    100, 149, 237;  % Cornflower Blue
    255, 127, 80;   % Coral
    144, 238, 144;  % Light Green (Pale Green)
    255, 228, 196;  % Bisque
    147, 112, 219;  % Medium Purple
    0, 206, 209;    % Dark Turquoise
    250, 128, 114;  % Salmon
    152, 251, 152;  % Pale Green (Light Green)
    218, 112, 214;  % Orchid
    ]/255;
h = figure; hold on;
for i = 1:size(OKeeffe,1)
    plot(i, '.', 'MarkerSize',20, 'Color', OKeeffe(i,:));
end
%% load data
% files = myfnames(fullfile(datadir, 'CDdata_*'));
sublist = {dir(fullfile(datadir, 'CDdata_*')).name};
mt = [];
for subi = 1:numel(sublist)
    if mod(subi-1, 12) == 0
        h = figure; hold on;
        filename = sprintf('ChoiceCounts_toSubj%i', subi);
    end
    subplot(4,3, mod(subi-1, 12)+1); hold on;
    load(fullfile(datadir, sublist{subi}));

    % Create table from matrices (assuming the first column is the key variable)
    indvbd = array2table(ContextDepData.BDM.data, 'VariableNames', ContextDepData.BDM.colhead);
    indvbd.('bdm std') = std([indvbd.("bdm 1"), indvbd.("bdm 2")], 0, 2);
    indvdcsn = array2table(ContextDepData.ChoiceBehav.data , 'VariableNames', ContextDepData.ChoiceBehav.colhead);

    indvbd.Properties.VariableNames{'ID'} = 'item 1';
    mergedTable = join(indvdcsn, indvbd(:, {'item 1', 'bdm mean', 'bdm std'}), 'Keys', 'item 1');
    mergedTable.Properties.VariableNames{'item 1'} = 'ID3';
    mergedTable.Properties.VariableNames{'bdm mean'} = 'V3';
    mergedTable.Properties.VariableNames{'bdm std'} = 'sdV3';
    indvbd.Properties.VariableNames{'item 1'} = 'item 2';
    mergedTable = join(mergedTable, indvbd(:, {'item 2', 'bdm mean', 'bdm std'}), 'Keys', 'item 2');
    mergedTable.Properties.VariableNames{'item 2'} = 'ID1';
    mergedTable.Properties.VariableNames{'bdm mean'} = 'V1';
    mergedTable.Properties.VariableNames{'bdm std'} = 'sdV1';
    indvbd.Properties.VariableNames{'item 2'} = 'item 3';
    mergedTable = join(mergedTable, indvbd(:, {'item 3', 'bdm mean', 'bdm std'}), 'Keys', 'item 3');
    mergedTable.Properties.VariableNames{'item 3'} = 'ID2';
    mergedTable.Properties.VariableNames{'bdm mean'} = 'V2';
    mergedTable.Properties.VariableNames{'bdm std'} = 'sdV2';
    key = [3, 1, 2]';
    mergedTable.chosenItem = key(mergedTable.choice);
    mergedTable = mergedTable(:, {'V1', 'V2', 'V3', 'sdV1', 'sdV2', 'sdV3', 'ID1', 'ID2', 'ID3','chosenItem', 'RT'});
    mergedTable.larger = nan(size(mergedTable.V1));
    mergedTable.smaller = nan(size(mergedTable.V1));
    mergedTable.distractor = nan(size(mergedTable.V1));
    mergedTable.chosenV = nan(size(mergedTable.V1));
    mergedTable.subCode = repmat({sublist{subi}(8:end-4)}, size(mergedTable.V1));
    mergedTable.subID = repmat(subi, size(mergedTable.V1));
    for i = 1:length(mergedTable.V1)
        mergedTable.chosenV(i) = table2array(mergedTable(i,mergedTable.chosenItem(i)));
        if mergedTable.chosenItem(i) == 3
            mergedTable.larger(i) = 0;
            mergedTable.smaller(i) = 0;
            mergedTable.distractor(i) = 1;
        elseif table2array(mergedTable(i, mergedTable.chosenItem(i))) > table2array(mergedTable(i, 3-mergedTable.chosenItem(i)))
            mergedTable.larger(i) = 1;
            mergedTable.smaller(i) = 0;
            mergedTable.distractor(i) = 0;
        elseif table2array(mergedTable(i, mergedTable.chosenItem(i))) < table2array(mergedTable(i, 3-mergedTable.chosenItem(i)))
            mergedTable.larger(i) = 0;
            mergedTable.smaller(i) = 1;
            mergedTable.distractor(i) = 0;
        end
    end
    mt = [mt; mergedTable];

    summaryData = groupsummary(mergedTable, {'V1', 'V2'}, {'sum', 'sum', 'sum'}, {'larger', 'smaller', 'distractor'});
    summaryData.sum_larger(summaryData.sum_larger == 0) = nan;
    summaryData.sum_smaller(summaryData.sum_smaller == 0) = nan;
    summaryData.sum_distractor(summaryData.sum_distractor == 0) = nan;

    scatter(summaryData.V1, summaryData.V2, summaryData.sum_larger*20, 'filled', 'MarkerFaceColor', OKeeffe(1,:), 'MarkerFaceAlpha', 0.5);
    scatter(summaryData.V1, summaryData.V2, summaryData.sum_smaller*20, 'filled', 'MarkerFaceColor', OKeeffe(2,:), 'MarkerFaceAlpha', 0.5);
    scatter(summaryData.V1, summaryData.V2, summaryData.sum_distractor*20, 'filled', 'MarkerFaceColor', OKeeffe(4,:), 'MarkerFaceAlpha', 0.5);
    minval = min([summaryData.V1; summaryData.V2]);
    maxval = max([summaryData.V1; summaryData.V2]);
    plot([minval, maxval], [minval, maxval], '--');
    if mod(subi-1, 12) == 0
        legend({'larger', 'smaller', 'distractor'}, 'Location','best');
    end
    title(sublist{subi}(8:end-4));
    %if mod(subi, 12) == 0
    filename = sprintf('ChoiceCounts_toSubj%i', subi);
    mysavefig(h, filename, plotdir, 12, [8,11]);
    %end
end
%% save the transformed data
mt = mt(:,{'subID','subCode','V1', 'V2', 'V3', 'sdV1', 'sdV2', 'sdV3', 'ID1', 'ID2', 'ID3','chosenItem', 'RT','larger','smaller','distractor'});
save(fullfile(datadir, 'TrnsfrmData.mat'), 'mt');
writetable(mt, fullfile(datadir, 'TrnsfrmData.csv'));
