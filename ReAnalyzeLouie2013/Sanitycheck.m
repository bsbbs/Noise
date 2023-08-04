%% Code for re-analyzing the data from Louie et al., 2013
rtdir = '/Users/bs3667/Noise/ReAnalyzeLouie2013';
datadir = '/Users/bs3667/Dropbox (NYU Langone Health)/Bo Shen Working files/NoiseComparison/IndividualTrinaryChoiceData';
cd(rtdir);
addpath('../utils');
datadir = rtdir;
svdir = fullfile(rtdir, 'Results');
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
    mergedTable.Properties.VariableNames{'item 1'} = 'ID1';
    mergedTable.Properties.VariableNames{'bdm mean'} = 'V1';
    mergedTable.Properties.VariableNames{'bdm std'} = 'sd1';
    indvbd.Properties.VariableNames{'item 1'} = 'item 2';
    mergedTable = join(mergedTable, indvbd(:, {'item 2', 'bdm mean', 'bdm std'}), 'Keys', 'item 2');
    mergedTable.Properties.VariableNames{'item 2'} = 'ID2';
    mergedTable.Properties.VariableNames{'bdm mean'} = 'V2';
    mergedTable.Properties.VariableNames{'bdm std'} = 'sd2';
    indvbd.Properties.VariableNames{'item 2'} = 'item 3';
    mergedTable = join(mergedTable, indvbd(:, {'item 3', 'bdm mean', 'bdm std'}), 'Keys', 'item 3');
    mergedTable.Properties.VariableNames{'item 3'} = 'ID3';
    mergedTable.Properties.VariableNames{'bdm mean'} = 'V3';
    mergedTable.Properties.VariableNames{'bdm std'} = 'sd3';
    mergedTable = mergedTable(:, {'V1', 'V2', 'V3', 'sd1', 'sd2', 'sd3', 'ID1', 'ID2', 'ID3','choice', 'RT'});
    mergedTable.larger = nan(size(mergedTable.choice));
    mergedTable.smaller = nan(size(mergedTable.choice));
    mergedTable.distractor = nan(size(mergedTable.choice));
    mergedTable.chosenV = nan(size(mergedTable.choice));
    for i = 1:length(mergedTable.choice)
        mergedTable.chosenV(i) = table2array(mergedTable(i,mergedTable.choice(i)));
        if mergedTable.choice(i) == 1
            mergedTable.larger(i) = 0;
            mergedTable.smaller(i) = 0;
            mergedTable.distractor(i) = 1;
        elseif table2array(mergedTable(i, mergedTable.choice(i))) > table2array(mergedTable(i, 5-mergedTable.choice(i)))
            mergedTable.larger(i) = 1;
            mergedTable.smaller(i) = 0;
            mergedTable.distractor(i) = 0;
        elseif table2array(mergedTable(i, mergedTable.choice(i))) < table2array(mergedTable(i, 5-mergedTable.choice(i)))
            mergedTable.larger(i) = 0;
            mergedTable.smaller(i) = 1;
            mergedTable.distractor(i) = 0;
        end
    end

    summaryData = groupsummary(mergedTable, {'V2', 'V3'}, {'sum', 'sum', 'sum'}, {'larger', 'smaller', 'distractor'});
    summaryData.sum_larger(summaryData.sum_larger == 0) = nan;
    summaryData.sum_smaller(summaryData.sum_smaller == 0) = nan;
    summaryData.sum_distractor(summaryData.sum_distractor == 0) = nan;

    scatter(summaryData.V2, summaryData.V3, summaryData.sum_larger*20, 'filled', 'MarkerFaceColor', OKeeffe(1,:), 'MarkerFaceAlpha', 0.5);
    scatter(summaryData.V2, summaryData.V3, summaryData.sum_smaller*20, 'filled', 'MarkerFaceColor', OKeeffe(2,:), 'MarkerFaceAlpha', 0.5);
    scatter(summaryData.V2, summaryData.V3, summaryData.sum_distractor*20, 'filled', 'MarkerFaceColor', OKeeffe(4,:), 'MarkerFaceAlpha', 0.5);
    minval = min([summaryData.V2; summaryData.V3]);
    maxval = max([summaryData.V2; summaryData.V3]);
    plot([minval, maxval], [minval, maxval], '--');
    if mod(subi-1, 12) == 0
        legend({'larger', 'smaller', 'distractor'}, 'Location','best');
    end
    title(sublist{subi}(1:end-4));
    %if mod(subi, 12) == 0
    filename = sprintf('ChoiceCounts_toSubj%i', subi);
    mysavefig(h, filename, plotdir, 12, [8,11]);
    %end

end
