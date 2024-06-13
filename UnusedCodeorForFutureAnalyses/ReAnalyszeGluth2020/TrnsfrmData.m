%% transform data
datadir = '/Users/bs3667/Dropbox (NYU Langone Health)/Bo Shen Working files/NoiseProject/Gluth2020/ReplicationStudy_Data';
filelist = {dir(fullfile(datadir, 'CD*')).name};
clear data;
outdir = '/Users/bs3667/Dropbox (NYU Langone Health)/Bo Shen Working files/NoiseProject/Gluth2020';
bidtxt = fullfile(outdir, 'TrnsfrmData','BidTask.txt'); % logfile for the bidding data
dpbid = fopen(bidtxt,'wt');
fprintf(dpbid,'%s\t%s\t%s\t%s\t%s\t%s\t%s\n', ...
    'subID','ID','trial1','trial2','bdm1','bdm2','bdmean');
choicetxt = fullfile(outdir, 'TrnsfrmData','Choice.txt'); % logfile for the choice data
dpch = fopen(choicetxt,'wt');
fprintf(dpch,'%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n', ...
    'subID','item1','item2','item3','V1','V2','V3','choice','RT');
fclose('all');
BDM = [];
Choice = [];
for subj = 1:numel(filelist)
    load(fullfile(datadir, filelist{subj}));
    tmp = strsplit(filelist{subj},'.mat');
    subID = str2double(tmp{1}(8:end));
    bdmIn = ContextDepData.BDM.data;
    BDM = [BDM; [repmat(subID,length(bdmIn),1), bdmIn]];
    
    dataIn = ContextDepData.ChoiceBehav.data;
    L = length(dataIn);
    V1 = bdmIn(dataIn(:,1),6);
    V2 = bdmIn(dataIn(:,2),6);
    V3 = bdmIn(dataIn(:,3),6);
    Choice = [Choice; [repmat(subID,L,1), dataIn(:,[1,2,3]), V1, V2, V3, dataIn(:,4), dataIn(:,5)]];
    
    data(s).y=4-ContextDepData.ChoiceBehav.data(:,4);
    
    bids=ContextDepData.BDM.data;
    setindices=ContextDepData.ChoiceBehav.data(:,1:3);
    
    data(s).X=mat2cell(fliplr(reshape(bids(setindices,6),250,3))',3,ones(250,1));
    
    data(s).J=ones(size(ContextDepData.ChoiceBehav.data(:,4)))*3;
    data(s).K=ones(size(ContextDepData.ChoiceBehav.data(:,4)));
    
    data(s).Z={};
    data(s).W={};
end
dlmwrite(bidtxt,BDM,'delimiter','\t','-append');
dlmwrite(choicetxt,Choice,'delimiter','\t','-append');
save('gluthdataout.mat','data');
%% transform data
bd = array2table(BDM, 'VariableNames', {'subID','ID','trial1','trial2','bdm1','bdm2','bdm mean'});
dcsn = array2table(Choice, 'VariableNames', {'subID','item 1','item 2','item 3','rV1','rV2','rV3','choice','RT'});

bd.('bdm std') = std([bd.("bdm1"), bd.("bdm2")], 0, 2);
 
bd.Properties.VariableNames{'ID'} = 'item 1';
mergedTable = join(dcsn, bd(:, {'subID', 'item 1', 'bdm mean', 'bdm std'}), 'Keys', {'subID', 'item 1'});
mergedTable.Properties.VariableNames{'item 1'} = 'ID3';
mergedTable.Properties.VariableNames{'bdm mean'} = 'V3';
mergedTable.Properties.VariableNames{'bdm std'} = 'sdV3';
bd.Properties.VariableNames{'item 1'} = 'item 2';
mergedTable = join(mergedTable, bd(:, {'subID', 'item 2', 'bdm mean', 'bdm std'}), 'Keys', {'subID', 'item 2'});
mergedTable.Properties.VariableNames{'item 2'} = 'ID1';
mergedTable.Properties.VariableNames{'bdm mean'} = 'V1';
mergedTable.Properties.VariableNames{'bdm std'} = 'sdV1';
bd.Properties.VariableNames{'item 2'} = 'item 3';
mergedTable = join(mergedTable, bd(:, {'subID', 'item 3', 'bdm mean', 'bdm std'}), 'Keys', {'subID', 'item 3'});
mergedTable.Properties.VariableNames{'item 3'} = 'ID2';
mergedTable.Properties.VariableNames{'bdm mean'} = 'V2';
mergedTable.Properties.VariableNames{'bdm std'} = 'sdV2';
    
key = [3, 1, 2]';
mergedTable.chosenItem = key(mergedTable.choice);
mergedTable = mergedTable(:, {'subID', 'V1', 'V2', 'V3', 'sdV1', 'sdV2', 'sdV3', 'ID1', 'ID2', 'ID3','chosenItem', 'RT'});

mergedTable.larger = nan(size(mergedTable.V1));
mergedTable.smaller = nan(size(mergedTable.V1));
mergedTable.distractor = nan(size(mergedTable.V1));
mergedTable.chosenV = nan(size(mergedTable.V1));

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
%% save the transformed data
mt = mergedTable(:,{'subID','V1', 'V2', 'V3', 'sdV1', 'sdV2', 'sdV3', 'ID1', 'ID2', 'ID3','chosenItem', 'RT','larger','smaller','distractor'});
save(fullfile(outdir, 'TrnsfrmData', 'TrnsfrmData.mat'), 'mt');
writetable(mt, fullfile(outdir, 'TrnsfrmData', 'TrnsfrmData.csv'));

