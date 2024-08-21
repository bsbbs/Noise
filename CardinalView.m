%% define directories
rootdir = '/Users/bs3667/Dropbox (NYU Langone Health)/Bo Shen Working files/NoiseProject';
Gitdir = '~/Noise';
addpath(genpath(Gitdir));
plot_dir = fullfile(rootdir, 'AnalysisII','Fig4');
sim_dir = fullfile(rootdir, 'AnalysisII','Fig4');
if ~exist(plot_dir, 'dir')
    mkdir(plot_dir);
end
if ~exist(sim_dir,'dir')
    mkdir(sim_dir);
end
%% Transform data
load(fullfile(Gitdir,'myData','TrnsfrmData.mat'));
Sublist = unique(mt.subID);
N = length(Sublist);
mtconvert = [];
for s = 1:N
    indvtask = mt(mt.subID == Sublist(s),:);
    Vtrgt = unique([indvtask.V1; indvtask.V2]);
    mintrgt = min(Vtrgt);
    if mintrgt > 0 % skip this subject if the min target value is zero, because the values cannot be scaled and the value space does not help in testing of the hypothesis
        indvtask.V1scld = indvtask.V1/mintrgt;
        indvtask.V2scld = indvtask.V2/mintrgt;
        indvtask.V3scld = indvtask.V3/mintrgt;
        indvtask.sdV1scld = indvtask.sdV1/mintrgt;
        indvtask.sdV2scld = indvtask.sdV2/mintrgt;
        indvtask.sdV3scld = indvtask.sdV3/mintrgt;
        mtconvert = [mtconvert; indvtask];
    end  
end
mtconvert.choice = mtconvert.chosenItem - 1;

%% Choice accuracy in individuals
dat = mtconvert(mtconvert.chosenItem ~= 3 & ~isnan(mtconvert.chosenItem),:);
GrpMeanraw = grpstats(dat, ["subID","TimeConstraint", "Vaguenesscode", "ID3"], "mean", "DataVars", ["V3", "sdV3", "V3scld", "sdV3scld", "choice"]);
Treatment = 'Raw'; %'Demean'; %
Sublist = unique(GrpMeanraw.subID);
N = length(Sublist);
if strcmp(Treatment, "Demean")
    GrpMean = [];
    for s = 1:N
        indv = GrpMeanraw(GrpMeanraw.subID == Sublist(s),:);
        indv.mean_choice = indv.mean_choice - mean(indv.mean_choice);
        GrpMean = [GrpMean; indv];
    end
elseif strcmp(Treatment, 'Raw')
    GrpMean = GrpMeanraw;
end
%% Choice accuracy in a heatmap of mean value and variance
Window = .15;
Bindow = .15/2;
LowestV3 = 0;
h = figure;
ti = 0;
for t = [10, 1.5] % low, high
    ti = ti + 1;
    dat = GrpMean(GrpMean.mean_V3scld <= 1 & GrpMean.TimeConstraint == t,:);
    v3vec = LowestV3:.03:max(dat.mean_V3scld);
    varvec = min(GrpMean.mean_sdV3scld):.015:.4;
    Ntrial = NaN(numel(varvec), numel(v3vec));
    choice = NaN(numel(varvec), numel(v3vec));
    choicese = NaN(numel(varvec), numel(v3vec));
    sdV3scld = NaN(numel(varvec), numel(v3vec));
    for vi = 1:numel(v3vec)
        for ri = 1:numel(varvec)
            v3 = v3vec(vi);
            r = varvec(ri);
            maskv3 = dat.mean_V3scld >= v3 - Window & dat.mean_V3scld <= v3 + Window;
            maskr3 = dat.mean_sdV3scld >= r - Bindow & dat.mean_sdV3scld <= r + Bindow;
            section = dat(maskv3 & maskr3,:);
            Ntrial(ri,vi) = sum(section.GroupCount);
            choice(ri,vi) = mean(section.mean_choice)*100;
            choicese(ri,vi) = std(section.mean_choice)/sqrt(length(section.mean_choice));
            sdV3scld(ri,vi) = mean(section.mean_sdV3scld);
        end
    end

    choice(Ntrial<30) = NaN;
    subplot(2,2,1+(ti-1)*2); hold on;
    colormap("bone");
    cmap = bone(numel(varvec));
    for ri = 1:numel(varvec)
        plot(v3vec, choice(ri,:), '.-', 'Color', cmap(ri,:));
    end
    xlabel('Scaled V3');
    ylabel('% Correct (V1 & V2)');
    ylim([45, 75]);
    title(sprintf('Time limit %1.1fs', t));

    subplot(2,2,2+(ti-1)*2);
    colormap("jet");
    imagesc(v3vec, varvec, choice, [45, 75]);
    title('');
    colorbar;
    xlabel('Scaled V3');
    ylabel('V3 Variance');
    set(gca, 'YDir', 'normal');
end
mysavefig(h, sprintf('ChoiceData_V3scld_%s', Treatment), plot_dir, 12, [10, 8]);

