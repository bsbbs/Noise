%% Analyzing Louie et al., 2013
%% define directories
[os, ~, ~] = computer;
if os == 'MACI64'
    rootdir = '/Users/bs3667/Dropbox (NYU Langone Health)/';
    Gitdir = '~/Noise';
end
addpath(genpath(Gitdir));
datadir = fullfile(rootdir,'Bo Shen Working files/NoiseProject/Gluth2020/TrnsfrmData');
svdir = fullfile(rootdir,'Bo Shen Working files/NoiseProject/Gluth2020', 'Results');
if ~exist(svdir, 'dir')
    mkdir(svdir);
end
AnalysName = 'ChoiceAccuracy';
plot_dir = fullfile(svdir, AnalysName);
if ~exist(plot_dir, 'dir')
    mkdir(plot_dir);
end
%% loading data
load(fullfile(datadir, "TrnsfrmData.mat"));
%% Transform data
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
GrpMeanraw = grpstats(dat, ["subID", "ID3"], "mean", "DataVars", ["V3", "sdV3", "V3scld", "sdV3scld", "choice"]);
Treatment = 'Raw';%'Point'; %'Raw'; %'Demean'; %
LowestV3 = 0;
Sublist = unique(GrpMeanraw.subID);
N = length(Sublist);
if strcmp(Treatment, "Point")
    GrpMean = [];
    for s = 1:N
        indv = GrpMeanraw(GrpMeanraw.subID == Sublist(s),:);
        for t = [10, 1.5]
            sect = indv(indv.TimeConstraint == t,:);
            trunk = sect(sect.mean_V3scld >=0 & sect.mean_V3scld <= .3 & sect.mean_sdV3scld>=0 & sect.mean_sdV3scld <= .3,:);
            if ~isempty(trunk)
                point = mean(trunk.mean_choice);
                sect.mean_choice = sect.mean_choice - point; %mean(indv.mean_choice);
                GrpMean = [GrpMean; sect];
            else
                warning("%s",Sublist(s));
            end
        end
    end
elseif strcmp(Treatment, 'Raw')
    GrpMean = GrpMeanraw;
end

%% Choice accuracy in a heatmap of mean value and variance
Window = 0.15;
Varrng = [min(GrpMean.mean_sdV3scld), .7];% max(GrpMean.mean_sdV3scld)];
Bindow = 0.05;
h = figure;
dat = GrpMean;
v3vec = LowestV3:.03:1;
varvec = Varrng(1):.03:Varrng(2);
Ntrial = NaN(numel(varvec), numel(v3vec));
Nsubj = NaN(numel(varvec), numel(v3vec));
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
        Nsubj(ri,vi) = numel(unique(section.subID));
        choice(ri,vi) = mean(section.mean_choice);
        choicese(ri,vi) = std(section.mean_choice)/sqrt(length(section.mean_choice));
        sdV3scld(ri,vi) = mean(section.mean_sdV3scld);
    end
end
[maxri, maxvi] = find(Nsubj == max(Nsubj(:)));
choice(Ntrial<25) = NaN;
subplot(1,2,1); hold on;
colormap("bone");
cmap = bone(numel(varvec));
secti = 12;
for ri = 1:numel(varvec)
    plot(v3vec, choice(ri,:), '.-', 'Color', cmap(ri,:));
    if ri == 1
        plot(v3vec, choice(ri,:), '.-', 'Color', 'r', 'LineWidth',2);
    elseif ri == secti
        plot(v3vec, choice(ri,:), '.-', 'Color', 'b', 'LineWidth',2);
    end

end
xlabel('Scaled V3');
ylabel('% Correct (V1 & V2)');
ylim([.5, .9]);

subplot(1,2,2); hold on;
colormap("hot");
%imagesc(v3vec, varvec, Nsubj);
%plot(varvec(maxri),v3vec(maxvi),'b-', 'LineWidth',2);
imagesc(v3vec, varvec, choice, [0.5, 0.9]);
%title('N subjects');

plot([0, 0, 1, 1, 0], [varvec(1), varvec(2), varvec(2), varvec(1), varvec(1)]-.015, 'r-', 'LineWidth', 0.5);
plot([0, 0, 1, 1, 0], [varvec(secti), varvec(secti+1), varvec(secti+1), varvec(secti), varvec(secti)]-.015, 'b-', 'LineWidth', 0.5);

colorbar;
ylim([0,.7]);
xlim([0,1]);
xlabel('Scaled V3');
ylabel('V3 Variance');

mysavefig(h, sprintf('Choice_Data_%s', Treatment), plot_dir, 12, [10, 4]);