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
sublist = unique(mt.subID);
blacklist = [22102405; 22102705; 22102708; 22071913; 22110306];
sublist = sublist(~ismember(sublist, blacklist));
N = length(sublist);
mtconvert = [];
for s = 1:N
    indvtask = mt(mt.subID == sublist(s),:);
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
sublist = unique(GrpMeanraw.subID);
N = length(sublist);
if strcmp(Treatment, "Demean")
    GrpMean = [];
    for s = 1:N
        indv = GrpMeanraw(GrpMeanraw.subID == sublist(s),:);
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

%% Choice accuracy in a heatmap of Targets difference and variances
dat = mtconvert(mtconvert.chosenItem ~= 3 & ~isnan(mtconvert.chosenItem),:);
dat.Vdscld = dat.V2scld - dat.V1scld;
dat.sdVdscld = sqrt(dat.sdV1scld.^2 + dat.sdV2scld.^2);

Window = .15;
Bindow = .15/2;
LowestVd = 0;
h = figure;
filename = 'ChoiceData_Targets';
vdvec = LowestVd:.03:1.5;
varvec = min(dat.sdVdscld):.015:1;
Ntrial = NaN(numel(varvec), numel(vdvec));
choice = NaN(numel(varvec), numel(vdvec));
% choicese = NaN(numel(varvec), numel(vdvec));
for vi = 1:numel(vdvec)
    for ri = 1:numel(varvec)
        vd = vdvec(vi);
        r = varvec(ri);
        maskvd = dat.Vdscld >= vd - Window & dat.Vdscld <= vd + Window;
        maskrd = dat.sdVdscld >= r - Bindow & dat.sdVdscld <= r + Bindow;
        section = dat(maskvd & maskrd,:);
        Ntrial(ri,vi) = length(section.choice);
        choice(ri,vi) = mean(section.choice)*100;
        % choicese(ri,vi) = std(section.choice)/sqrt(length(section.choice));
    end
end
colormap("jet");
imagesc(vdvec, varvec, choice);
title('');
colorbar;
xlabel('Scaled V1 - V2');
ylabel('V1 & V2 Variance');
set(gca, 'YDir', 'normal');
mysavefig(h, filename, plot_dir, 12, [5, 4]);

%% Using logistic slope instead of mean accuracy
fdat = mtconvert(mtconvert.chosenItem ~= 3 & ~isnan(mtconvert.chosenItem),:);
fdat.dprime = (fdat.V2scld - fdat.V1scld);%./sqrt(fdat.sdV1scld.^2 + fdat.sdV2scld.^2);
fdat.choice = double(fdat.choice);
sublist = unique(dat.subID);
N = numel(sublist);
Window = .15;
Bindow = .15/2;
LowestV3 = 0;
h = figure;
filename = "ChoiceData_LogisticSlope_VD";
simdat = fullfile(plot_dir, [filename+".mat"]);
v3vec = LowestV3:.03:1;
varvec = 0:.015:.4;
Ntrial = NaN(numel(varvec), numel(v3vec), 2);
choice = NaN(numel(varvec), numel(v3vec), 2);
% choicese = NaN(numel(varvec), numel(v3vec));
sdV3scld = NaN(numel(varvec), numel(v3vec), 2);
slope = NaN(numel(varvec), numel(v3vec), 2);
mdprime = NaN(numel(varvec), numel(v3vec), 2);
if ~exist(simdat, 'file')
    ti = 0;
    for t = [10, 1.5] % low, high
        ti = ti + 1;
        dat = fdat(fdat.TimeConstraint == t & fdat.V3scld <= 1,:);
        for vi = 1:numel(v3vec)
            for ri = 1:numel(varvec)
                v3 = v3vec(vi);
                r = varvec(ri);
                maskv3 = dat.V3scld >= v3 - Window & dat.V3scld <= v3 + Window;
                maskr3 = dat.sdV3scld >= r - Bindow & dat.sdV3scld <= r + Bindow;
                section = dat(maskv3 & maskr3,:);
                Ntrial(ri,vi,ti) = numel(section.trial);
                choice(ri,vi,ti) = mean(section.choice)*100;
                
                % choicese(ri,vi) = std(section.mean_choice)/sqrt(length(section.mean_choice));
                sdV3scld(ri,vi,ti) = mean(section.sdV3scld);
                formula = 'choice ~ dprime';
                maskdp = ~isnan(section.dprime) & (section.dprime)<Inf;
                mdprime(ri,vi,ti) = mean(section.dprime(maskdp));
                glm = fitglm(section(maskdp,:),formula,'distr','binomial', 'Intercept', false);
                slope(ri, vi,ti) = glm.Coefficients.Estimate;
            end
        end

        slope(Ntrial<30) = NaN;
        subplot(2,2,1+(ti-1)*2); hold on;
        colormap("bone");
        cmap = bone(numel(varvec));
        for ri = 1:numel(varvec)
            plot(v3vec, slope(ri,:,ti), '.-', 'Color', cmap(ri,:));
        end
        xlabel('Scaled V3');
        ylabel('Logistic slope| V1 & V2');
        % ylim([45, 75]);
        title(sprintf('Time limit %1.1fs', t));

        subplot(2,2,2+(ti-1)*2);
        colormap("jet");
        imagesc(v3vec, varvec, mdprime(:,:,ti));
        title('');
        colorbar;
        xlabel('Scaled V3');
        ylabel('V3 Variance');
        set(gca, 'YDir', 'normal');
    end
    mysavefig(h, filename, plot_dir, 12, [10, 8]);
    save(simdat, "slope","mdprime","choice","choicese","Ntrial","sdV3scld",'-mat');
else
    load(simdat);
end
%% Recovery from the model fitting


%% Individual difference in choice accuarcy as a function of mean V3 magnitude and mean V3 variance
GrpMean = grpstats(mtconvert, ["subID"], "mean", "DataVars", ["V3", "sdV3", "V3scld", "sdV3scld"]);
h = figure; hold on;
filename = 'IdvDffrnc_v3_sdv3';
sectdat = GrpMean;
plot(sectdat.mean_V3scld, sectdat.mean_sdV3scld, '.');

