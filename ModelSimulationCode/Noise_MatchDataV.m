%% 
%% define directories
[os, ~, ~] = computer;
if os == 'MACI64'
    rootdir = '/Users/bs3667/Dropbox (NYU Langone Health)/';
    fitdir = '/Users/bs3667/Noise/modelfit';
end
plot_dir = fullfile(rootdir, 'Bo Shen Working files/NoiseProject/Prediction');
addpath(genpath(Gitdir));
%% Loading the data transformed in the code: /Users/bs3667/Noise/modelfit/ModelFit-DataTrnsfrm.m
datadir = '/Users/bs3667/Dropbox (NYU Langone Health)/CESS-Bo/myData';
load(fullfile(datadir, 'TrnsfrmData.mat'), 'mt');
model = 'FastBADS_FixMwsMean';
fit = tdfread(fullfile(fitdir,'Results', model, 'Best.txt'));
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
%% visualize fitting parameters
h = figure; 
subplot(2,2,1); hold on;
x = fit.eta(fit.modeli == 1 & fit.TimeConstraint == 10);
y = fit.eta(fit.modeli == 1 & fit.TimeConstraint == 1.5);
plot(x, y, 'k.', 'MarkerSize',12);
plot([0, 2], [0, 2],'k--');
xlabel('Low Time Pressure \eta');
ylabel('High Time Pressure \eta');
title('Model 1');
[~, p, ci, stats] = ttest(y-x);
p
text(.2, 1.8, sprintf('p = %.3f', p));
mysavefig(h, 'eta', fullfile(fitdir,'Results', model, 'plot'), 12, [6, 6]);

subplot(2,2,2); hold on;
x = fit.eta(fit.modeli == 2 & fit.TimeConstraint == 10);
y = fit.eta(fit.modeli == 2 & fit.TimeConstraint == 1.5);
plot(x, y, 'k.', 'MarkerSize',12);
plot([0, 2], [0, 2],'k--');
xlabel('Low Time Pressure \eta');
ylabel('High Time Pressure \eta');
title('Model 2');
mysavefig(h, 'eta', fullfile(fitdir,'Results', model, 'plot'), 12, [6, 6]);
[~, p, ci, stats] = ttest(y-x);
p
text(.2, 1.8, sprintf('p = %.3f', p))
subplot(2,2,3);hold on;
x = fit.eta(fit.modeli == 1 & fit.TimeConstraint == 10);
y = fit.eta(fit.modeli == 2 & fit.TimeConstraint == 10);
plot(x, y, '.', 'Color', 'k', 'MarkerSize',12);
plot([0, 2], [0, 2],'k--');
xlabel('Model 1 \eta');
ylabel('Model 2 \eta');
title('Low time pressure');
mysavefig(h, 'eta', fullfile(fitdir,'Results', model, 'plot'), 12, [6, 6]);
subplot(2,2,4);hold on;
x = fit.eta(fit.modeli == 1 & fit.TimeConstraint == 1.5);
y = fit.eta(fit.modeli == 2 & fit.TimeConstraint == 1.5);
plot(x, y, 'k.', 'MarkerSize',12);
plot([0, 2], [0, 2],'k--');
xlabel('Model 1 \eta');
ylabel('Model 2 \eta');
title('High time pressure');
mysavefig(h, 'eta', fullfile(fitdir,'Results', model, 'plot'), 12, [6, 6]);
%% Simulation
modeli = 2;
fulllist = unique(mt.subID);
Sublist = unique(mtconvert.subID);
N = length(Sublist);
mtmodel = [];
for s = 1:N
    fprintf('Subject %d:\t', s);
    for t = [10, 1.5]
        fprintf('TimeConstraint %1.1f:\t', t);
        dat = mtconvert(mtconvert.subID == Sublist(s) & mtconvert.TimeConstraint == t, :);
        scl = fit.scl(fulllist(fit.subID) == Sublist(s) & fit.TimeConstraint == t & fit.modeli == modeli);
        eta = fit.eta(fulllist(fit.subID) == Sublist(s) & fit.TimeConstraint == t & fit.modeli == modeli);
        x = [scl, eta];
        switch modeli
            case 1
                [probs, nLL] = dDNb(x, dat, mode);
                name = 'dDNb'; %, cut input, independent';
            case 2
                [probs, nLL] = dDNd(x, dat, mode);
                name = 'dDNd'; %, cut SIGMA, independent';
        end
        dat.modelprob1 = probs(1,:)';
        dat.modelprob2 = probs(2,:)';
        dat.modelprob3 = probs(3,:)';
        mtmodel = [mtmodel; dat];
    end
    fprintf('\n');
end
mtmodel.ratio = mtmodel.modelprob2./(mtmodel.modelprob1 + mtmodel.modelprob2);
% Visualization in heatmap
Treatment = sprintf('%s%i',model,modeli);
dat = mtmodel(mtmodel.chosenItem ~= 3 & ~isnan(mtmodel.chosenItem),:);
GrpMean = grpstats(dat, ["subID", "TimeConstraint", "Vaguenesscode", "ID3"], "mean", "DataVars", ["V3", "sdV3", "V3scld", "sdV3scld", "ratio"]);
Window = 0.15;
Varrng = [min(GrpMean.mean_sdV3scld), 1];% max(GrpMean.mean_sdV3scld)];
Bindow = 0.15;
h = figure;
ti = 0;
for t = [10, 1.5] % low, high
    ti = ti + 1;
    dat = GrpMean(GrpMean.TimeConstraint == t,:);
    v3vec = LowestV3:.03:1;
    varvec = Varrng(1):.03:Varrng(2);
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
            choice(ri,vi) = mean(section.mean_ratio);
            choicese(ri,vi) = std(section.mean_ratio)/sqrt(length(section.mean_ratio));
            sdV3scld(ri,vi) = mean(section.mean_sdV3scld);
        end
    end
    
    choice(Ntrial<50) = NaN;
    subplot(2, 2, 1+(ti-1)*2); hold on;
    colormap("bone");
    cmap = bone(numel(varvec));
    for ri = 1:numel(varvec)
        plot(v3vec, choice(ri,:), '.-', 'Color', cmap(ri,:));
    end
    xlabel('Scaled V3');
    ylabel('% Correct (V1 & V2)');

    subplot(2, 2, 2+(ti-1)*2); %hold on;
    colormap("hot");
    imagesc(v3vec, varvec, choice);
    title('% Correct (V1 & V2)');
    colorbar;
    xlabel('Scaled V3');
    ylabel('V3 Variance');
end
mysavefig(h, sprintf('ChoiceModel_heatmap_%s', Treatment), plot_dir, 12, [10, 8]);
% Visualize in sliding windows
dat = mtmodel(mtmodel.chosenItem ~= 3 & ~isnan(mtmodel.chosenItem),:);
GrpMean = grpstats(dat, ["subID","TimeConstraint", "Vaguenesscode", "ID3"], "mean", "DataVars", ["V3", "sdV3", "V3scld", "sdV3scld", "choice","ratio"]);
colorpalette = {'#0000FF', '#FFC0CB', '#ADD8E6', '#FF0000'};
rgbMatrix = [
    0, 0, 255;   % Blue
    255, 192, 203; % Pink
    173, 216, 230; % Light Blue
    255, 0, 0     % Red
]/255;
Window = 0.15;
h = figure;
hold on;
vi = 0;
for t = [10, 1.5] % low, high
    for v = [1, 0] % vague, precise
        vi = vi + 1;
        dat = GrpMean(GrpMean.TimeConstraint == t & GrpMean.Vaguenesscode == v,:);
        % plot(dat.mean_V3scld, dat.mean_choice, '.', 'Color', colorpalette{vi});
        Ntrial = [];
        choice = [];
        choicese = [];
        ratio = [];
        ratiose = [];
        sdV3scld = [];
        v3vec = LowestV3:.015:1;
        for v3 = v3vec
            section = dat(dat.mean_V3scld >= v3 - Window & dat.mean_V3scld <= v3 + Window,:);
            Ntrial = [Ntrial, sum(section.GroupCount)];
            choice = [choice, mean(section.mean_choice)];
            choicese = [choicese, std(section.mean_choice)/sqrt(length(section.mean_choice))];
            ratio = [ratio, mean(section.mean_ratio)];
            ratiose = [ratiose, std(section.mean_ratio)/sqrt(length(section.mean_ratio))];
            sdV3scld = [sdV3scld, mean(section.mean_sdV3scld)];
        end
        plot(v3vec, ratio, '-', 'Color', colorpalette{vi}, 'LineWidth', 2);
        % fill([v3vec fliplr(v3vec)], [ratio-ratiose fliplr(ratio+ratiose)], rgbMatrix(vi,:), 'FaceAlpha', 0.3, 'EdgeColor', 'none');
    end
end
xlim([0,1]);
xlabel('Scaled V3');
ylabel('% Correct (V1 & V2)');
mysavefig(h, sprintf('Ratio_ModelPredict_%s', Treatment), plot_dir, 12, [4, 4]);
%%
function [probs, nll] = dDNb(x, dat, mode) % cut inputs, independent
% set the lower boundary for every input value distribution as zero
% samples in the denominator are independent from the numerator
% the SIGMA term in the denominator is non-negative after that.
if gpuDeviceCount > 0
    gpuparallel = 1;
else
    gpuparallel = 0;
end
Mp = 1;%x(1);
wp = 1;%x(2);
scl = x(1); % scaling for early noise
eta = x(2); % late noise standard deviation
data = dat(:, {'V1', 'V2', 'V3', 'sdV1','sdV2','sdV3','chosenItem'});
num_samples = 2000;
Ntrl = size(dat,1);
if strcmp(mode, 'absorb')
    samples = [];
    for ci = 1:3
        if gpuparallel
            values = gpuArray(data.(['V',num2str(ci)])');
            stds = gpuArray(data.(['sdV', num2str(ci)])')*scl;
            samples(ci,:,:) = max(gpuArray.randn([num_samples, Ntrl]).*stds + repmat(values, num_samples, 1), 0);
        else
            values = data.(['V',num2str(ci)])';
            stds = data.(['sdV', num2str(ci)])'*scl;
            samples(ci,:,:) = max(randn([num_samples, Ntrl]).*stds + repmat(values, num_samples, 1), 0);
        end
    end
    samplesD1 = [];
    samplesD2 = [];
    samplesD3 = [];
    for ci = 1:3
        if gpuparallel
            values = gpuArray(data.(['V',num2str(ci)])');
            stds = gpuArray(data.(['sdV', num2str(ci)])')*scl;
            samplesD1(ci,:,:) = max(gpuArray.randn([num_samples, Ntrl]).*stds + repmat(values, num_samples, 1), 0);
            samplesD2(ci,:,:) = max(gpuArray.randn([num_samples, Ntrl]).*stds + repmat(values, num_samples, 1), 0);
            samplesD3(ci,:,:) = max(gpuArray.randn([num_samples, Ntrl]).*stds + repmat(values, num_samples, 1), 0);
        else
            values = data.(['V',num2str(ci)])';
            stds = data.(['sdV', num2str(ci)])'*scl;
            samplesD1(ci,:,:) = max(randn([num_samples, Ntrl]).*stds + repmat(values, num_samples, 1), 0);
            samplesD2(ci,:,:) = max(randn([num_samples, Ntrl]).*stds + repmat(values, num_samples, 1), 0);
            samplesD3(ci,:,:) = max(randn([num_samples, Ntrl]).*stds + repmat(values, num_samples, 1), 0);
        end
    end
elseif strcmp(mode, 'cutoff')
    error('The cutoff boundary algorithm has not been developped yet.');
end
D1 = sum(samplesD1, 1)*wp + Mp;
D2 = sum(samplesD1, 1)*wp + Mp;
D3 = sum(samplesD1, 1)*wp + Mp;
D = [D1; D2; D3];
if gpuparallel
    SVs = samples./D + gpuArray.randn(size(samples))*eta;
    choice = gpuArray(data.chosenItem');
else
    SVs = samples./D + randn(size(samples))*eta;
    choice = data.chosenItem';
end
max_from_each_distribution = SVs == max(SVs, [], 1);
probs = squeeze(sum(max_from_each_distribution, 2) / size(SVs, 2));
nll = -sum(log(max(probs(sub2ind(size(probs), choice, 1:size(probs, 2))), eps)));
if gpuparallel
    nll = gather(nll);
end
end

function [probs, nll] = dDNd(x, dat, mode) % cut SIGMA, independent
% set the lower boundary for the summed SIGMA in the denominator
% but not for the input values in the numerator
% samples in the denominator are independent from the numerator
if gpuDeviceCount > 0
    gpuparallel = 1;
else
    gpuparallel = 0;
end
Mp = 1;%x(1);
wp = 1;%x(2);
scl = x(1); % scaling for early noise
eta = x(2); % late noise standard deviation
data = dat(:, {'V1', 'V2', 'V3', 'sdV1','sdV2','sdV3','chosenItem'});
num_samples = 2000;
Ntrl = size(dat,1);
samples = [];
for ci = 1:3
    if gpuparallel
        values = gpuArray(data.(['V',num2str(ci)])');
        stds = gpuArray(data.(['sdV', num2str(ci)])')*scl;
        samples(ci,:,:) = gpuArray.randn([num_samples, Ntrl]).*stds + repmat(values, num_samples, 1);
    else
        values = data.(['V',num2str(ci)])';
        stds = data.(['sdV', num2str(ci)])'*scl;
        samples(ci,:,:) = randn([num_samples, Ntrl]).*stds + repmat(values, num_samples, 1);
    end
end
samplesD1 = [];
samplesD2 = [];
samplesD3 = [];
for ci = 1:3
    if gpuparallel
        values = gpuArray(data.(['V',num2str(ci)])');
        stds = gpuArray(data.(['sdV', num2str(ci)])')*scl;
        samplesD1(ci,:,:) = gpuArray.randn([num_samples, Ntrl]).*stds + repmat(values, num_samples, 1);
        samplesD2(ci,:,:) = gpuArray.randn([num_samples, Ntrl]).*stds + repmat(values, num_samples, 1);
        samplesD3(ci,:,:) = gpuArray.randn([num_samples, Ntrl]).*stds + repmat(values, num_samples, 1);
    else
        values = data.(['V',num2str(ci)])';
        stds = data.(['sdV', num2str(ci)])'*scl;
        samplesD1(ci,:,:) = randn([num_samples, Ntrl]).*stds + repmat(values, num_samples, 1);
        samplesD2(ci,:,:) = randn([num_samples, Ntrl]).*stds + repmat(values, num_samples, 1);
        samplesD3(ci,:,:) = randn([num_samples, Ntrl]).*stds + repmat(values, num_samples, 1);
    end
end
if strcmp(mode, 'absorb')
    D1 = max(sum(samplesD1, 1),0)*wp + Mp;
    D2 = max(sum(samplesD2, 1),0)*wp + Mp;
    D3 = max(sum(samplesD3, 1),0)*wp + Mp;
elseif strcmp(mode, 'cutoff')
    error('The cutoff boundary algorithm has not been developped yet.');
end
D = [D1; D2; D3];
if gpuparallel
    SVs = samples./D + gpuArray.randn(size(samples))*eta;
    choice = gpuArray(data.chosenItem');
else
    SVs = samples./D + randn(size(samples))*eta;
    choice = data.chosenItem';
end
max_from_each_distribution = SVs == max(SVs, [], 1);
probs = squeeze(sum(max_from_each_distribution, 2) / size(SVs, 2));
nll = -sum(log(max(probs(sub2ind(size(probs), choice, 1:size(probs, 2))), eps)));
if gpuparallel
    nll = gather(nll);
end
end
