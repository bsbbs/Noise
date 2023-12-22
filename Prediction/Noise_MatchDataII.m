%% Simulation on noise that matches the value settings in the empirical data
%% define directories
[os, ~, ~] = computer;
if os == 'MACI64'
    rootdir = '/Users/bs3667/Dropbox (NYU Langone Health)/';
end
plot_dir = fullfile(rootdir, 'Bo Shen Working files/NoiseProject/Prediction');
Gitdir = '~/Documents/Noise';
addpath(genpath(Gitdir));
%% Loading the data transformed in the code: /Users/bs3667/Noise/modelfit/ModelFit-DataTrnsfrm.m
datadir = '/Users/bs3667/Dropbox (NYU Langone Health)/CESS-Bo/myData';
load(fullfile(datadir, 'TrnsfrmData.mat'), 'mt');

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
GrpMeanraw = grpstats(dat, ["subID","TimeConstraint", "Vaguenesscode", "ID3"], "mean", "DataVars", ["V3", "sdV3", "V3scld", "sdV3scld", "choice"]);
Treatment = 'Point';%'Point'; %'Raw'; %'Demean'; %
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
%%
GrpMean = sortrows(GrpMean, {'subID', 'mean_V3scld'}, {'ascend', 'ascend'});
Sublist = unique(GrpMean.subID);
N = length(Sublist);
h = figure;
ti = 0;
for t = [10, 1.5]
    ti = ti + 1;
    subplot(2,1,ti); hold on;
    dat = GrpMean(GrpMean.TimeConstraint == t,:);
    for s = 1:N
        indv = dat(dat.subID == Sublist(s),:);
        plot(indv.mean_V3scld, indv.mean_choice,'.-');
    end
    xlim([0,1]);
end
xlabel('Scaled V3');
ylabel('% Correct (V1 & V2)');
mysavefig(h, sprintf('Choice_Data_Subjects_%s', Treatment), plot_dir, 12, [5, 8]);

%% Choice accuracy in a heatmap of mean value and variance
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
    subplot(2,2,1+(ti-1)*2); hold on;
    colormap("bone");
    cmap = bone(numel(varvec));
    for ri = 1:numel(varvec)
        plot(v3vec, choice(ri,:), '.-', 'Color', cmap(ri,:));
    end
    xlabel('Scaled V3');
    ylabel('% Correct (V1 & V2)');
    %ylim([.4, .8]);
    title(sprintf('Time limit %1.1fs', t));

    subplot(2,2,2+(ti-1)*2); hold on;
    colormap("hot");
    %imagesc(v3vec, varvec, Nsubj);
    %plot(varvec(maxri),v3vec(maxvi),'b-', 'LineWidth',2);
    imagesc(v3vec, varvec, choice); % , [0.4, 0.8]
    %title('N subjects');
    colorbar;
    xlabel('Scaled V3');
    ylabel('V3 Variance');
    ylim([0,1]);
end
mysavefig(h, sprintf('Choice_Data_%s', Treatment), plot_dir, 12, [10, 8]);

%% Simulations
M = 1;
w = 1;
sL = .1;
lL = .1;
mode = 'absorb';
mtmodel = [];
for t = [10, 1.5] % low, high
    if t == 10
        x = [M, 1, .1];
    elseif t == 1.5
        x = [M, .5, lL];
    end
    for v = [1, 0] % vague, precise
        dat = mtconvert(mtconvert.TimeConstraint == t & mtconvert.Vaguenesscode == v,:);
        probs = dDN(x, dat, mode);
        ratio = probs(2,:)./sum(probs(1:2,:),1);
        dat.modelprob1 = probs(1,:)';
        dat.modelprob2 = probs(2,:)';
        dat.modelprob3 = probs(3,:)';
        mtmodel = [mtmodel; dat];
    end
end
mtmodel.ratio = mtmodel.modelprob2./(mtmodel.modelprob1 + mtmodel.modelprob2);
%% Visualization in heatmap
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

%% Simulations - based on the data structure
M = 1;
w = 1;
sL = .06;
lL = .12;
mode = 'absorb';
mtmodel = [];
for t = [10, 1.5] % low, high
    if t == 10
        x = [M, 1, sL];
    elseif t == 1.5
        x = [M, 1, lL];
    end
    for v = [1, 0] % vague, precise
        dat = mtconvert(mtconvert.TimeConstraint == t & mtconvert.Vaguenesscode == v,:);
        probs = dDN(x, dat, mode);
        ratio = probs(2,:)./sum(probs(1:2,:),1);
        dat.modelprob1 = probs(1,:)';
        dat.modelprob2 = probs(2,:)';
        dat.modelprob3 = probs(3,:)';
        mtmodel = [mtmodel; dat];
    end
end
mtmodel.ratio = mtmodel.modelprob2./(mtmodel.modelprob1 + mtmodel.modelprob2);

%% Visualize in sliding windows
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
        fill([v3vec fliplr(v3vec)], [ratio-ratiose fliplr(ratio+ratiose)], rgbMatrix(vi,:), 'FaceAlpha', 0.3, 'EdgeColor', 'none');
    end
end
xlim([0,1]);
xlabel('Scaled V3');
ylabel('% Correct (V1 & V2)');
mysavefig(h, sprintf('Ratio_ModelPredict_%s', Treatment), plot_dir, 12, [4, 4]);
%% functions
function probs = dDN(x, dat, mode) % cut inputs, independent
% set the lower boundary for every input value distribution as zero
% samples in the denominator are independent from the numerator
% the SIGMA term in the denominator will be natually non-negative after that.
if gpuDeviceCount > 0
    gpuparallel = 1;
else
    gpuparallel = 0;
end
Mp = x(1);
wp = x(2);
eta = x(3);
scl = 1;
data = dat(:, {'V1', 'V2', 'V3', 'sdV1','sdV2','sdV3','chosenItem'});
num_samples = 15;
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
D2 = sum(samplesD2, 1)*wp + Mp;
D3 = sum(samplesD3, 1)*wp + Mp;
D = [D1; D2; D3];
if gpuparallel
    SVs = samples./D + gpuArray.randn(size(samples))*eta;
    %choice = gpuArray(data.chosenItem');
else
    SVs = samples./D + randn(size(samples))*eta;
    %choice = data.chosenItem';
end
max_from_each_distribution = SVs == max(SVs, [], 1);
probs = squeeze(sum(max_from_each_distribution, 2) / size(SVs, 2));
end
