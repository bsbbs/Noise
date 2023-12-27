%% define directories
[os, ~, ~] = computer;
if os == 'MACI64'
    rootdir = '/Users/bs3667/Dropbox (NYU Langone Health)/';
    fitdir = '/Users/bs3667/Noise/modelfit';
end
plot_dir = fullfile(rootdir, 'Bo Shen Working files/NoiseProject/Prediction');
dumpdir = plot_dir;
Gitdir = '~/Documents/Noise';
addpath(genpath(Gitdir));
%% Loading the data transformed in the code: /Users/bs3667/Noise/modelfit/ModelFit-DataTrnsfrm.m
datadir = '/Users/bs3667/Dropbox (NYU Langone Health)/CESS-Bo/myData';
load(fullfile(datadir, 'TrnsfrmData.mat'), 'mt');
model = 'FastBADS_FixMw';
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
%% Simulation
modeli = 1;
mode = 'absorb';
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
dat = mtmodel(mtmodel.chosenItem ~= 3 & ~isnan(mtmodel.chosenItem),:);
GrpMeanMdl = grpstats(dat, ["subID","TimeConstraint", "Vaguenesscode", "ID3"], "mean", "DataVars", ["V3", "sdV3", "V3scld", "sdV3scld", "choice","ratio"]);

%% Choice accuracy in individuals
dat = mtconvert(mtconvert.chosenItem ~= 3 & ~isnan(mtconvert.chosenItem),:);
GrpMeanraw = grpstats(dat, ["subID","TimeConstraint", "Vaguenesscode", "ID3"], "mean", "DataVars", ["V3", "sdV3", "V3scld", "sdV3scld", "choice"]);
Treatment = 'Raw';%'Point'; %'Raw'; %'Demean'; %
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
%% Visualize in lines plot in x-axis of mean scaled V3
colorpalette = {'#FFC0CB', '#FF0000', '#0000FF', '#ADD8E6'};
rgbMatrix = [
    255, 192, 203; % Pink
    255, 0, 0     % Red
    0, 0, 255;   % Blue
    173, 216, 230; % Light Blue
    ]/255;
Window = 0.15;
LowestV3 = 0;
HighestV3 = 1;
h = figure;
hold on;
vi = 0;
Sldwndw = [];
for v = [0, 1] % vague, precise
    if v == 0
        subplot(1,2,1); hold on;
    elseif v == 1
        subplot(1,2,2); hold on;
    end
    for t = [10, 1.5] % low, high
        vi = vi + 1;
        % dat = GrpMean(GrpMean.TimeConstraint == t & GrpMean.Vaguenesscode == v & GrpMean.mean_V3scld >= LowestV3 & GrpMean.mean_V3scld <= HighestV3,:);
        dat = GrpMeanMdl(GrpMeanMdl.TimeConstraint == t & GrpMeanMdl.Vaguenesscode == v & GrpMeanMdl.mean_V3scld >= LowestV3 & GrpMeanMdl.mean_V3scld <= HighestV3,:);
        % plot data pattern overlapping with model fitting
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
            choice = [choice, mean(section.mean_choice*100)];
            choicese = [choicese, std(section.mean_choice*100)/sqrt(length(section.mean_choice))];
            ratio = [ratio, mean(section.mean_ratio*100)];
            ratiose = [ratiose, std(section.mean_ratio*100)/sqrt(length(section.mean_ratio))];
            sdV3scld = [sdV3scld, mean(section.mean_sdV3scld)];
        end
        
        plot(v3vec, choice, '-', 'Color', colorpalette{vi}, 'LineWidth', 2);
        fill([v3vec fliplr(v3vec)], [choice-choicese fliplr(choice+choicese)], rgbMatrix(vi,:), 'FaceAlpha', 0.3, 'EdgeColor', 'none');
        plot(v3vec, ratio-2, 'k--', 'LineWidth', 2);
        % Ntrial = [];
        % choice = [];
        % choicese = [];
        % ratio = [];
        % ratiose = [];
        % sdV3scld = [];
        % v3vec = LowestV3:.015:1;
        % for v3 = v3vec
        %     section = dat(dat.mean_V3scld >= v3 - Window & dat.mean_V3scld <= v3 + Window,:);
        %     Ntrial = [Ntrial, sum(section.GroupCount)];
        %     choice = [choice, mean(section.mean_choice*100)];
        %     choicese = [choicese, std(section.mean_choice*100)/sqrt(length(section.mean_choice))];
        %     sdV3scld = [sdV3scld, mean(section.mean_sdV3scld)];
        % end
        % tmp = [];
        % tmp.TimeConstraint = t*ones(numel(Ntrial),1);
        % tmp.Vaguenesscode = v*ones(numel(Ntrial),1);
        % tmp.Ntrial = Ntrial';
        % tmp.choice = choice';
        % tmp.choicese = choicese';
        % tmp.V3scld = v3vec';
        % tmp.sdV3scld = sdV3scld';
        % tmp = struct2table(tmp);
        % Sldwndw = [Sldwndw; tmp];
        
        % xlim([-.05,.85]);
    end
    xlabel('Scaled V3');
    ylabel('% Correct (V1 & V2)');
    mysavefig(h, sprintf('Ratio_ModelmData_%s', Treatment), plot_dir, 12, [8, 4]);
end

%% 2by2
Test = '2by2';
colorpalette = {'#0000FF', '#FFC0CB', '#ADD8E6', '#FF0000'};
rgbMatrix = [
    0, 0, 255;   % Blue
    255, 192, 203; % Pink
    173, 216, 230; % Light Blue
    255, 0, 0     % Red
    ]/255;

V1mean = 83;% 53;
V2mean = 88;% 58;
epsV1 = 9;
epsV2 = 9;

h = figure;
hold on;
vi = 0;
for t = [10, 1.5] % low, high
    if t == 10
        eta = .8;
    elseif t == 1.5
        eta = 1.9;
    end
    for v = [1, 0] % vague, precise
        vi = vi + 1;
        if v == 1
            eps = 9;
        elseif v == 0
            eps = 4;
        end
        V3 = linspace(0, V1mean*.8, 40)';
        V1 = V1mean*ones(size(V3));
        V2 = V2mean*ones(size(V3));
        sdV1 = epsV1*ones(size(V3))/2;
        sdV2 = epsV2*ones(size(V3))/2;
        sdV3 = eps*ones(size(V3));
        chosenItem = randi(3, size(V3));
        dat = table(V1,V2,V3,sdV1,sdV2,sdV3, chosenItem);
        %
        x = [1, 1, eta];
        probs = dDN(x, dat, 'absorb');
        ratio = probs(2,:)./(probs(1,:) + probs(2,:));
        plot(V3/V1mean, ratio, '.-', 'Color', colorpalette{vi});

    end
end
xlabel('Scaled V3');
ylabel('% Correct (V1 & V2)');
mysavefig(h, sprintf('Ratio_Model_2by2'), plot_dir, 12, [4, 4]);
%% graded color, two panels
V1mean = 83;% 53;
V2mean = 88;% 58;
epsV1 = 9; % early noise for V1
epsV2 = 9; % early noise for V2
V3 = linspace(0, V1mean*.8, 20)';
V1 = V1mean*ones(size(V3));
V2 = V2mean*ones(size(V3));
sdV1 = epsV1*ones(size(V3))/2;
sdV2 = epsV2*ones(size(V3))/2;
chosenItem = randi(3, size(V3)); % dummy variable required by the function, meaningless used here
etavec = linspace(.8, 1.9, 8); % different levels of late noise
filename = sprintf('Ratio_Model_%s', '2Panels');
% simulation
SimDatafile = fullfile(dumpdir, [filename, '.mat']);
if exist(SimDatafile,'file')
    load(SimDatafile);
else
    Ratios = nan(2, numel(etavec), numel(V3));
    tmp = nan(2, numel(etavec), 3, numel(V3));
    DNPStats = tmp;
    V3Stats = tmp;
    SVStats = tmp;
    for i = 1:2
        if i == 1
            v = 0;
            eps = 4;
        elseif i == 2
            v = 1;
            eps = 9;
        end
        sdV3 = eps*ones(size(V3));
        dat = table(V1,V2,V3,sdV1,sdV2,sdV3, chosenItem);
        for ti = 1:numel(etavec)
            eta = etavec(ti);
            x = [1, 1, eta]; % M , w, eta
            [probs, ~, dDNPQntls, V3Qntls, dSVQtls] = dDN(x, dat, 'absorb');
            DNPStats(i, ti, :, :) = dDNPQntls;
            V3Stats(i, ti, :, :) = V3Qntls;
            SVStats(i, ti, :, :) = dSVQtls;
            Ratios(i,ti,:) = probs(2,:)./(probs(1,:) + probs(2,:));
        end
    end
    xval = V3'/V1mean;
    save(SimDatafile, "Ratios","DNPStats","V3Stats","SVStats","xval",'-mat');
end
%% visualization
red = [1, 0, 0];
orange = [255, 165, 0] / 255;
green = [0, 1, 0];
colors = [orange; red; green];
h = figure;
for i = 1:2
    subplot(8, 2, 2+i); hold on;
    for opt = [3]
        Med = squeeze(V3Stats(i, 1, 2,:))';
        Qntls = squeeze(V3Stats(i, 1, [1,3],:));
        fill([xval, fliplr(xval)], [Qntls(1,:), fliplr(Qntls(2,:))], 'k', 'FaceAlpha', 0.3, 'EdgeColor', 'none');
        plot(xval, Med, 'k-', "LineWidth", 1);
    end
    ylabel("V3 (a.u.)");
    mysavefig(h, filename, plot_dir, 12, [8, 8]);

    if i == 1
        v = 0;
        startColor = [1, 0.75, 0.8]; % Pink
        endColor = [1, 0, 0]; % Red
    elseif i == 2
        v = 1;
        startColor = [0, 0, 1]; % Blue
        endColor = [.68, .85, .9]; % Light-blue
    end
    cmap = GradColor(startColor, endColor, numel(etavec));
    subplot(4, 2, 2+i); hold on;
    Med = squeeze(DNPStats(i, 1, 2,:))';
    Qntls = squeeze(DNPStats(i, 1, [1,3],:));
    fill([xval, fliplr(xval)], [Qntls(1,:), fliplr(Qntls(2,:))], 'k', 'FaceAlpha', 0.3, 'EdgeColor', 'none');
    plot(xval, Med, 'k-', "LineWidth",1);
    for ti = 1:numel(etavec)
        Qntls = squeeze(SVStats(i, ti, [1,3],:));
        plot(xval, Qntls(1,:), '-', 'Color', cmap(ti,:));
        plot(xval, Qntls(2,:), '-', 'Color', cmap(ti,:));
    end
    plot(xval, ones(size(xval))*0,'k--');
    xlim([0, .8]);
    ylim([-1.2, 5.3]);
    ylabel("V1 - V2 (a.u.)");
    mysavefig(h, filename, plot_dir, 12, [8, 8]);

    subplot(2,2,2+i); hold on;
    for ti = 1:numel(etavec)
        ratio = squeeze(Ratios(i, ti, :))';
        plot(xval, ratio*100, '.-', 'Color', cmap(ti,:));
    end
    xlabel('Scaled V3');
    ylabel('% Correct (V1 & V2)');
    mysavefig(h, filename, plot_dir, 12, [8, 8]);
end

% %% Simulations - based on the trends of V3 noise
% M = .1;
% w = 1;
% sL = .0000;
% lL = .05;
% eps = .1;
% mode = 'absorb';
% mtmodel = [];
% for t = [10, 1.5] % low, high
%     if t == 10
%         x = [M, 1, sL];
%     elseif t == 1.5
%         x = [M, 1, lL];
%     end
%     for v = [1, 0] % vague, precise
%         sect = Sldwndw(Sldwndw.TimeConstraint == t & Sldwndw.Vaguenesscode == v,:);
%         dat = sect(:,{'TimeConstraint','Vaguenesscode','V3scld', 'sdV3scld', 'choice'});
%         dat.Properties.VariableNames = {'TimeConstraint','Vaguenesscode','V3', 'sdV3', 'chosenItem'};
%         dat.V1 = 1.05*ones(size(dat.V3));
%         dat.V2 = 1.1533*ones(size(dat.V3));
%         dat.sdV1 = eps*dat.V1;
%         dat.sdV2 = eps*dat.V2;
%         probs = dDN(x, dat, mode);
%         ratio = probs(2,:)./sum(probs(1:2,:),1);
%         dat.modelprob1 = probs(1,:)';
%         dat.modelprob2 = probs(2,:)';
%         dat.modelprob3 = probs(3,:)';
%         mtmodel = [mtmodel; dat];
%     end
% end
% mtmodel.ratio = mtmodel.modelprob2./(mtmodel.modelprob1 + mtmodel.modelprob2);
% %
% h = figure;
% hold on;
% vi = 0;
% for t = [10, 1.5] % low, high
%     for v = [1, 0] % vague, precise
%         vi = vi + 1;
%         dat = mtmodel(mtmodel.TimeConstraint == t & mtmodel.Vaguenesscode == v,:);
%         plot(dat.V3, dat.ratio, '-', 'Color', colorpalette{vi}, 'LineWidth', 2);
%         %fill([dat.V3 fliplr(dat.V3)], [ratio-ratiose fliplr(ratio+ratiose)], rgbMatrix(vi,:), 'FaceAlpha', 0.3, 'EdgeColor', 'none');
%     end
% end
% xlim([0,1]);
% xlabel('Scaled V3');
% ylabel('% Correct (V1 & V2)');
% mysavefig(h, sprintf('Ratio_ModelPredict_MatchOnlyV3_%s', Treatment), plot_dir, 12, [4, 4]);
% %% Simulation, Mixed noise, early decrease and late increase
% Test = 'Mixed';
% mycolors = jet(8);
% V1mean = 150;
% V2mean = 158;
% epsvec = linspace(0,13,8);
% epsvec = linspace(5,36,8);
% etavec = linspace(3, 0, 8);
% epsV1 = 8;
% epsV2 = 8;
% h = figure; hold on;
% for frame = 1:numel(epsvec)
%     eps = epsvec(frame);
%     eta = etavec(frame);
%     eta = 2;
%     V3 = [0:(V2mean - 1)]';
%     V1 = V1mean*ones(size(V3));
%     V2 = V2mean*ones(size(V3));
%     sdV1 = epsV1*ones(size(V3));
%     sdV2 = epsV2*ones(size(V3));
%     sdV3 = eps*ones(size(V3));
%     dat = table(V1,V2,V3,sdV1,sdV2,sdV3);
%     %
%     x = [1, 1, eta];
%     probs = dDN(x, dat, 'absorb');
%     ratio = probs(2,:)./(probs(1,:) + probs(2,:));
%     plot(V3, ratio, '.-', 'Color',mycolors(frame,:));
% end

%% functions
function cmap = GradColor(startColor, endColor, numColors)
% Generate the colormap
cmap = zeros(numColors, 3); % Initialize the colormap matrix
for i = 1:3
    cmap(:, i) = linspace(startColor(i), endColor(i), numColors);
end
end
%%
function [probs, nll, dDNPQntls, V3Qntls, dSVQtls] = dDN(x, dat, mode) % cut inputs, independent
% set the lower boundary for every input value distribution as zero
% samples in the denominator are independent from the numerator
% the SIGMA term in the denominator will be natually non-negative after that.
if gpuDeviceCount > 0
    gpuparallel = 1;
else
    gpuparallel = 0;
end
M = x(1);
w = x(2);
eta = x(3);
Rmax = 75;
data = dat(:, {'V1', 'V2', 'V3', 'sdV1','sdV2','sdV3', 'chosenItem'});
num_samples = 5e6;
Ntrl = size(dat,1);
if strcmp(mode, 'absorb')
    samples = nan([3,num_samples, Ntrl]);
    for ci = 1:3
        if gpuparallel
            values = gpuArray(data.(['V',num2str(ci)])');
            stds = gpuArray(data.(['sdV', num2str(ci)])');
            samples(ci,:,:) = max(gpuArray.randn([num_samples, Ntrl]).*stds + repmat(values, num_samples, 1), 0);
        else
            values = data.(['V',num2str(ci)])';
            stds = data.(['sdV', num2str(ci)])';
            samples(ci,:,:) = max(randn([num_samples, Ntrl]).*stds + repmat(values, num_samples, 1), 0);
        end
    end
    D1 = nan([3,num_samples, Ntrl]);
    D2 = D1;
    D3 = D1;
    for ci = 1:3
        if gpuparallel
            values = gpuArray(data.(['V',num2str(ci)])');
            stds = gpuArray(data.(['sdV', num2str(ci)])');
            D1(ci,:,:) = max(gpuArray.randn([num_samples, Ntrl]).*stds + repmat(values, num_samples, 1), 0);
            D2(ci,:,:) = max(gpuArray.randn([num_samples, Ntrl]).*stds + repmat(values, num_samples, 1), 0);
            D3(ci,:,:) = max(gpuArray.randn([num_samples, Ntrl]).*stds + repmat(values, num_samples, 1), 0);
        else
            values = data.(['V',num2str(ci)])';
            stds = data.(['sdV', num2str(ci)])';
            D1(ci,:,:) = max(randn([num_samples, Ntrl]).*stds + repmat(values, num_samples, 1), 0);
            D2(ci,:,:) = max(randn([num_samples, Ntrl]).*stds + repmat(values, num_samples, 1), 0);
            D3(ci,:,:) = max(randn([num_samples, Ntrl]).*stds + repmat(values, num_samples, 1), 0);
        end
    end
elseif strcmp(mode, 'cutoff')
    error('The cutoff boundary algorithm has not been developped yet.');
end
D1 = sum(D1, 1)*w + M;
D2 = sum(D2, 1)*w + M;
D3 = sum(D3, 1)*w + M;
D = [D1; D2; D3];
clear D1 D2 D3;
% The product of divisive normalization before adding late noise
DNP = Rmax*samples./D;

% The final decision variables (subjective values) with late noise
if gpuparallel
    SVs = DNP + gpuArray.randn(size(samples))*eta;
    choice = gpuArray(data.chosenItem');
else
    SVs = DNP + randn(size(samples))*eta;
    choice = data.chosenItem';
end
max_from_each_distribution = SVs == max(SVs, [], 1);
probs = squeeze(sum(max_from_each_distribution, 2) / size(SVs, 2));
nll = -sum(log(max(probs(sub2ind(size(probs), choice, 1:size(probs, 2))), eps)));
dDNPQntls = quantile(squeeze(DNP(2,:,:) - DNP(1,:,:)), 3, 1);
V3Qntls = quantile(squeeze(DNP(3,:,:)), [.05, .5, .95], 1);
dSVQtls = quantile(squeeze(SVs(2,:,:) - SVs(1,:,:)), 3, 1);
if gpuparallel
    probs = gather(probs);
    nll = gather(nll);
    dDNPQntls = gather(dDNPQntls);
    V3Qntls = gather(V3Qntls);
    dSVQtls = gather(dSVQtls);
end
end

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
%%
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
