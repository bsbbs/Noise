% Code for Grid search visualization
% rtdir = '/Users/bs3667/Noise/modelfit';
rtdir = '/gpfs/data/glimcherlab/BoShen/Noise/modelfit';
cd(rtdir);
addpath('../utils');
datadir = rtdir;
svdir = fullfile(rtdir, 'Results');
if ~exist(svdir, 'dir')
    mkdir(svdir);
end
AnalysName = 'GridSearch_Mtlb';
outdir = fullfile(svdir, AnalysName);
if ~exist(outdir, 'dir')
    mkdir(outdir);
end
%% load data
load(fullfile(datadir, 'TrnsfrmData.mat'));
% disp(head(mt));
%%
mode = 'absorb';
eta = linspace(-20,20,41); % range of eta
wp = linspace(-1,1,21);
Mp = linspace(-20,20,41); % range of Mp and wp
sublist = unique(mt.subID);
% Rslts = table('Size', [0 6], 'VariableTypes', {'double', 'string', 'double', 'double', 'double', 'double'}, 'VariableNames', {'subID', 'Model', 'eta', 'Mp', 'wp', 'nll'});
testfile = fullfile(svdir, AnalysName, 'Rslts_GrdSrch.txt');
fp = fopen(testfile, 'w+');
fprintf(fp, '%s\t%s\t%s\t%s\t%s\t%s\n', 'subID', 'Model', 'eta', 'Mp', 'wp', 'nll');
fclose(fp);
parfor subj = 1:length(sublist)
    fprintf('Subj %i ', subj);
    dat = mt(mt.subID == sublist(subj), :);
    h = figure;
    for modeli = 1:7
        fprintf('Mdl%i ', modeli);
        % if mod(subj-1, 12) == 0
        %     h(modeli) = figure;
        %     fprintf('\n');
        % end
        switch modeli
            case 1
                nLLfunc = @(x) McFadden(x, dat);
                name = 'McFadden';
            case 2
                nLLfunc = @(x) Mdl2(x, dat);
                name = 'Model2';
            case 3
                nLLfunc = @(x) DN(x, dat);
                name = 'DN';
            case 4
                nLLfunc = @(x) dDNa(x, dat, mode);
                name = 'dDNa, cut input, dependent';
            case 5
                nLLfunc = @(x) dDNb(x, dat, mode);
                name = 'dDNb, cut input, independent';
            case 6
                nLLfunc = @(x) dDNc(x, dat, mode);
                name = 'dDNc, cut SIGMA, dependent';
            case 7
                nLLfunc = @(x) dDNd(x, dat, mode);
                name = 'dDNd, cut SIGMA, independent';
        end
        subplot(3,3,modeli);
        if modeli <= 2
            nlls = [];
            for i = 1:numel(eta)
                nlls(i) = nLLfunc(eta(i));
            end
            plot(eta, nlls, '-');
            xlabel('\eta');
            ylabel('nLL');
            xOpt = eta(nlls == min(nlls));
            fval = min(nlls);
        elseif modeli >= 3
            nlls = [];
            [X, Y] = meshgrid(wp, Mp);
            for k = 1:numel(X)
                nlls(k) = nLLfunc([Y(k), X(k)]);
            end
            [minVal, Idx] = min(nlls);
            Mpbst = Y(Idx);
            wpbst = X(Idx);
            xOpt = [Mpbst, wpbst];
            fval = minVal;
            nlls = reshape(nlls, size(X));
            surf(X, Y, nlls);
            hold on;
            contour(X, Y, nlls, 'c');
            plot3(wpbst, Mpbst, minVal+30, 'm.', "MarkerSize", 18);
            view(0,90);
            xlabel('wp');
            ylabel('Mp');
            zlabel('nLL');
        end
        title(sprintf('SubID %i\n%s',sublist(subj), name));
        filename = sprintf('Gridsrch_Subj%02i', subj);
        mysavefig(h, filename, outdir, 14, [8,11]);
        if modeli <= 2
            dlmwrite(testfile, [subj, modeli, xOpt, NaN, NaN, fval],'delimiter','\t','precision','%d%i%.6f%.6f%.6f%.6f','-append');
        elseif modeli >= 3
            dlmwrite(testfile, [subj, modeli, 1, xOpt(1), xOpt(2), fval],'delimiter','\t','precision','%d%i%.6f%.6f%.6f%.6f','-append');
        end
        % save(fullfile(outdir, [filename, '.mat']), 'nlls', 'modeli', 'eta', 'Mp', 'wp');
        % if modeli <= 2
        %     new_row = table(subj, {name}, xOpt, NaN, NaN, fval, 'VariableNames', Rslts.Properties.VariableNames);
        % elseif modeli >= 3
        %     new_row = table(subj, {name}, 1, xOpt(1), xOpt(2), fval, 'VariableNames', Rslts.Properties.VariableNames);
        % end
        % Rslts = [Rslts; new_row];
        % writetable(Rslts, fullfile(svdir, AnalysName, 'Rslts_GrdSrch_Best.txt'), 'Delimiter', '\t');
    end
end

function nll = McFadden(x, dat)
if gpuDeviceCount > 0
    gpuparallel = 1;
else
    gpuparallel = 0;
end
eta = x(1);
data = dat(:, {'V1', 'V2', 'V3', 'chosenItem'});
num_samples = 100000;
Ntrl = size(dat,1);
samples = [];
for ci = 1:3
    if gpuparallel
        values = gpuArray(data.(['V',num2str(ci)])');
    else
        values = data.(['V',num2str(ci)])';
    end
    samples(ci,:,:) = repmat(values, num_samples, 1);
end
% assume subjective values no cutoff at zeros
% unrealistic in biology but probably capture the probability of
% potential floating
if gpuparallel
    SVs = samples/eta + gpuArray.randn(size(samples))*(2^-0.5);
    choice = gpuArray(data.chosenItem');
else
    SVs = samples/eta + randn(size(samples))*(2^-0.5);
    choice = data.chosenItem';
end
max_from_each_distribution = SVs == max(SVs, [], 1);
probs = squeeze(sum(max_from_each_distribution, 2) / size(SVs, 2));
nll = -sum(log(max(probs(sub2ind(size(probs), choice, 1:size(probs, 2))), eps)));
if gpuparallel
    nll = gather(nll);
end
end

function nll = Mdl2(x, dat)
if gpuDeviceCount > 0
    gpuparallel = 1;
else
    gpuparallel = 0;
end
eta = x(1);
data = dat(:, {'V1', 'V2', 'V3', 'sdV1','sdV2','sdV3','chosenItem'});
num_samples = 100000;
Ntrl = size(dat,1);
samples = [];
for ci = 1:3
    if gpuparallel
        values = gpuArray(data.(['V',num2str(ci)])');
        stds = gpuArray(data.(['sdV', num2str(ci)])');
        samples(ci,:,:) = gpuArray.randn([num_samples, Ntrl]).*stds + repmat(values, num_samples, 1);
    else
        values = data.(['V',num2str(ci)])';
        stds = data.(['sdV', num2str(ci)])';
        samples(ci,:,:) = randn([num_samples, Ntrl]).*stds + repmat(values, num_samples, 1);
    end
end
% assume subjective values no cutoff at zeros
% unrealistic in biology but probably capture the probability of
% potential floating
if gpuparallel
    SVs = samples/eta + gpuArray.randn(size(samples))*(2^-0.5);
    choice = gpuArray(data.chosenItem');
else
    SVs = samples/eta + randn(size(samples))*(2^-0.5);
    choice = data.chosenItem';
end
max_from_each_distribution = SVs == max(SVs, [], 1);
probs = squeeze(sum(max_from_each_distribution, 2) / size(SVs, 2));
nll = -sum(log(max(probs(sub2ind(size(probs), choice, 1:size(probs, 2))), eps)));
if gpuparallel
    nll = gather(nll);
end
end

function nll = DN(x, dat)
if gpuDeviceCount > 0
    gpuparallel = 1;
else
    gpuparallel = 0;
end
Mp = x(1);
wp = x(2);
data = dat(:, {'V1', 'V2', 'V3', 'chosenItem'});
num_samples = 100000;
Ntrl = size(dat,1);
samples = [];
for ci = 1:3
    if gpuparallel
        values = gpuArray(data.(['V',num2str(ci)])');
    else
        values = data.(['V',num2str(ci)])';
    end
    samples(ci,:,:) = repmat(values, num_samples, 1);
end
D = sum(samples, 1)*wp + Mp;
% assume subjective values no cutoff at zeros
% unrealistic in biology but probably capture the probability of
% potential floating
if gpuparallel
    SVs = samples./D + gpuArray.randn(size(samples))*(2^-0.5);
    choice = gpuArray(data.chosenItem');
else
    SVs = samples./D + randn(size(samples))*(2^-0.5);
    choice = data.chosenItem';
end
max_from_each_distribution = SVs == max(SVs, [], 1);
probs = squeeze(sum(max_from_each_distribution, 2) / size(SVs, 2));
nll = -sum(log(max(probs(sub2ind(size(probs), choice, 1:size(probs, 2))), eps)));
if gpuparallel
    nll = gather(nll);
end
end

function nll = dDNa(x, dat, mode) % cut inputs, dependent
% set the lower boundary for every input value distribution as zero
% samples in the denominator are the same as the numerator
% the SIGMA term in the denominator is non-negative after that.
if gpuDeviceCount > 0
    gpuparallel = 1;
else
    gpuparallel = 0;
end
Mp = x(1);
wp = x(2);
data = dat(:, {'V1', 'V2', 'V3', 'sdV1','sdV2','sdV3','chosenItem'});
num_samples = 100000;
Ntrl = size(dat,1);
if strcmp(mode, 'absorb')
    samples = [];
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
elseif strcmp(mode, 'cutoff')
    error('The cutoff boundary algorithm has not been developped yet.');
end
D = sum(samples, 1)*wp + Mp;
if gpuparallel
    SVs = samples./D + gpuArray.randn(size(samples))*(2^-0.5);
    choice = gpuArray(data.chosenItem');
else
    SVs = samples./D + randn(size(samples))*(2^-0.5);
    choice = data.chosenItem';
end
max_from_each_distribution = SVs == max(SVs, [], 1);
probs = squeeze(sum(max_from_each_distribution, 2) / size(SVs, 2));
nll = -sum(log(max(probs(sub2ind(size(probs), choice, 1:size(probs, 2))), eps)));
if gpuparallel
    nll = gather(nll);
end
end

function nll = dDNb(x, dat, mode) % cut inputs, independent
% set the lower boundary for every input value distribution as zero
% samples in the denominator are independent from the numerator
% the SIGMA term in the denominator is non-negative after that.
if gpuDeviceCount > 0
    gpuparallel = 1;
else
    gpuparallel = 0;
end
Mp = x(1);
wp = x(2);
data = dat(:, {'V1', 'V2', 'V3', 'sdV1','sdV2','sdV3','chosenItem'});
num_samples = 100000;
Ntrl = size(dat,1);
if strcmp(mode, 'absorb')
    samples = [];
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
    samplesD = [];
    for ci = 1:3
        if gpuparallel
            values = gpuArray(data.(['V',num2str(ci)])');
            stds = gpuArray(data.(['sdV', num2str(ci)])');
            samplesD(ci,:,:) = max(gpuArray.randn([num_samples, Ntrl]).*stds + repmat(values, num_samples, 1), 0);
        else
            values = data.(['V',num2str(ci)])';
            stds = data.(['sdV', num2str(ci)])';
            samplesD(ci,:,:) = max(randn([num_samples, Ntrl]).*stds + repmat(values, num_samples, 1), 0);
        end
    end
elseif strcmp(mode, 'cutoff')
    error('The cutoff boundary algorithm has not been developped yet.');
end
D = sum(samplesD, 1)*wp + Mp;
if gpuparallel
    SVs = samples./D + gpuArray.randn(size(samples))*(2^-0.5);
    choice = gpuArray(data.chosenItem');
else
    SVs = samples./D + randn(size(samples))*(2^-0.5);
    choice = data.chosenItem';
end
max_from_each_distribution = SVs == max(SVs, [], 1);
probs = squeeze(sum(max_from_each_distribution, 2) / size(SVs, 2));
nll = -sum(log(max(probs(sub2ind(size(probs), choice, 1:size(probs, 2))), eps)));
if gpuparallel
    nll = gather(nll);
end
end

function nll = dDNc(x, dat, mode) % cut SIGMA, dependent
% set the lower boundary for the summed SIGMA in the denominator
% but not for the input values in the numerator
% samples in the denominator are the same as the numerator
if gpuDeviceCount > 0
    gpuparallel = 1;
else
    gpuparallel = 0;
end
Mp = x(1);
wp = x(2);
data = dat(:, {'V1', 'V2', 'V3', 'sdV1','sdV2','sdV3','chosenItem'});
num_samples = 100000;
Ntrl = size(dat,1);
samples = [];
for ci = 1:3
    if gpuparallel
        values = gpuArray(data.(['V',num2str(ci)])');
        stds = gpuArray(data.(['sdV', num2str(ci)])');
        samples(ci,:,:) = gpuArray.randn([num_samples, Ntrl]).*stds + repmat(values, num_samples, 1);
    else
        values = data.(['V',num2str(ci)])';
        stds = data.(['sdV', num2str(ci)])';
        samples(ci,:,:) = randn([num_samples, Ntrl]).*stds + repmat(values, num_samples, 1);
    end
end
if strcmp(mode, 'absorb')
    D = max(sum(samples, 1),0)*wp + Mp;
elseif strcmp(mode, 'cutoff')
    error('The cutoff boundary algorithm has not been developped yet.');
end
if gpuparallel
    SVs = samples./D + gpuArray.randn(size(samples))*(2^-0.5);
    choice = gpuArray(data.chosenItem');
else
    SVs = samples./D + randn(size(samples))*(2^-0.5);
    choice = data.chosenItem';
end
max_from_each_distribution = SVs == max(SVs, [], 1);
probs = squeeze(sum(max_from_each_distribution, 2) / size(SVs, 2));
nll = -sum(log(max(probs(sub2ind(size(probs), choice, 1:size(probs, 2))), eps)));
if gpuparallel
    nll = gather(nll);
end
end

function nll = dDNd(x, dat, mode) % cut SIGMA, independent
% set the lower boundary for the summed SIGMA in the denominator
% but not for the input values in the numerator
% samples in the denominator are independent from the numerator
if gpuDeviceCount > 0
    gpuparallel = 1;
else
    gpuparallel = 0;
end
Mp = x(1);
wp = x(2);
data = dat(:, {'V1', 'V2', 'V3', 'sdV1','sdV2','sdV3','chosenItem'});
num_samples = 100000;
Ntrl = size(dat,1);
samples = [];
for ci = 1:3
    if gpuparallel
        values = gpuArray(data.(['V',num2str(ci)])');
        stds = gpuArray(data.(['sdV', num2str(ci)])');
        samples(ci,:,:) = gpuArray.randn([num_samples, Ntrl]).*stds + repmat(values, num_samples, 1);
    else
        values = data.(['V',num2str(ci)])';
        stds = data.(['sdV', num2str(ci)])';
        samples(ci,:,:) = randn([num_samples, Ntrl]).*stds + repmat(values, num_samples, 1);
    end
end
samplesD = [];
for ci = 1:3
    if gpuparallel
        values = gpuArray(data.(['V',num2str(ci)])');
        stds = gpuArray(data.(['sdV', num2str(ci)])');
        samplesD(ci,:,:) = gpuArray.randn([num_samples, Ntrl]).*stds + repmat(values, num_samples, 1);
    else
        values = data.(['V',num2str(ci)])';
        stds = data.(['sdV', num2str(ci)])';
        samplesD(ci,:,:) = randn([num_samples, Ntrl]).*stds + repmat(values, num_samples, 1);
    end
end
if strcmp(mode, 'absorb')
    D = max(sum(samplesD, 1),0)*wp + Mp;
elseif strcmp(mode, 'cutoff')
    error('The cutoff boundary algorithm has not been developped yet.');
end
if gpuparallel
    SVs = samples./D + gpuArray.randn(size(samples))*(2^-0.5);
    choice = gpuArray(data.chosenItem');
else
    SVs = samples./D + randn(size(samples))*(2^-0.5);
    choice = data.chosenItem';
end
max_from_each_distribution = SVs == max(SVs, [], 1);
probs = squeeze(sum(max_from_each_distribution, 2) / size(SVs, 2));
nll = -sum(log(max(probs(sub2ind(size(probs), choice, 1:size(probs, 2))), eps)));
if gpuparallel
    nll = gather(nll);
end
end

