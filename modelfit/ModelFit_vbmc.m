%% define directoriress
% Switch to the working directory
rtdir = '/Users/bs3667/Noise/modelfit';
% rtdir = '/gpfs/data/glimcherlab/BoShen/Noise/modelfit';
cd(rtdir);

% Define I/O directories
datadir = rtdir;
svdir = fullfile(rtdir, 'Results');
if ~exist(svdir, 'dir')
    mkdir(svdir);
end
% load packages
addpath(genpath('/Users/bs3667/Library/vbmc'));
%% load data
load(fullfile(datadir, 'TrnsfrmData.mat'));
%disp(mt);
sublist = unique(mt.subID);
%% Maximum likelihood fitting to the choice behavior
AnalysName = 'vbmc_Mtlb';
if ~exist(fullfile(svdir, AnalysName), 'dir')
    mkdir(fullfile(svdir, AnalysName));
end
dat = mt(mt.subID == sublist(6), :);
LB = [0, -1];
UB = [1000, 1];
PLB = [1.4, 0.1];
PUB = [100, 0.5];
nLLfunc = @(x) McFadden(x, dat);
name = 'DN';
lpriorfun = @(x) msplinetrapezlogpdf(x,LB,PLB,PUB,UB);
fun = @(x) lpostfun(x,nLLfunc,lpriorfun);     % Log joint

options = vbmc('defaults');
options.Plot = true;        % Plot iterations
options.SpecifyTargetNoise = true;  % Noisy function evaluations
% Run VBMC
x0 = 0.5*(PLB+PUB);
[vp,elbo,elbo_sd] = vbmc(fun,x0,LB,UB,PLB,PUB,options);

%%

options = bads('defaults');     % Default options
options.Display = 'None';
options.UncertaintyHandling = true;    %s Function is stochastic
options.NoiseFinalSamples = 30;

Rslts = table('Size', [0 9], 'VariableTypes', {'double', 'string', 'double', 'double', 'double', 'double', 'double', 'logical', 'double'}, 'VariableNames', {'subID', 'Model', 'eta', 'Mp', 'wp', 'nll', 'nllsd', 'success', 'iterations'});
testfile = fullfile(svdir, AnalysName, 'Rslts_vbmc_Test.txt');
fp = fopen(testfile, 'w+');
fprintf(fp, '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n', 'subID', 'Model', 'randi', 'eta', 'Mp', 'wp', 'nll', 'nllsd', 'success', 'iterations');
fclose(fp);
Npar = 40;
mypool = parpool(Npar);
sublist = unique(mt.subID);
subj = 1;
while subj <= length(sublist)
    fprintf('Subject %d:\t', subj);
    dat = mt(mt.subID == sublist(subj), :);
    for modeli = 1:7
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
        fprintf('Model %s nll=', model);
        modeldir = fullfile(svdir, AnalysName, sprintf('Model%s', model));
        if ~exist(modeldir, 'dir')
            mkdir(modeldir);
        end
        if modeli <= 2
            LB = 0;
            UB = 1000;
            PLB = 1.4;
            PUB = 100;
        else
            LB = [0, -1];
            UB = [1000, 1];
            PLB = [1.4, 0.1];
            PUB = [100, 0.5];
        end
        nLL = [];
        params = {};
        success = [];
        res = {};
        parfor i = 1:Npar
            x0 = PLB + (PUB - PLB) .* rand(size(PLB));
            [xOpt,fval,exitflag,output] = bads(nLLfunc,x0,LB,UB,PLB,PUB,[],options);
            if modeli <= 2
                dlmwrite(testfile, [subj, modeli, i, xOpt, NaN, NaN, fval, output.fsd, exitflag, output.iterations],'delimiter','\t','precision','%.6f','-append');
                %fprintf(fp, '%i\t%s\t%f\t%f\t%f\t%f\t%f\t%f\t%i\t%i\n', subj, model, i, xOpt, NaN, NaN, fval, output.fsd, exitflag, output.iterations);
            elseif modeli >= 3
                dlmwrite(testfile, [subj, modeli, i, 1, xOpt(1), xOpt(2), fval, output.fsd, exitflag, output.iterations],'delimiter','\t','precision','%.6f','-append');
                %fprintf(fp, '%i\t%s\t%f\t%f\t%f\t%f\t%f\t%f\t%i\t%i\n', subj, model, i, 1, xOpt(1), xOpt(2), fval, output.fsd, exitflag, output.iterations);
            end
            params{i} = xOpt;
            nLL(i) = fval;
            success(i) = exitflag;
            res{i} = output;
        end
        besti = find(nLL == min(nLL));
        xOpt = params{besti};
        fval = nLL(besti);
        exitflag = success(besti);
        output = res{besti};
        filename = fullfile(modeldir, sprintf('BADS_subj%02i.mat', subj));
        save(filename, 'xOpt', 'fval', 'exitflag', 'output');
        if modeli <= 2
            new_row = table(subj, {model}, xOpt, NaN, NaN, fval, output.fsd, exitflag, output.iterations, 'VariableNames', Rslts.Properties.VariableNames);
        elseif modeli >= 3
            new_row = table(subj, {model}, 1, xOpt(1), xOpt(2), fval, output.fsd, exitflag, output.iterations, 'VariableNames', Rslts.Properties.VariableNames);
        end
        Rslts = [Rslts; new_row];
        writetable(Rslts, fullfile(svdir, AnalysName, 'Rslts_BADS_Best.txt'), 'Delimiter', '\t');
        fprintf('%f\t', fval);
    end
    subj = subj + 1;
end
%fclose(fp);

%% 
function nll = McFadden(x, dat)
if gpuDeviceCount > 0
    gpuparallel = 1;
else
    gpuparallel = 0;
end
eta = x(1);
data = dat(:, {'V1', 'V2', 'V3', 'chosenItem'});
num_samples = 20000;
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
num_samples = 20000;
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
num_samples = 20000;
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
num_samples = 20000;
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
num_samples = 20000;
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
num_samples = 20000;
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
num_samples = 20000;
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

