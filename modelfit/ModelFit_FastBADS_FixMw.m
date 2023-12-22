%% define directoriress
% Switch to the working directory
% rtdir = '/Users/bs3667/Noise/modelfit';
rtdir = '/gpfs/data/glimcherlab/BoShen/Noise/modelfit';
cd(rtdir);

% Define I/O directories
datadir = rtdir;
svdir = fullfile(rtdir, 'Results');
if ~exist(svdir, 'dir')
    mkdir(svdir);
end
AnalysName = 'FastBADS_FixMw';
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
% load packages
addpath(genpath('/gpfs/data/glimcherlab/BoShen/bads'));
% addpath(genpath('/Users/bs3667/Dropbox (NYU Langone Health)/Bo Shen Working files/NoiseProject/Modelfit/bads-master'));
%% load data
load(fullfile(datadir, 'TrnsfrmData.mat'));
% blacklist = [22102401, 22102405, 22110306]; % these subjects report they aimed to choose smaller-value items.
% mt = mt(~ismember(mt.subID, blacklist),:);
sublist = unique(mt.subID);
% disp(head(mt));
%% Maximum likelihood fitting to the choice behavior
options = bads('defaults');     % Default options
options.Display = 'None';
options.UncertaintyHandling = true;    %s Function is stochastic
options.NoiseFinalSamples = 30;
mode = 'absorb';

Rslts = table('Size', [0 11], 'VariableTypes', {'double', 'double', 'string', 'double', 'double', 'double', 'double', 'double', 'double', 'logical', 'double'},...
    'VariableNames', {'subID', 'modeli', 'name', 'scl', 'eta', 'Mp', 'wp', 'nll', 'nllsd', 'success', 'iterations'});
testfile = fullfile(svdir, AnalysName, 'Rslts_FastBADS_rndsd.txt');
fp = fopen(testfile, 'w+');
fprintf(fp, '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n', 'subID', 'Model', 'randi', 'scl0', 'eta0', 'Mp0', 'wp0', 'scl', 'eta', 'Mp', 'wp', 'nll', 'nllsd', 'success', 'iterations');
fclose(fp);
Npar = 40;
mypool = parpool(Npar);
sublist = unique(mt.subID);
subj = 1;
while subj <= length(sublist)
    fprintf('Subject %d:\t', subj);
    dat = mt(mt.subID == sublist(subj), :);
    for modeli = 1:2
        switch modeli
            case 1
                nLLfunc = @(x) dDNb(x, dat, mode);
                name = 'dDNb'; %, cut input, independent';
            case 2
                nLLfunc = @(x) dDNd(x, dat, mode);
                name = 'dDNd'; %, cut SIGMA, independent';
        end
        fprintf('Model %i nll=', modeli);

        LB = [.2, 0];
        UB = [2, 2];
        PLB = [.9, .1];
        PUB = [1.1, .4];

        nLL = [];
        params = {};
        success = [];
        res = {};
        parfor i = 1:Npar
            x0 = PLB + (PUB - PLB) .* rand(size(PLB));
            [xOpt,fval,exitflag,output] = bads(nLLfunc,x0,LB,UB,PLB,PUB,[],options);
        
            dlmwrite(testfile, [subj, modeli, i, x0(1), x0(2), 1, 1, xOpt(1), xOpt(2), 1, 1, fval, output.fsd, exitflag, output.iterations],'delimiter','\t','precision','%.6f','-append');
            
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
        filename = fullfile(mtrxdir, sprintf('FastBADS_Subj%02i_Mdl%i.mat', subj, modeli));
        save(filename, 'xOpt', 'fval', 'exitflag', 'output');
        new_row = table(subj, modeli, {name}, xOpt(1), xOpt(2), 1, 1, fval, output.fsd, exitflag, output.iterations, 'VariableNames', Rslts.Properties.VariableNames);
        Rslts = [Rslts; new_row];
        writetable(Rslts, fullfile(outdir, 'Rslts_FastBADS_Best.txt'), 'Delimiter', '\t');
        fprintf('%f\t', fval);
    end
    subj = subj + 1;
end

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

