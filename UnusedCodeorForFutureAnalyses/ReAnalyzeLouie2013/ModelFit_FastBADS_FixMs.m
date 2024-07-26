%% define directoriress
% Switch to the working directory
% rtdir = '/Users/bs3667/Noise/UnusedCodeorForFutureAnalyses/ReAnalyzeLouie2013';
rtdir = '/gpfs/data/glimcherlab/BoShen/Noise/UnusedCodeorForFutureAnalyses/ReAnalyzeLouie2013';
cd(rtdir);

% Define I/O directories
datadir = rtdir;
svdir = fullfile(rtdir, 'Results');
if ~exist(svdir, 'dir')
    mkdir(svdir);
end
AnalysName = 'FastBADS_FixMs';
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
options.Display = 'final';
options.UncertaintyHandling = true;    %s Function is stochastic
options.NoiseFinalSamples = 30;
mode = 'absorb';
Rslts = table('Size', [0 9], 'VariableTypes', {'double', 'double', 'string', 'double', 'double', 'double', 'double', 'logical', 'uint16'},...
    'VariableNames', {'subID', 'modeli', 'name', 'Mp', 'wp', 'nll', 'nllsd', 'success', 'iterations'});
testfile = fullfile(svdir, AnalysName, 'AllRslts.txt');
fp = fopen(testfile, 'w+');
fprintf(fp, '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n', ...
    'subID', 'Model', 'randi', 'Mp0', 'wp0', 'Mp', 'wp', 'nll', 'nllsd', 'success', 'iterations');
fclose(fp);
Npar = 40;
mypool = parpool(Npar);
sublist = unique(mt.subID);
for subj = 1:numel(sublist)
    %%
    fprintf('Subject %d:\n', subj);
    dat = mt(mt.subID == sublist(subj), :);
    for modeli = 1:5
        switch modeli
            case 1
                nLLfunc = @(x) McFadden(x, dat);
                name = 'McFadden';
            case 2
                nLLfunc = @(x) Mdl2(x, dat);
                name = 'LinearDistrb';
            case 3
                nLLfunc = @(x) DN(x, dat);
                name = 'DN'; %, cut input, independent';
            case 4
                nLLfunc = @(x) dDNb(x, dat, mode);
                name = 'dDNb'; %, cut input, independent';
            case 5
                nLLfunc = @(x) dDNd(x, dat, mode);
                name = 'dDNd'; %, cut SIGMA, independent';
        end
        fprintf('\tModel %i\n', modeli);

        if modeli <= 2 % the only parameter is eta
            LB = [0]; % [Mp]
            UB = [1000];
            PLB = [1];
            PUB = [100];
        else
            LB = [0, 0]; % [Mp, wp]
            UB = [1000, 4];
            PLB = [1, 0];
            PUB = [50, 1.4];
        end
        nLL = [];
        params = {};
        success = [];
        res = {};
        parfor i = 1:Npar
            x0 = PLB + (PUB - PLB) .* rand(size(PLB));
            [xOpt,fval,exitflag,output] = bads(nLLfunc,x0,LB,UB,PLB,PUB,[],options);
            if modeli <= 2
                dlmwrite(testfile, [subj, modeli, i, x0, NaN, xOpt, NaN, fval, output.fsd, exitflag, output.iterations],'delimiter','\t','precision','%.6f','-append');
            elseif modeli >= 3
                dlmwrite(testfile, [subj, modeli, i, x0, xOpt, fval, output.fsd, exitflag, output.iterations],'delimiter','\t','precision','%.6f','-append');
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
        filename = fullfile(mtrxdir, sprintf('Subj%02i_Mdl%i.mat', subj, modeli));
        save(filename, 'xOpt', 'fval', 'exitflag', 'output');
        if modeli <= 2
            new_row = table(subj, modeli, {name}, xOpt(1), NaN, fval, output.fsd, exitflag, output.iterations, 'VariableNames', Rslts.Properties.VariableNames);
        else
            new_row = table(subj, modeli, {name}, xOpt(1), xOpt(2), fval, output.fsd, exitflag, output.iterations, 'VariableNames', Rslts.Properties.VariableNames);
        end
        Rslts = [Rslts; new_row];
        writetable(Rslts, fullfile(outdir, 'BestRslts.txt'), 'Delimiter', '\t');
        fprintf('nll = %f\n', fval);
    end
end

%% 
function nll = McFadden(x, dat)
if gpuDeviceCount > 0
    gpuparallel = 1;
else
    gpuparallel = 0;
end
Mp = x(1); % change M to be M', absorbing the magnitude of late noise
eta = 1; % after the transformation, the late noise term is standardized as 1
data = dat(:, {'V1', 'V2', 'V3', 'sdV1','sdV2','sdV3','chosenItem'});
num_samples = 20000;
samples = [];
for ci = 1:3
    if gpuparallel
        values = gpuArray(data.(['V',num2str(ci)])');
    else
        values = data.(['V',num2str(ci)])';
    end
    samples(:,:,ci) = repmat(values, num_samples, 1);
end
if gpuparallel
    SVs = samples/Mp + gpuArray.randn(size(samples))*eta;
    choice = gpuArray(data.chosenItem');
else
    SVs = samples/Mp + randn(size(samples))*eta;
    choice = data.chosenItem';
end
max_from_each_distribution = SVs == max(SVs, [], 3);
probs = squeeze(sum(max_from_each_distribution, 1) / size(SVs, 1));
nll = -sum(log(max(probs(sub2ind(size(probs), 1:size(probs, 1), choice)), eps)));
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
Mp = x(1); % change M to be M', absorbing the magnitude of late noise
eta = 1; % after the transformation, the late noise term is standardized as 1
data = dat(:, {'V1', 'V2', 'V3', 'sdV1','sdV2','sdV3','chosenItem'});
num_samples = 20000;
Ntrl = size(dat,1);
samples = [];
for ci = 1:3
    if gpuparallel
        values = gpuArray(data.(['V',num2str(ci)])');
        stds = gpuArray(data.(['sdV', num2str(ci)])');
        samples(:,:,ci) = gpuArray.randn([num_samples, Ntrl]).*stds + repmat(values, num_samples, 1);
    else
        values = data.(['V',num2str(ci)])';
        stds = data.(['sdV', num2str(ci)])';
        samples(:,:,ci) = randn([num_samples, Ntrl]).*stds + repmat(values, num_samples, 1);
    end
end
if gpuparallel
    SVs = samples/Mp + gpuArray.randn(size(samples))*eta;
    choice = gpuArray(data.chosenItem');
else
    SVs = samples/Mp + randn(size(samples))*eta;
    choice = data.chosenItem';
end
max_from_each_distribution = SVs == max(SVs, [], 3);
probs = squeeze(sum(max_from_each_distribution, 1) / size(SVs, 1));
nll = -sum(log(max(probs(sub2ind(size(probs), 1:size(probs, 1), choice)), eps)));
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
Mp = x(1); % change M to be M', absorbing the magnitude of late noise
eta = 1; % after the transformation, the late noise term is standardized as 1
wp = x(2); % do the same transformation on w
data = dat(:, {'V1', 'V2', 'V3', 'sdV1','sdV2','sdV3','chosenItem'});
num_samples = 20000;
samples = [];
for ci = 1:3
    if gpuparallel
        values = gpuArray(data.(['V',num2str(ci)])');
    else
        values = data.(['V',num2str(ci)])';
    end
    samples(:,:,ci) = repmat(values, num_samples, 1);
end
% The product of divisive normalization before adding late noise
DNP = samples./(sum(samples, 3)*wp + Mp);
% adding late noise
if gpuparallel
    SVs = DNP + gpuArray.randn(size(samples))*eta;
    choice = gpuArray(data.chosenItem');
else
    SVs = DNP + randn(size(samples))*eta;
    choice = data.chosenItem';
end
max_from_each_distribution = SVs == max(SVs, [], 3);
probs = squeeze(sum(max_from_each_distribution, 1) / size(SVs, 1));
nll = -sum(log(max(probs(sub2ind(size(probs), 1:size(probs, 1), choice)), eps)));
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
Mp = x(1); % change M to be M', absorbing the magnitude of late noise
eta = 1; % after the transformation, the late noise term is standardized as 1
wp = x(2); % do the same transformation on w
scl = 1; % scaling parameter on the early noise
data = dat(:, {'V1', 'V2', 'V3', 'sdV1','sdV2','sdV3','chosenItem'});
num_samples = 20000;
Ntrl = size(dat,1);
if strcmp(mode, 'absorb')
    samples = [];
    for ci = 1:3
        if gpuparallel
            values = gpuArray(data.(['V',num2str(ci)])');
            stds = gpuArray(data.(['sdV', num2str(ci)])')*scl;
            samples(:,:,ci) = max(gpuArray.randn([num_samples, Ntrl]).*stds + repmat(values, num_samples, 1), 0);
        else
            values = data.(['V',num2str(ci)])';
            stds = data.(['sdV', num2str(ci)])'*scl;
            samples(:,:,ci) = max(randn([num_samples, Ntrl]).*stds + repmat(values, num_samples, 1), 0);
        end
    end
    D1 = [];
    D2 = [];
    D3 = [];
    for ci = 1:3
        if gpuparallel
            values = gpuArray(data.(['V',num2str(ci)])');
            stds = gpuArray(data.(['sdV', num2str(ci)])')*scl;
            D1(:,:,ci) = max(gpuArray.randn([num_samples, Ntrl]).*stds + repmat(values, num_samples, 1), 0);
            D2(:,:,ci) = max(gpuArray.randn([num_samples, Ntrl]).*stds + repmat(values, num_samples, 1), 0);
            D3(:,:,ci) = max(gpuArray.randn([num_samples, Ntrl]).*stds + repmat(values, num_samples, 1), 0);
        else
            values = data.(['V',num2str(ci)])';
            stds = data.(['sdV', num2str(ci)])'*scl;
            D1(:,:,ci) = max(randn([num_samples, Ntrl]).*stds + repmat(values, num_samples, 1), 0);
            D2(:,:,ci) = max(randn([num_samples, Ntrl]).*stds + repmat(values, num_samples, 1), 0);
            D3(:,:,ci) = max(randn([num_samples, Ntrl]).*stds + repmat(values, num_samples, 1), 0);
        end
    end
elseif strcmp(mode, 'cutoff')
    error('The cutoff boundary algorithm has not been developped yet.');
end
D1 = sum(D1, 3)*wp + Mp;
D2 = sum(D2, 3)*wp + Mp;
D3 = sum(D3, 3)*wp + Mp;
% The product of divisive normalization before adding late noise
DNP = samples./cat(3, D1, D2, D3);
clear D1 D2 D3;
if gpuparallel
    SVs = DNP + gpuArray.randn(size(samples))*eta;
    choice = gpuArray(data.chosenItem');
else
    SVs = DNP + randn(size(samples))*eta;
    choice = data.chosenItem';
end
max_from_each_distribution = SVs == max(SVs, [], 3);
probs = squeeze(sum(max_from_each_distribution, 1) / size(SVs, 1));
nll = -sum(log(max(probs(sub2ind(size(probs), 1:size(probs, 1), choice)), eps)));
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
Mp = x(1); % change M to be M', absorbing the magnitude of late noise
eta = 1; % after the transformation, the late noise term is standardized as 1
wp = x(2); % do the same transformation on w
scl = 1; % scaling parameter on the early noise
data = dat(:, {'V1', 'V2', 'V3', 'sdV1','sdV2','sdV3','chosenItem'});
num_samples = 20000;
Ntrl = size(dat,1);
samples = [];
for ci = 1:3
    if gpuparallel
        values = gpuArray(data.(['V',num2str(ci)])');
        stds = gpuArray(data.(['sdV', num2str(ci)])')*scl;
        samples(:,:,ci) = gpuArray.randn([num_samples, Ntrl]).*stds + repmat(values, num_samples, 1);
    else
        values = data.(['V',num2str(ci)])';
        stds = data.(['sdV', num2str(ci)])'*scl;
        samples(:,:,ci) = randn([num_samples, Ntrl]).*stds + repmat(values, num_samples, 1);
    end
end
D1 = [];
D2 = [];
D3 = [];
for ci = 1:3
    if gpuparallel
        values = gpuArray(data.(['V',num2str(ci)])');
        stds = gpuArray(data.(['sdV', num2str(ci)])')*scl;
        D1(:,:,ci) = gpuArray.randn([num_samples, Ntrl]).*stds + repmat(values, num_samples, 1);
        D2(:,:,ci) = gpuArray.randn([num_samples, Ntrl]).*stds + repmat(values, num_samples, 1);
        D3(:,:,ci) = gpuArray.randn([num_samples, Ntrl]).*stds + repmat(values, num_samples, 1);
    else
        values = data.(['V',num2str(ci)])';
        stds = data.(['sdV', num2str(ci)])'*scl;
        D1(:,:,ci) = randn([num_samples, Ntrl]).*stds + repmat(values, num_samples, 1);
        D2(:,:,ci) = randn([num_samples, Ntrl]).*stds + repmat(values, num_samples, 1);
        D3(:,:,ci) = randn([num_samples, Ntrl]).*stds + repmat(values, num_samples, 1);
    end
end
if strcmp(mode, 'absorb')
    D1 = max(sum(D1, 3),0)*wp + Mp;
    D2 = max(sum(D2, 3),0)*wp + Mp;
    D3 = max(sum(D3, 3),0)*wp + Mp;
elseif strcmp(mode, 'cutoff')
    error('The cutoff boundary algorithm has not been developped yet.');
end
% The product of divisive normalization before adding late noise
DNP = samples./cat(3, D1, D2, D3);
clear D1 D2 D3;
if gpuparallel
    SVs = DNP + gpuArray.randn(size(samples))*eta;
    choice = gpuArray(data.chosenItem');
else
    SVs = DNP + randn(size(samples))*eta;
    choice = data.chosenItem';
end
max_from_each_distribution = SVs == max(SVs, [], 3);
probs = squeeze(sum(max_from_each_distribution, 1) / size(SVs, 1));
nll = -sum(log(max(probs(sub2ind(size(probs), 1:size(probs, 1), choice)), eps)));
if gpuparallel
    nll = gather(nll);
end
end

