%% define directoriress
[os, ~, ~] = computer;
if strcmp(os,'MACI64')
    rootdir = '/Users/bs3667/Dropbox (NYU Langone Health)/Bo Shen Working files/NoiseProject';
    Gitdir = '~/Noise';
elseif strcmp(os,'GLNXA64')
    rootdir = '/gpfs/data/glimcherlab/BoShen/NoiseProject';
    % rootdir = '/scratch/bs3667/NoiseProject';
    Gitdir = '~/Noise';
end
addpath(genpath(Gitdir));
datadir = fullfile(Gitdir,'myData');

Fitdir = fullfile(rootdir, 'Modelfit','Pooled');
if ~exist(Fitdir, 'dir')
    mkdir(Fitdir);
end
plotdir = fullfile(Fitdir, 'plot');
if ~exist(plotdir, 'dir')
    mkdir(plotdir);
end
mtrxdir = fullfile(Fitdir, 'Objs');
if ~exist(mtrxdir, 'dir')
    mkdir(mtrxdir);
end
% Switch to the working directory
cd(Fitdir);

%% load data
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
        indvtask.V1 = indvtask.V1/mintrgt;
        indvtask.V2 = indvtask.V2/mintrgt;
        indvtask.V3 = indvtask.V3/mintrgt;
        indvtask.sdV1 = indvtask.sdV1/mintrgt;
        indvtask.sdV2 = indvtask.sdV2/mintrgt;
        indvtask.sdV3 = indvtask.sdV3/mintrgt;
        mtconvert = [mtconvert; indvtask];
    end  
end
mtconvert.choice = mtconvert.chosenItem - 1;
%% Maximum likelihood fitting to the choice behavior
options = bads('defaults');     % Default options
options.Display = 'final';
options.UncertaintyHandling = true;    %s Function is stochastic
options.NoiseFinalSamples = 30;
Rslts = table('Size', [0 10], 'VariableTypes', {'double', 'string', 'double', 'double', 'double', 'double', 'double', 'double', 'logical', 'uint16'},...
    'VariableNames', {'modeli', 'name', 'Mp', 'delta', 'wp', 'scl', 'nll', 'nllsd', 'success', 'iterations'});
testfile = fullfile(Fitdir, 'AllRslts.txt');
if ~exist(testfile, 'file')
    fp = fopen(testfile, 'w+');
    fprintf(fp, '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n', ...
        'Model', 'randi', 'Mp0', 'delta0', 'wp0', 'scl0', 'Mp', 'delta', 'wp', 'scl', 'nll', 'nllsd', 'success', 'iterations');
    fclose(fp);
end
%Myclust = parcluster();
%Npar = Myclust.NumWorkers;
%mypool = parpool(Npar/2);
repetition = 48;
dat = mtconvert;
for modeli = 4
    switch modeli
        case 1
            nLLfunc = @(x) McFadden(x, dat);
            name = 'McFadden';
        case 2
            nLLfunc = @(x) Mdl2(x, dat);
            name = 'dnLinear';
        case 3
            nLLfunc = @(x) DNM(x, dat);
            name = 'DNM'; %, cut input, independent';
        case 4
            nLLfunc = @(x) dnDNM(x, dat);
            name = 'dnDNM'; %, cut input, independent';
    end
    fprintf('\tModel %i\n', modeli);
    filename = fullfile(mtrxdir, sprintf('Mdl%i.mat', modeli));
    if ~exist(filename, 'file')
        % shared parameters for all models
        LB = [0, -2]; % [Mp, delta]
        UB = [1000, 4];
        PLB = [1, -.2];
        PUB = [100, .4];
        % nest other parameters
        if modeli == 2 % nest scaling on early noise for model 2
            LB = [LB, 0]; % [Mp, delta, scl]
            UB = [UB, 4];
            PLB = [PLB, 0];
            PUB = [PUB, 2];
        end
        if modeli == 3 % nest divisive normalization for model 3
            LB = [LB, 0]; % [Mp, delta, wp]
            UB = [UB, 4];
            PLB = [PLB, 0];
            PUB = [PUB, 1.4];
        end
        if modeli == 4 % nest for model 4
            LB = [LB, 0, 0]; % [Mp, delta, wp, scl]
            UB = [UB, 4, 4];
            PLB = [PLB, 0, 0];
            PUB = [PUB, 1.4, 2];
        end

        nLL = [];
        params = {};
        success = [];
        res = {};
        for i = 1:repetition
            x0 = PLB + (PUB - PLB) .* rand(size(PLB));
            [xOpt,fval,exitflag,output] = bads(nLLfunc,x0,LB,UB,PLB,PUB,[],options);
            % 'Mp', 'delta', 'wp', 'scl',
            if modeli == 1
                dlmwrite(testfile, [modeli, i, x0, NaN, NaN, xOpt, NaN, NaN, fval, output.fsd, exitflag, output.iterations],'delimiter','\t','precision','%.6f','-append');
            elseif modeli == 2
                dlmwrite(testfile, [modeli, i, x0(1:2), NaN, x0(3), xOpt(1:2), NaN, xOpt(3), fval, output.fsd, exitflag, output.iterations],'delimiter','\t','precision','%.6f','-append');
            elseif modeli == 3
                dlmwrite(testfile, [modeli, i, x0, NaN, xOpt, NaN, fval, output.fsd, exitflag, output.iterations],'delimiter','\t','precision','%.6f','-append');
            elseif modeli == 4
                dlmwrite(testfile, [modeli, i, x0, xOpt, fval, output.fsd, exitflag, output.iterations],'delimiter','\t','precision','%.6f','-append');
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
        save(filename, 'xOpt', 'fval', 'exitflag', 'output');
    else
        load(filename);
    end
    if modeli == 1
        new_row = table(modeli, {name}, xOpt(1), xOpt(2), NaN, NaN, fval, output.fsd, exitflag, output.iterations, 'VariableNames', Rslts.Properties.VariableNames);
    elseif modeli == 2
        new_row = table(modeli, {name}, xOpt(1), xOpt(2), NaN, xOpt(3), fval, output.fsd, exitflag, output.iterations, 'VariableNames', Rslts.Properties.VariableNames);
    elseif modeli == 3
        new_row = table(modeli, {name}, xOpt(1), xOpt(2), xOpt(3), NaN, fval, output.fsd, exitflag, output.iterations, 'VariableNames', Rslts.Properties.VariableNames);
    elseif modeli == 4
        new_row = table(modeli, {name}, xOpt(1), xOpt(2), xOpt(3), xOpt(4), fval, output.fsd, exitflag, output.iterations, 'VariableNames', Rslts.Properties.VariableNames);
    end
    Rslts = [Rslts; new_row];
    writetable(Rslts, fullfile(Fitdir, 'BestRslts.txt'), 'Delimiter', '\t');
    fprintf('Model %i, nll = %f\n', modeli, fval);
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
delta = x(2); % late noise difference between time-pressure conditions
data = dat(:, {'V1', 'V2', 'V3', 'sdV1','sdV2','sdV3','chosenItem','TimeConstraint'});
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
    SVs = samples/Mp + gpuArray.randn(size(samples)).*(1 + delta*repmat(data.TimeConstraint'==1.5,num_samples,1,3))*eta;
    choice = gpuArray(data.chosenItem');
else
    SVs = samples/Mp + randn(size(samples)).*(1 + delta*repmat(data.TimeConstraint'==1.5,num_samples,1,3))*eta;
    choice = data.chosenItem';
end
max_from_each_distribution = SVs == max(SVs, [], 3);
tmp = squeeze(sum(max_from_each_distribution, 1));
probs = tmp ./ sum(tmp,2);
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
delta = x(2); % late noise difference between time-pressure conditions
scl = x(3); % scaling parameter on early noise
data = dat(:, {'V1', 'V2', 'V3', 'sdV1','sdV2','sdV3','chosenItem','TimeConstraint'});
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
if gpuparallel
    SVs = samples/Mp + gpuArray.randn(size(samples)).*(1 + delta*repmat(data.TimeConstraint'==1.5,num_samples,1,3))*eta;
    choice = gpuArray(data.chosenItem');
else
    SVs = samples/Mp + randn(size(samples)).*(1 + delta*repmat(data.TimeConstraint'==1.5,num_samples,1,3))*eta;
    choice = data.chosenItem';
end
max_from_each_distribution = SVs == max(SVs, [], 3);
tmp = squeeze(sum(max_from_each_distribution, 1));
probs = tmp ./ sum(tmp,2);
nll = -sum(log(max(probs(sub2ind(size(probs), 1:size(probs, 1), choice)), eps)));
if gpuparallel
    nll = gather(nll);
end
end

function nll = DNM(x, dat)
if gpuDeviceCount > 0
    gpuparallel = 1;
else
    gpuparallel = 0;
end
Mp = x(1); % change M to be M', absorbing the magnitude of late noise
eta = 1; % after the transformation, the late noise term is standardized as 1
delta = x(2); % late noise difference between time-pressure conditions
wp = x(3); % do the same transformation on w
data = dat(:, {'V1', 'V2', 'V3', 'sdV1','sdV2','sdV3','chosenItem','TimeConstraint'});
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
    SVs = DNP + gpuArray.randn(size(samples)).*(1 + delta*repmat(data.TimeConstraint'==1.5,num_samples,1,3))*eta;
    choice = gpuArray(data.chosenItem');
else
    SVs = DNP + randn(size(samples)).*(1 + delta*repmat(data.TimeConstraint'==1.5,num_samples,1,3))*eta;
    choice = data.chosenItem';
end
max_from_each_distribution = SVs == max(SVs, [], 3);
tmp = squeeze(sum(max_from_each_distribution, 1));
probs = tmp ./ sum(tmp,2);
nll = -sum(log(max(probs(sub2ind(size(probs), 1:size(probs, 1), choice)), eps)));
if gpuparallel
    nll = gather(nll);
end
end

function nll = dnDNM(x, dat) % cut inputs, independent
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
delta = x(2); % late noise difference between time-pressure conditions
wp = x(3); % do the same transformation on w
scl = x(4); % scaling parameter on the early noise
data = dat(:, {'V1', 'V2', 'V3', 'sdV1','sdV2','sdV3','chosenItem','TimeConstraint'});
num_samples = 20000;
Ntrl = size(dat,1);
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

D1 = sum(D1, 3)*wp + Mp;
D2 = sum(D2, 3)*wp + Mp;
D3 = sum(D3, 3)*wp + Mp;
% The product of divisive normalization before adding late noise
DNP = samples./cat(3, D1, D2, D3);
clear D1 D2 D3;
if gpuparallel
    SVs = DNP + gpuArray.randn(size(samples)).*(1 + delta*repmat(data.TimeConstraint'==1.5,num_samples,1,3))*eta;
    choice = gpuArray(data.chosenItem');
else
    SVs = DNP + randn(size(samples)).*(1 + delta*repmat(data.TimeConstraint'==1.5,num_samples,1,3))*eta;
    choice = data.chosenItem';
end
max_from_each_distribution = SVs == max(SVs, [], 3);
tmp = squeeze(sum(max_from_each_distribution, 1));
probs = tmp ./ sum(tmp,2);
nll = -sum(log(max(probs(sub2ind(size(probs), 1:size(probs, 1), choice)), eps)));
if gpuparallel
    nll = gather(nll);
end
end

