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
AnalysName = 'FastBADS_FixMwsMean'; % fix M, w and scaling on early noise, and fit to the mean choice accuracy of each ID3 individually.
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
Rslts = table('Size', [0 12], 'VariableTypes', {'double', 'double', 'double', 'string', 'double', 'double', 'double', 'double', 'double', 'double', 'logical', 'double'},...
    'VariableNames', {'subID', 'TimeConstraint', 'modeli', 'name', 'scl', 'eta', 'Mp', 'wp', 'nll', 'nllsd', 'success', 'iterations'});
testfile = fullfile(svdir, AnalysName, 'AllRslt.txt');
fp = fopen(testfile, 'w+');
fprintf(fp, '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n', 'subID', 'TimeConstraint', 'Model', 'randi', 'scl0', 'eta0', 'Mp0', 'wp0', 'scl', 'eta', 'Mp', 'wp', 'nll', 'nllsd', 'success', 'iterations');
fclose(fp);
Npar = 40;
mypool = parpool(Npar);
sublist = unique(mt.subID);
subj = 1;
while subj <= length(sublist)
    %%
    fprintf('Subject %d:\t', subj);
    for t = [10, 1.5]
        fprintf('TimeConstraint %1.1f:\t', t);
        dat = mt(mt.subID == sublist(subj) & mt.TimeConstraint == t, :);
        for modeli = 1
            nLLfunc = @(x) dDNb(x, dat, mode);
            name = 'dDNb'; %, cut input, independent';
            fprintf('Model %i nll=', modeli);

            LB = [1, 0];
            UB = [1, 2];
            PLB = [1, .1];
            PUB = [1, .4];

            nLL = [];
            params = {};
            success = [];
            res = {};
            parfor i = 1:Npar
                x0 = PLB + (PUB - PLB) .* rand(size(PLB));
                [xOpt,fval,exitflag,output] = bads(nLLfunc,x0,LB,UB,PLB,PUB,[],options);

                dlmwrite(testfile, [subj, t, modeli, i, x0(1), x0(2), 1, 1, xOpt(1), xOpt(2), 1, 1, fval, output.fsd, exitflag, output.iterations],'delimiter','\t','precision','%.6f','-append');

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
            filename = fullfile(mtrxdir, sprintf('Subj%02i_TC%1.1f_Mdl%i.mat', subj, t, modeli));
            save(filename, 'xOpt', 'fval', 'exitflag', 'output');
            new_row = table(subj, t, modeli, {name}, xOpt(1), xOpt(2), 1, 1, fval, output.fsd, exitflag, output.iterations, 'VariableNames', Rslts.Properties.VariableNames);
            Rslts = [Rslts; new_row];
            writetable(Rslts, fullfile(outdir, 'Best.txt'), 'Delimiter', '\t');
            fprintf('%f\t', fval);
        end
    end
    subj = subj + 1;
end

%% define the cost function
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
num_samples = 20000;
Ntrl = size(dat,1);
if strcmp(mode, 'absorb')
    samples = [];
    for ci = 1:3
        if gpuparallel
            values = gpuArray(dat.(['V',num2str(ci)])');
            stds = gpuArray(dat.(['sdV', num2str(ci)])')*scl;
            samples(ci,:,:) = max(gpuArray.randn([num_samples, Ntrl]).*stds + repmat(values, num_samples, 1), 0);
        else
            values = dat.(['V',num2str(ci)])';
            stds = dat.(['sdV', num2str(ci)])'*scl;
            samples(ci,:,:) = max(randn([num_samples, Ntrl]).*stds + repmat(values, num_samples, 1), 0);
        end
    end
    D1 = [];
    D2 = [];
    D3 = [];
    for ci = 1:3
        if gpuparallel
            values = gpuArray(dat.(['V',num2str(ci)])');
            stds = gpuArray(dat.(['sdV', num2str(ci)])')*scl;
            D1(ci,:,:) = max(gpuArray.randn([num_samples, Ntrl]).*stds + repmat(values, num_samples, 1), 0);
            D2(ci,:,:) = max(gpuArray.randn([num_samples, Ntrl]).*stds + repmat(values, num_samples, 1), 0);
            D3(ci,:,:) = max(gpuArray.randn([num_samples, Ntrl]).*stds + repmat(values, num_samples, 1), 0);
        else
            values = dat.(['V',num2str(ci)])';
            stds = dat.(['sdV', num2str(ci)])'*scl;
            D1(ci,:,:) = max(randn([num_samples, Ntrl]).*stds + repmat(values, num_samples, 1), 0);
            D2(ci,:,:) = max(randn([num_samples, Ntrl]).*stds + repmat(values, num_samples, 1), 0);
            D3(ci,:,:) = max(randn([num_samples, Ntrl]).*stds + repmat(values, num_samples, 1), 0);
        end
    end
elseif strcmp(mode, 'cutoff')
    error('The cutoff boundary algorithm has not been developped yet.');
end
D1 = sum(D1, 1)*wp + Mp;
D2 = sum(D1, 1)*wp + Mp;
D3 = sum(D1, 1)*wp + Mp;
D = [D1; D2; D3];
clear D1 D2 D3;
if gpuparallel
    SVs = samples./D + gpuArray.randn(size(samples))*eta;
    %choice = gpuArray(data.chosenItem');
else
    SVs = samples./D + randn(size(samples))*eta;
    %choice = data.chosenItem';
end
max_from_each_distribution = SVs == max(SVs, [], 1);
probs = squeeze(sum(max_from_each_distribution, 2) / size(SVs, 2));
dat.modelprob1 = probs(1,:)';
dat.modelprob2 = probs(2,:)';
dat.modelprob3 = probs(3,:)';
dat.choice = dat.chosenItem - 1;
dat.ratio = dat.modelprob2./(dat.modelprob1 + dat.modelprob2);

GrpMean = grpstats(dat(dat.chosenItem ~= 3,:), "ID3", "mean", "DataVars", ["choice", "ratio"]);
observedProbs = GrpMean.mean_choice;
predictedProbs = GrpMean.mean_ratio;
N = GrpMean.GroupCount;
% Sum over negative binomial log-likelihood for each ID3
nll = -sum(log(max(binopdf(round(observedProbs .* N), N, predictedProbs), eps)));
if gpuparallel
    nll = gather(nll);
end
end
