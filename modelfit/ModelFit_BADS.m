%% define directoriress
% Switch to the working directory
rtdir = '/gpfs/data/glimcherlab/BoShen/Noise/modelfit';
cd(rtdir);

% Define I/O directories
datadir = rtdir;
svdir = fullfile(rtdir, 'Results');
if ~exist(svdir, 'dir')
    mkdir(svdir);
end
% load packages
addpath(genpath('/gpfs/data/glimcherlab/BoShen/bads'));
%% load data
load(fullfile(datadir, 'TrnsfrmData.mat'));
%disp(mt);

%% Maximum likelihood fitting to the choice behavior
AnalysName = 'BADS_Mtlb';
if ~exist(fullfile(svdir, AnalysName), 'dir')
    mkdir(fullfile(svdir, AnalysName));
end

options = bads('defaults');     % Default options
options.Display = 'None';
options.UncertaintyHandling = true;    %s Function is stochastic
options.NoiseFinalSamples = 30;
sublist = unique(mt.subID);
Rslts = table('Size', [0 9], 'VariableTypes', {'double', 'string', 'double', 'double', 'double', 'double', 'double', 'logical', 'double'}, 'VariableNames', {'subID', 'Model', 'eta', 'Mp', 'wp', 'nll', 'nllsd', 'success', 'iterations'});
testfile = fullfile(svdir, AnalysName, 'Rslts_BADS_Test.txt');
fp = fopen(testfile, 'w+');
fprintf(fp, '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n', 'subID', 'Model', 'randi', 'eta', 'Mp', 'wp', 'nll', 'nllsd', 'success', 'iterations');
fclose(fp);
Npar = 40;
mypool = parpool(Npar);
subj = 1;
while subj <= length(sublist)
    fprintf('Subject %d:\t', subj);
    dat = mt(mt.subID == sublist(subj), :);
    for modeli = 3:5
        switch modeli
            case 1
                model = '1';
                nLLfunc = @(x) neg_ll_indv1(x, dat);
            case 2
                model = '2';
                nLLfunc = @(x) neg_ll_indv2(x, dat);
            case 3
                model = '3';
                nLLfunc = @(y) neg_ll_indv3(y, dat);
            case 4
                model = '4';
                nLLfunc = @(y) neg_ll_indv4(y, dat);
            case 5
                model = '4b';
                nLLfunc = @(y) neg_ll_indv4b(y, dat);
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
function nll = neg_ll_indv4b(y, dat)
% Let's assume the bid value with bid variance is divisively normalized
% there is still a fixed noise during the selection process
% so that we can calculate the probability of choosing one from the other two options
% by computing the probability of the largest value of the randomly drawn is from which option
% y(1) = Mp, y(2) = wp
mode = 'cutoff';
data = dat(:, {'V1', 'V2', 'V3', 'sdV1', 'sdV2', 'sdV3'});
probs = zeros(size(data, 1), 1);

for i = 1:size(data, 1)
    row = data(i, :);
    prob = dDNb(row.V1, row.V2, row.V3, row.sdV1, row.sdV2, row.sdV3, y(1), y(2), mode);
    probs(i) = max(prob(dat.chosenItem(i)), eps);
end

ll = sum(log(probs));
nll = gather(-ll);
end

%%
function nll = neg_ll_indv4(y, dat)
% Let's assume the bid value with bid variance is divisively normalized
% there is still a fixed noise during the selection process
% so that we can calculate the probability of choosing one from the other two options
% by computing the probability of the largest value of the randomly drawn is from which option
% y(1) = Mp, y(2) = wp
mode = 'cutoff';
data = dat(:, {'V1', 'V2', 'V3', 'sdV1', 'sdV2', 'sdV3'});
probs = zeros(size(data, 1), 1);

for i = 1:size(data, 1)
    row = data(i, :);
    prob = dDN(row.V1, row.V2, row.V3, row.sdV1, row.sdV2, row.sdV3, y(1), y(2), mode);
    probs(i) = max(prob(dat.chosenItem(i)), eps);
end

ll = sum(log(probs));
nll = gather(-ll);
end

%%
function nll = neg_ll_indv3(y, dat)
% Let's assume the mean value of the bid is divisively normalized
% there is still a fixed noise during the selection process
% so that we can calculate the probability of choosing one from the other two options
% by computing the probability of the largest value of the randomly drawn is from which option
% y(1) = Mp, y(2) = wp

data = dat(:, {'V1', 'V2', 'V3'});
probs = zeros(size(data, 1), 1);

for i = 1:size(data, 1)
    row = data(i, :);
    prob = DN(row.V1, row.V2, row.V3, y(1), y(2));
    probs(i) = max(prob(dat.chosenItem(i)), eps);
end

ll = sum(log(probs));
nll = gather(-ll);
end
%%
function nll = neg_ll_indv2(x, dat)
% Let's assume the bid value has noise as measured in bid variance
% there is still a fixed noise during the selection process
% so that we can calculate the probability of choosing one from the other two options
% by computing the probability of the largest value of the randomly drawn is from which option

data = dat(:, {'V1', 'V2', 'V3', 'sdV1', 'sdV2', 'sdV3'});
eta = x(1);
probs = zeros(size(data, 1), 1);

for i = 1:size(data, 1)
    row = data(i, :);
    prob = Mdl2(row.V1, row.V2, row.V3, row.sdV1, row.sdV2, row.sdV3, eta);
    probs(i) = max(prob(dat.chosenItem(i)), eps);
end

ll = sum(log(probs));
nll = -ll;
end
%%
function nll = neg_ll_indv1(x, dat)
% Let's assume the value of each item has no noise
% but there is a fixed noise during the selection process
% so that we can calculate the probability of choosing one from the other two options
% by computing the probability of the largest value of the randomly drawn is from which option

data = dat(:, {'V1', 'V2', 'V3'});
eta = x(1);
probs = zeros(size(data,1),1);

for i = 1:size(data, 1)
    row = data(i, :);
    prob = McFadden(row.V1, row.V2, row.V3, eta);
    probs(i) = max(prob(dat.chosenItem(i)), eps);
end

ll = sum(log(probs));
nll = -ll;
end

%%
function df = load_data(directory, file_pattern)
file_paths = dir(fullfile(directory, file_pattern));
data = cell(numel(file_paths), 1);

for i = 1:numel(file_paths)
    file_path = fullfile(file_paths(i).folder, file_paths(i).name);
    file_data = readtable(file_path, 'Delimiter', '\t');
    data{i} = file_data;
end

df = vertcat(data{:});
end

function probs = McFadden(mean1, mean2, mean3, eta)
show = false;

options = [mean1, eta/(2^0.5); mean2, eta/(2^0.5); mean3, eta/(2^0.5)];
num_samples = 100000;
for i = 1:3
    SVs(:, i) = normrnd(options(i, 1), eta, [num_samples, 1]);
end

[~, max_indices] = max(SVs, [], 2);
max_from_each_distribution = zeros(size(SVs));
max_from_each_distribution(sub2ind(size(SVs), 1:size(SVs, 1), max_indices')) = 1;
probs = sum(max_from_each_distribution, 1) / size(SVs, 1);

if show
    disp(['Probability that a variable drawn from sample 1 is: ', num2str(probs(1))]);
    disp(['Probability that a variable drawn from sample 2 is: ', num2str(probs(2))]);
    disp(['Probability that a variable drawn from sample 3 is: ', num2str(probs(3))]);
end
end

function probs = Mdl2(mean1, mean2, mean3, sd1, sd2, sd3, eta)
show = false;

num_samples = 100000;
options = [mean1, sd1; mean2, sd2; mean3, sd3];
for i = 1:3
    SVs(:, i) = normrnd(options(i, 1), sqrt(options(i, 2)^2+eta^2), [num_samples, 1]);
end

[~, max_indices] = max(SVs, [], 2);
max_from_each_distribution = zeros(size(SVs));
max_from_each_distribution(sub2ind(size(SVs), 1:size(SVs, 1), max_indices')) = 1;
probs = sum(max_from_each_distribution, 1) / size(SVs, 1);

if show
    disp(['Probability that a variable drawn from sample 1 is: ', num2str(probs(1))]);
    disp(['Probability that a variable drawn from sample 2 is: ', num2str(probs(2))]);
    disp(['Probability that a variable drawn from sample 3 is: ', num2str(probs(3))]);
end
end

function probs = DN(mean1, mean2, mean3, Mp, wp)
show = false;
D = (mean1 + mean2 + mean3)*wp + Mp;
options = [mean1/D, 2^(-0.5); mean2/D, 2^(-0.5); mean3/D, 2^(-0.5)];
num_samples = 100000;
% for i = 1:3
%     SVs(:, i) = options(i, 1) / (Mp + wp * (mean1 + mean2 + mean3)) + normrnd(0, options(i, 2), [num_samples, 1]);
% end
SVs = [gpuArray.randn([num_samples, 1])*options(1,2)+options(1,1), gpuArray.randn([num_samples, 1])*options(2,2)+options(2,1), gpuArray.randn([num_samples, 1])*options(3,2)+options(3,1)];
% SVs = [gpuArray.randn([num_samples, 1])*(sqrt(1/2))+options(1,1), gpuArray.randn([num_samples, 1])*(sqrt(1/2))+options(2,1), gpuArray.randn([num_samples, 1])*(sqrt(1/2))+options(3,1)];

[~, max_indices] = max(SVs, [], 2);
% probs = accumarray(max_indices, 1)/length(max_indices);
max_from_each_distribution = gpuArray.zeros(size(SVs));
max_from_each_distribution(sub2ind(size(SVs), 1:size(SVs, 1), max_indices')) = 1;
probs = sum(max_from_each_distribution, 1) / size(SVs, 1);

if show
    disp(['Probability that a variable drawn from sample 1 is: ', num2str(probs(1))]);
    disp(['Probability that a variable drawn from sample 2 is: ', num2str(probs(2))]);
    disp(['Probability that a variable drawn from sample 3 is: ', num2str(probs(3))]);
end
end

function probs = dDN(mean1, mean2, mean3, sd1, sd2, sd3, Mp, wp, mode)
show = false;

num_samples = 100000;
options = [mean1, sd1; mean2, sd2; mean3, sd3];
% for i = 1:3
%     samples(:, i) = normrnd(options(i, 1), options(i, 2), [num_samples, 1]);
% end
samples = [gpuArray.randn([num_samples, 1])*options(1,2)+options(1,1), gpuArray.randn([num_samples, 1])*options(2,2)+options(2,1), gpuArray.randn([num_samples, 1])*options(3,2)+options(3,1)];
if strcmp(mode, 'cutoff')
    tmp1=samples(samples(:,1)>0, 1);
    tmp2=samples(samples(:,2)>0, 2);
    tmp3=samples(samples(:,3)>0, 3);
    len = min([length(tmp1),length(tmp2),length(tmp3)]);
    samples = [tmp1(1:len), tmp2(1:len), tmp3(1:len)];
elseif strcmp(mode, 'absorb')
    samples = max(samples, zeros(size(samples)));
end
D = sum(samples, 2)*wp + Mp;
SVs = samples./D + gpuArray.randn(size(samples))*(2^-0.5);

[~, max_indices] = max(SVs, [], 2);
max_from_each_distribution = gpuArray.zeros(size(SVs));
max_from_each_distribution(sub2ind(size(SVs), 1:size(SVs, 1), max_indices')) = 1;
probs = sum(max_from_each_distribution, 1) / size(SVs, 1);

if show
    disp(['Probability that a variable drawn from sample 1 is: ', num2str(probs(1))]);
    disp(['Probability that a variable drawn from sample 2 is: ', num2str(probs(2))]);
    disp(['Probability that a variable drawn from sample 3 is: ', num2str(probs(3))]);
end
end

function probs = dDNb(mean1, mean2, mean3, sd1, sd2, sd3, Mp, wp, mode)
show = false;

num_samples = 100000;
options = [mean1, sd1; mean2, sd2; mean3, sd3];
samples = [gpuArray.randn([num_samples, 1])*options(1,2)+options(1,1), gpuArray.randn([num_samples, 1])*options(2,2)+options(2,1), gpuArray.randn([num_samples, 1])*options(3,2)+options(3,1)];
samplesD = [gpuArray.randn([num_samples, 1])*options(1,2)+options(1,1), gpuArray.randn([num_samples, 1])*options(2,2)+options(2,1), gpuArray.randn([num_samples, 1])*options(3,2)+options(3,1)];
SIGMA = sum(samplesD, 2);

if strcmp(mode, 'cutoff')
    SIGMA = SIGMA(SIGMA > 0);
    samples = samples(1:length(SIGMA),:);
elseif strcmp(mode, 'absorb')
    SIGMA = max(SIGMA, 0);
end

D = SIGMA * wp + Mp;
SVs = samples./D + gpuArray.randn(size(samples))*(2^-0.5);

[~, max_indices] = max(SVs, [], 2);
max_from_each_distribution = gpuArray.zeros(size(SVs));
max_from_each_distribution(sub2ind(size(SVs), 1:size(SVs, 1), max_indices')) = 1;
probs = sum(max_from_each_distribution, 1) / size(SVs, 1);

if show
    disp(['Probability that a variable drawn from sample 1 is: ', num2str(probs(1))]);
    disp(['Probability that a variable drawn from sample 2 is: ', num2str(probs(2))]);
    disp(['Probability that a variable drawn from sample 3 is: ', num2str(probs(3))]);
end
end
