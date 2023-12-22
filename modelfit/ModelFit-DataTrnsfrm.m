% Example usage
mean1 = 0.5;
mean2 = 0.3;
mean3 = 0.2;
sd1 = 0.1;
sd2 = 0.1;
sd3 = 0.1;
eta = 0.2;
Mp = 1;
wp = 0.5;

%df = load_data('C:\Users\Bo\Dropbox (NYU Langone Health)\CESS-Bo\TaskProgram\log\txtDat', '*.txt');
probs_McFadden = McFadden(mean1, mean2, mean3, eta)
probs_Mdl2 = Mdl2(mean1, mean2, mean3, sd1, sd2, sd3, eta)
probs_DN = DN(mean1, mean2, mean3, Mp, wp)
probs_dDN_cutoff = dDN(mean1, mean2, mean3, sd1, sd2, sd3, Mp, wp, 'cutoff')
probs_dDN_absorb = dDN(mean1, mean2, mean3, sd1, sd2, sd3, Mp, wp, 'absorb')
probs_dDNb_cutoff = dDNb(mean1, mean2, mean3, sd1, sd2, sd3, Mp, wp, 'cutoff')
probs_dDNb_absorb = dDNb(mean1, mean2, mean3, sd1, sd2, sd3, Mp, wp, 'absorb')

%%
% Switch to the working directory
cd 'C:\Users\Bo\PycharmProjects\NoiseProject'

% Define I/O directories
datadir = 'C:\Users\Bo\Dropbox (NYU Langone Health)\CESS-Bo\TaskProgram\log\txtDat';
svdir = 'C:\Users\Bo\Dropbox (NYU Langone Health)\CESS-Bo\pyResults';

bd = load_data(datadir, 'BidTask_22*');
bd.Definitive = (bd.Group == bd.patch);
bd.Definitive = double(bd.Definitive);
% Rename the variables
bd = renamevars(bd, {'item'}, {'Item'});
disp(head(bd));

%%
% Raw scale of bid value
bdw = unstack(bd(:,[1,2,4,6,7,8,14]), 'bid', 'bid_times');
bdw.BidMean = mean(bdw(:, ["x1", "x2", "x3"]).Variables, 2);
bdw.BidSd = std(bdw(:, ["x1", "x2", "x3"]).Variables, 0, 2);
bdw.sd12 = std(bdw(:, ["x1", "x2"]).Variables, 0, 2);
bdw.sd23 = std(bdw(:, ["x2", "x3"]).Variables, 0, 2);
bdw.sd13 = std(bdw(:, ["x1", "x3"]).Variables, 0, 2);
bdw.cv = bdw.BidSd ./ bdw.BidMean;
disp(head(bdw));
%% To load choice data
mt = load_data(datadir, 'MainTask_22*');
mt.Vaguenesscode = mt.Vaguenesscode - 1;
mt.Definitive = 1 - mt.Vaguenesscode;
disp(head(mt));
%%
% To merge the bidding variance information into the choice matrix
IDs = {'ID1', 'V1'; 'ID2', 'V2'; 'ID3', 'V3'};
for i = 1:size(IDs, 1)
    ID = IDs{i, 1};
    V = IDs{i, 2};
    mt.Item = mt.(ID);
    tmp = innerjoin(bdw(:, {'subID', 'Item', 'BidSd'}), mt(:, {'subID', 'Item', 'trial'}));
    tmpp = renamevars(tmp, 'BidSd', ['sd' V]);
    mt = innerjoin(mt, tmpp(:, {'subID', 'trial', ['sd' V]}));
end
mt = removevars(mt, 'Item');
mt = rmmissing(mt, 'DataVariables', 'chosenItem');
mt.chosenItem = int32(mt.chosenItem);
disp(head(mt));
%% Maximum likelihood fitting to the choice behavior
AnalysName = 'ModelFitting_Mtlb';
if ~exist(fullfile(svdir, AnalysName), 'dir')
    mkdir(fullfile(svdir, AnalysName));
end
addpath(genpath('C:\Users\Bo\Documents\bads'));
options = optimset('Display', 'iter', 'MaxFunEvals', 3000, 'TolFun', 0.1, 'MaxIter', 1000);
sublist = unique(mt.subID);
Rslts = table('Size', [0 9], 'VariableTypes', {'double', 'string', 'double', 'double', 'double', 'double', 'double', 'logical', 'double'}, 'VariableNames', {'subID', 'Model', 'eta', 'Mp', 'wp', 'nll', 'nllsd', 'success', 'iterations'});
Npar = 4;
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
        if modeli <= 2
            LB = 0;
            UB = 1000;
            PLB = 1.4;
            PUB = 100;
        else
            LB = [0; -1];
            UB = [1000; 1];
            PLB = [1.4; 0.1];
            PUB = [100; 0.5];
        end
        nLL = [];
        params = {};
        success = [];
        res = {};
        parfor i = 1:Npar
            x0 = PLB + (PUB - PLB) .* rand(size(PLB));
            [xOpt, fval, exitflag, output] = fmincon(nLLfunc, x0, [],[],[],[], LB, UB,[],options);
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

        filename = fullfile(svdir, AnalysName, sprintf('Model%s', model), sprintf('fmincon_%s.mat', sublist(subj)));
        save(filename, 'xOpt', 'fval', 'exitflag', 'output');
        if modeli <= 2
            new_row = table(sublist(subj), {model}, xOpt, NaN, NaN, fval, NaN, exitflag, output.iterations, 'VariableNames', Rslts.Properties.VariableNames);
        elseif modeli >= 3
            new_row = table(sublist(subj), {model}, 1, xOpt(1), xOpt(2), fval, NaN, exitflag, output.iterations, 'VariableNames', Rslts.Properties.VariableNames);
        end
        Rslts = [Rslts; new_row];
        fprintf('Model %s nll=%f\t', model, fval);
        writetable(Rslts, fullfile(svdir, AnalysName, 'Rslts_fmincon.txt'), 'Delimiter', '\t');
    end
    %     % Model 2
    %     nLLfunc = @(x) neg_ll_indv2(x, dat);
    %     LB = [0];
    %     UB = [1000];
    %     PLB = [1.4];
    %     PUB = [100];
    %     x0 = PLB + (PUB - PLB) * rand;
    %     [xOpt, fval, exitflag, output] = fmincon(nLLfunc, x0, [],[],[],[], LB, UB,[],options);
    %     filename = fullfile(svdir, AnalysName, 'Model2', sprintf('fmincon_%s.mat', sublist(subj)));
    %     save(filename, 'xOpt', 'fval', 'exitflag', 'output');
    %     new_row = table(sublist(subj), {'2'}, xOpt, NaN, NaN, fval, NaN, exitflag, output.iterations, 'VariableNames', Rslts.Properties.VariableNames);
    %     Rslts = [Rslts; new_row];
    %     fprintf('Model 2 nll=%f\t', fval);
    %
    %
    %     % Model 3
    %     nLLfunc = @(y) neg_ll_indv3(y, dat);
    %     LB = [0; -1];
    %     UB = [1000; 1];
    %     PLB = [1.4; 0.1];
    %     PUB = [100; 0.5];
    %     x0 = PLB + (PUB - PLB) .* rand(2, 1);
    %     [xOpt, fval, exitflag, output] = fmincon(nLLfunc, x0, [],[],[],[], LB, UB,[],options);
    %     filename = fullfile(svdir, AnalysName, 'Model3', sprintf('fmincon_%s.mat', sublist(subj)));
    %     save(filename, 'xOpt', 'fval', 'exitflag', 'output');
    %     new_row = table(sublist(subj), {'3'}, 1, xOpt(1), xOpt(2), fval, NaN, exitflag, output.iterations, 'VariableNames', Rslts.Properties.VariableNames);
    %     Rslts = [Rslts; new_row];
    %     fprintf('Model 3 nll=%f\t', fval);
    %
    %     % Model 4
    %     nLLfunc = @(y) neg_ll_indv4(y, dat);
    %     LB = [0; -1];
    %     UB = [1000; 1];
    %     PLB = [1.4; 0.1];
    %     PUB = [100; 0.5];
    %     x0 = PLB + (PUB - PLB) .* rand(2, 1);
    %     [xOpt, fval, exitflag, output] = fmincon(nLLfunc, x0, [],[],[],[], LB, UB,[],options);
    %     filename = fullfile(svdir, AnalysName, 'Model4', sprintf('fmincon_%s.mat', sublist(subj)));
    %     save(filename, 'xOpt', 'fval', 'exitflag', 'output');
    %     new_row = table(sublist(subj), {'4'}, 1, xOpt(1), xOpt(2), fval, NaN, exitflag, output.iterations, 'VariableNames', Rslts.Properties.VariableNames);
    %     Rslts = [Rslts; new_row];
    %     fprintf('Model 4 nll=%f\t', fval);
    %
    %     % Model 4b
    %     nLLfunc = @(y) neg_ll_indv4b(y, dat);
    %     LB = [0; -1];
    %     UB = [1000; 1];
    %     PLB = [1.4; 0.1];
    %     PUB = [100; 0.5];
    %     x0 = PLB + (PUB - PLB) .* rand(2, 1);
    %     [xOpt, fval, exitflag, output] = fmincon(nLLfunc, x0, [],[],[],[], LB, UB,[],options);
    %     filename = fullfile(svdir, AnalysName, 'Model4b', sprintf('fmincon_%s.mat', sublist(subj)));
    %     save(filename, 'xOpt', 'fval', 'exitflag', 'output');
    %     new_row = table(sublist(subj), {'4b'}, 1, xOpt(1), xOpt(2), fval, NaN, exitflag, output.iterations, 'VariableNames', Rslts.Properties.VariableNames);
    %     Rslts = [Rslts; new_row];
    %     fprintf('Model 4b nll=%f\t', fval);
    subj = subj + 1;
end

writetable(Rslts, fullfile(svdir, AnalysName, 'Rslts_fmincom.txt'), 'Delimiter', '\t');

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
